#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F-A+B 全任务链路启动器（task1-4 官方实验流程口径）。

- F-A：功能保持损失（module F / functional_fibre_loss_v2），由 --fotg_on 1 接入。
- F-B：Jacobian 软投影（nullspace_projection），由 --fab_on 1 在 engine.py 的
  backward 阶段把特征空间梯度投影到旧读出雅可比零空间。

本启动器严格复刻 configs/from-scratch/o1o_mowod_task{1-4}_{train,ft}.sh 的超参
（PREV/CUR/superclass/batch/lr/epochs/lr_drop/resume），仅在官方参数之外追加
F-A+B 开关，并把每个任务的输出目录命名空间化为 exps/fab_*，使其与已有 baseline
运行隔离、互不污染。

用法：
    python -u owod_run_fab.py --all --gpu 5
        # 从 task1_train 起，逐阶段串行训练，每阶段结束自动拉起下一阶段（自链）。
    python -u owod_run_fab.py --stage 2 --gpu 5
        # 只跑单个阶段（0-based：0=t1_train,1=t2_train,2=t2_ft,3=t3_train,
        # 4=t3_ft,5=t4_train,6=t4_ft）。
"""
import os
import sys
import time
import argparse
import subprocess

os.environ.setdefault("TMPDIR", "/dev/shm/tmp")
os.environ["PYTHONNOUSERSITE"] = "1"
os.environ["WANDB_MODE"] = "offline"          # 无 API key 直接崩

PROJ = "/icislab/volume1/wyy/O1O-main"
# 输出根目录：/dev/sda2（volume1/volume2）已 100% 满，checkpoint 无法落盘。
# 全部 fab_* 输出与 checkpoint 链重定向到 /dev/shm（tmpfs，70G 可用），
# 与既有 fotg_t2_* 目录同处。注意 /dev/shm 重启即清空，长跑需在每阶段后手动归档。
FAB_OUT_ROOT = "/dev/shm/fab_chain"
SUPERVISOR_LOG = "/dev/shm/fab_chain_supervisor.log"

# F-A+B 固定开关（F 模块开、O/T/U 关，λ_fib 用标定初值 0.0637，对偶上升关；
# F-B 软投影开）。这些是训练内部 flag，不在命令行暴露可调（保持与 F-only 协议一致）。
FAB_FLAGS = [
    "--fotg_on", "1",
    "--fotg_dim", "256",
    "--fotg_k_sub", "8",
    "--fotg_w_threshold", "0.0",
    "--fotg_w_topk", "10",
    "--fotg_w_soft", "1",
    "--fotg_lam_fib", "0.0637",
    "--fotg_lam_ov", "0",
    "--fotg_lam_topo", "0",
    "--fotg_dual", "0",
    "--fotg_eps_fib", "1e-3",
    "--fotg_eps_ov", "1e9",
    "--fotg_eps_topo", "1e9",
    "--fotg_lam_route", "0.1",
    "--fotg_eps_route", "0.05",
    "--fotg_lam_unit_route", "1.0",
    "--fotg_dual_lr_route", "1.0",
    "--fotg_lam_max_route", "1e3",
    "--fotg_unk_w", "0",
    "--fotg_dual_every", "20",
    "--fotg_accum_every", "20",
    # --fab_on / --fab_lambda 不写死，由 build_argv 按 CLI 覆盖拼接（F-A=0 / F-B=1）。
]

# MFFC-F 固定开关（MFFC-F-only 运行口径）：F 模块用 MFFC-F 替换 F-A/F-B/PCA-FAB。
# 仅把 teacher 加载好（main 里由 --mffc_f_on 一并构建 criterion.fotg），其余 fotg 原约束
# 全部置 0（lam_fib/ov/topo/route/unk=0、dual=0、accum_every=0），使原 fotg 自身损失归零、
# 只有 MFFC-F 的 F1/F2/F3(+可选 L_fib) 生效。F-B/PCA-FAB 投影也保持关闭。
MFFC_FLAGS = [
    "--fotg_on", "0",
    "--fotg_lam_fib", "0",
    "--fotg_lam_ov", "0",
    "--fotg_lam_topo", "0",
    "--fotg_lam_route", "0",
    "--fotg_dual", "0",
    "--fotg_accum_every", "0",
    "--fotg_unk_w", "0",
    "--margin_on", "0",
]

# 官方 task1-4 链路（严格对齐 configs/from-scratch/*.sh）。
# resume / old_ckpt 均为相对 PROJ 的路径；epochs 为全局 END-epoch（resume 续跑）。
STAGES = [
    dict(name="task1_train", train_set="owod_t1_train", PREV=0, CUR=20,
         psc=0, csc=6, nsc=13, batch=4, lr=2e-4, epochs=30, lr_drop=17,
         eval_every=1, resume="", old_ckpt="", ft=False,
         pseudo="data/OWOD/pseudo-boxes/TOWOD/mowod_until_t1_normal_pseudo_labels.json",
         out="/dev/shm/fab_chain/task1_train",
         ex_prev="", ex_cur="learned_owod_t1_ft.txt", max_len=850),
    dict(name="task2_train", train_set="owod_t2_train", PREV=20, CUR=20,
         psc=6, csc=3, nsc=13, batch=4, lr=2e-5, epochs=40, lr_drop=50,
         eval_every=1,
         resume="/icislab/volume1/wyy/O1O-main/exps/o1o_mowod_task1_train/checkpoint0029.pth",
         old_ckpt="/icislab/volume1/wyy/O1O-main/exps/o1o_mowod_task1_train/checkpoint0029.pth", ft=False,
         pseudo="data/OWOD/pseudo-boxes/TOWOD/mowod_until_t2_normal_pseudo_labels.json",
         out="/dev/shm/fab_chain/task2_train",
         ex_prev="learned_owod_t1_ft.txt", ex_cur="learned_owod_t2_ft.txt", max_len=1743),
    dict(name="task2_ft", train_set="o1o_mowod_exemplars/learned_owod_t2_ft",
         PREV=20, CUR=20, psc=6, csc=3, nsc=13, batch=4, lr=2e-4, epochs=50,
         lr_drop=50, eval_every=5,
         resume="/dev/shm/fab_chain/task2_train/checkpoint0039.pth",
         old_ckpt="/icislab/volume1/wyy/O1O-main/exps/o1o_mowod_task1_train/checkpoint0029.pth", ft=True,
         pseudo="data/OWOD/pseudo-boxes/TOWOD/mowod_until_t2_normal_pseudo_labels.json",
         out="/dev/shm/fab_chain/task2_ft", ex_prev="", ex_cur="", max_len=0),
    dict(name="task3_train", train_set="owod_t3_train", PREV=40, CUR=20,
         psc=9, csc=2, nsc=13, batch=4, lr=2e-5, epochs=70, lr_drop=80,
         eval_every=1,
         resume="/dev/shm/fab_chain/task2_ft/checkpoint0059.pth",
         old_ckpt="/dev/shm/fab_chain/task2_ft/checkpoint0059.pth", ft=False,
         pseudo="data/OWOD/pseudo-boxes/TOWOD/mowod_until_t3_normal_pseudo_labels.json",
         out="/dev/shm/fab_chain/task3_train",
         ex_prev="learned_owod_t2_ft.txt", ex_cur="learned_owod_t3_ft.txt", max_len=2361),
    dict(name="task3_ft", train_set="o1o_mowod_exemplars/learned_owod_t3_ft",
         PREV=40, CUR=20, psc=9, csc=2, nsc=13, batch=4, lr=2e-4, epochs=90,
         lr_drop=80, eval_every=5,
         resume="/dev/shm/fab_chain/task3_train/checkpoint0069.pth",
         old_ckpt="/dev/shm/fab_chain/task2_ft/checkpoint0059.pth", ft=True,
         pseudo="data/OWOD/pseudo-boxes/TOWOD/mowod_until_t3_normal_pseudo_labels.json",
         out="/dev/shm/fab_chain/task3_ft", ex_prev="", ex_cur="", max_len=0),
    dict(name="task4_train", train_set="owod_t4_train", PREV=60, CUR=20,
         psc=11, csc=1, nsc=13, batch=4, lr=2e-5, epochs=100, lr_drop=110,
         eval_every=1,
         resume="/dev/shm/fab_chain/task3_ft/checkpoint0089.pth",
         old_ckpt="/dev/shm/fab_chain/task3_ft/checkpoint0089.pth", ft=False,
         pseudo="data/OWOD/pseudo-boxes/TOWOD/mowod_until_t4_normal_pseudo_labels.json",
         out="/dev/shm/fab_chain/task4_train",
         ex_prev="learned_owod_t3_ft.txt", ex_cur="learned_owod_t4_ft.txt", max_len=2749),
    dict(name="task4_ft", train_set="owod_t4_train", PREV=60, CUR=20,
         psc=11, csc=1, nsc=13, batch=4, lr=2e-4, epochs=120,
         lr_drop=110, eval_every=5,
         resume="/dev/shm/fab_chain/task4_train/checkpoint0099.pth",
         old_ckpt="/dev/shm/fab_chain/task3_ft/checkpoint0089.pth", ft=True,
         pseudo="data/OWOD/pseudo-boxes/TOWOD/mowod_until_t4_normal_pseudo_labels.json",
         out="/dev/shm/fab_chain/task4_ft", ex_prev="", ex_cur="", max_len=0),
]


def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        with open(SUPERVISOR_LOG, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def build_argv(stage, dev, resume_override=None, old_ckpt_override=None,
               skip_resume=False, batch_size=None, num_gpus=1,
               fab_on=0, fab_lambda=1e-3, out_root=None,
               lr_scale=1.0, pca_fab_on=0, pca_dirs="", pca_basis_pt="", pca_fab_lam=1e-3, pca_fab_every=20,
               mffc_f_on=0, mffc_f_topk=5, mffc_f_iou=0.5,
               mffc_f_lam_id=1.0, mffc_f_lam_fg=1.0, mffc_f_lam_loc=1.0,
               mffc_f_lam_fib=0.0, mffc_f_w=1.0, mffc_f_loc_scale=1.0,
               mffc_f_fib_w_fg=1.0, mffc_f_fib_w_loc=1.0, mffc_f_temp=1.0):
    a = stage
    out = (out_root or "") + a["out"].lstrip("/dev/shm")
    if out_root is not None:
        out = out_root.rstrip("/") + "/" + a["out"].split("/")[-1]
    bs = a["batch"] if batch_size is None else batch_size
    lr = a["lr"] * lr_scale
    argv = [
        "--output_dir", out,
        "--data_root", "data/OWOD",
        "--model_type", "dn_dab_deformable_detr_o1o",
        "--train_set", a["train_set"],
        "--test_set", "owod_all_task_test_nodup",
        "--dataset", "TOWOD",
        "--PREV_INTRODUCED_CLS", str(a["PREV"]),
        "--CUR_INTRODUCED_CLS", str(a["CUR"]),
        "--prev_num_superclass", str(a["psc"]),
        "--cur_num_superclass", str(a["csc"]),
        "--num_superclass", str(a["nsc"]),
        "--batch_size", str(bs),
        "--eval_every", str(a["eval_every"]),
        "--lr", str(lr),
        "--epochs", str(a["epochs"]),
        "--use_dn",
        "--empty_weight", "0.0",
        "--sup_loss_coef", "2",
        "--lr_drop", str(a["lr_drop"]),
        "--lr_drop_gamma", "0.1",
        "--pseudo_path", a["pseudo"],
        "--pseudo_threshold", "0.5",
        "--pseudo_nms_iou", "0.5",
        "--num_unk_objects", "20",
        "--device", dev,
        "--num_workers", "4",
        "--sep_loss_w", "0.0",
        "--wandb_project", "",
        "--num_gpus", str(num_gpus),
    ]
    if not skip_resume:
        if resume_override:
            argv += ["--resume", resume_override]
        elif a["resume"]:
            r = a["resume"]
            if out_root is not None and r.startswith("/dev/shm/fab_chain/"):
                r = out_root.rstrip("/") + "/" + r.split("/")[-2] + "/" + r.split("/")[-1]
            argv += ["--resume", r]
    if old_ckpt_override:
        argv += ["--fotg_old_ckpt", old_ckpt_override]
    elif a["old_ckpt"]:
        o = a["old_ckpt"]
        if out_root is not None and o.startswith("/dev/shm/fab_chain/"):
            o = out_root.rstrip("/") + "/" + o.split("/")[-2] + "/" + o.split("/")[-1]
        argv += ["--fotg_old_ckpt", o]
    if not a["ft"]:
        argv += [
            "--exemplar_replay_selection",
            "--exemplar_replay_dir", "o1o_mowod_exemplars",
            "--exemplar_replay_max_length", str(a["max_len"]),
        ]
        if a["ex_prev"]:
            argv += ["--exemplar_replay_prev_file", a["ex_prev"]]
        argv += ["--exemplar_replay_cur_file", a["ex_cur"]]
    if mffc_f_on:
        # MFFC-F-only：用 MFFC_FLAGS（原 fotg 约束全置 0）替代 FAB_FLAGS，
        # 并注入 MFFC-F 各通道权重。
        argv += MFFC_FLAGS
    else:
        argv += FAB_FLAGS
    argv += ["--fab_on", str(fab_on), "--fab_lambda", str(fab_lambda)]
    # 新 F-B（PCA-directional Jacobian Protection）：仅 pca_fab_on=1 时拼参数
    if pca_fab_on:
        argv += [
            "--pca_fab_on", str(pca_fab_on),
            "--pca_fab_lam", str(pca_fab_lam),
            "--pca_fab_every", str(pca_fab_every),
        ]
        if pca_dirs:
            argv += ["--pca_dirs", pca_dirs]
        if pca_basis_pt:
            argv += ["--pca_basis_pt", pca_basis_pt]
    # ---- MFFC-F：多通道功能纤维约束（--mffc_f_on 1 时拼参数） ----
    if mffc_f_on:
        argv += [
            "--mffc_f_on", "1",
            "--mffc_f_topk", str(mffc_f_topk),
            "--mffc_f_iou", str(mffc_f_iou),
            "--mffc_f_lam_id", str(mffc_f_lam_id),
            "--mffc_f_lam_fg", str(mffc_f_lam_fg),
            "--mffc_f_lam_loc", str(mffc_f_lam_loc),
            "--mffc_f_lam_fib", str(mffc_f_lam_fib),
            "--mffc_f_w", str(mffc_f_w),
            "--mffc_f_loc_scale", str(mffc_f_loc_scale),
            "--mffc_f_fib_w_fg", str(mffc_f_fib_w_fg),
            "--mffc_f_fib_w_loc", str(mffc_f_fib_w_loc),
            "--mffc_f_temp", str(mffc_f_temp),
        ]
    return argv


def run_stage(idx, gpus, smoke=0, resume_override=None, old_ckpt_override=None,
              num_gpus=1, batch_size=None, fab_on=0, fab_lambda=1e-3,
              out_root=None, lr_scale=1.0, pca_fab_on=0, pca_dirs="", pca_basis_pt="", pca_fab_lam=1e-3, pca_fab_every=20,
              mffc_f_on=0, mffc_f_topk=5, mffc_f_iou=0.5, mffc_f_lam_id=1.0, mffc_f_lam_fg=1.0, mffc_f_lam_loc=1.0,
              mffc_f_lam_fib=0.0, mffc_f_w=1.0, mffc_f_loc_scale=1.0, mffc_f_fib_w_fg=1.0, mffc_f_fib_w_loc=1.0, mffc_f_temp=1.0,
              code_dir="/dev/shm"):
    a = STAGES[idx]
    os.chdir(PROJ)
    gpu_list = [int(g) for g in gpus.split(",")]
    if len(gpu_list) < num_gpus:
        log(f"[fatal] 指定 GPU {gpu_list} 不足 num_gpus={num_gpus}")
        sys.exit(3)
    gpu_list = gpu_list[:num_gpus]
    # 物理 GPU 通过 CUDA_VISIBLE_DEVICES 映射（单卡→cuda:0；多卡→cuda:0..n-1）。
    # 否则 dn_components 里写死的 .cuda() 默认 cuda:0，会与 --device cuda:{g} 冲突崩溃。
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_list)
    os.environ["FOTG_DEVICE"] = "cuda:0"
    os.environ["FOTG_SAVE_EVAL_CKPTS"] = "1"
    # checkpoint 策略：每 10 轮存一个编号 ckpt + 每 task 最后一轮必存（用户要求）。
    # 避免 eval_every=1 的 train 阶段每轮都存编号 ckpt 写爆 /dev/shm（tmpfs）。
    os.environ["FOTG_CKPT_EVERY"] = "10"
    os.environ["WANDB_MODE"] = "offline"
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.environ["TMPDIR"] = "/dev/shm/tmp"
    # F-A+B 运行代码全部在 code_dir（默认 /dev/shm）；O1O-main 保持纯净、仅作参照（models/util 等通过 PYTHONPATH 复用）。
    os.environ["PYTHONPATH"] = code_dir + ":" + PROJ
    out_dir = a["out"] if out_root is None else out_root.rstrip("/") + "/" + a["out"].split("/")[-1]
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs("/dev/shm/tmp", exist_ok=True)
    sys.path.insert(0, PROJ)
    sys.path.insert(0, code_dir)

    # 每卡 batch：默认取官方单卡 batch 除以卡数，保持全局 batch 不变
    # （用户已确认：每卡 2、全局 4，与官方 baseline 口径一致）。
    if batch_size is None:
        bs = a["batch"] // num_gpus
        if bs * num_gpus != a["batch"]:
            log(f"[warn] 官方 batch={a['batch']} 不能被 num_gpus={num_gpus} 整除，"
                f"每卡取 {bs}（全局={bs*num_gpus}）")
    else:
        bs = batch_size

    log(f"[stage {idx} {a['name']}] start  gpus={gpu_list} num_gpus={num_gpus} "
        f"batch_per_gpu={bs} (global={bs*num_gpus}) "
        f"PREV={a['PREV']} CUR={a['CUR']} epochs={a['epochs']} lr={a['lr']} lr_scale={lr_scale} -> {a['lr']*lr_scale:.3e} "
        f"resume={resume_override or a['resume'] or '(scratch)'} "
        f"fotg_old_ckpt={old_ckpt_override or a['old_ckpt'] or '(none)'}"
        f" pca_fab_on={pca_fab_on}")

    if num_gpus > 1:
        # ---- DDP 路径：用 torchrun 拉起 main_open_world_o1o.py ----
        # 构造纯命令行参数（不含 --num_gpus，交给 torchrun 的环境变量控制 DDP），
        # 由 torchrun 设置 RANK/WORLD_SIZE/LOCAL_RANK，main 里 init_distributed_mode 据此生效。
        argv = build_argv(a, "cuda", resume_override=resume_override,
                          old_ckpt_override=old_ckpt_override,
                          skip_resume=(smoke > 0), batch_size=bs, num_gpus=num_gpus,
                          fab_on=fab_on, fab_lambda=fab_lambda, out_root=out_root,
                          lr_scale=lr_scale, pca_fab_on=pca_fab_on, pca_dirs=pca_dirs,
                          pca_basis_pt=pca_basis_pt, pca_fab_lam=pca_fab_lam,
                          pca_fab_every=pca_fab_every,
                          mffc_f_on=mffc_f_on, mffc_f_topk=mffc_f_topk, mffc_f_iou=mffc_f_iou,
                          mffc_f_lam_id=mffc_f_lam_id, mffc_f_lam_fg=mffc_f_lam_fg,
                          mffc_f_lam_loc=mffc_f_lam_loc, mffc_f_lam_fib=mffc_f_lam_fib,
                          mffc_f_w=mffc_f_w, mffc_f_loc_scale=mffc_f_loc_scale,
                          mffc_f_fib_w_fg=mffc_f_fib_w_fg, mffc_f_fib_w_loc=mffc_f_fib_w_loc,
                          mffc_f_temp=mffc_f_temp)
        # smoke 模式额外注入：限制 batch + 跳 eval（通过环境变量，由子进程 monkeypatch 不可行；
        # 这里改为直接不改，DDP smoke 用 --smoke 走单卡已覆盖。DDP 冒烟仅验证能否启动+训几步。）
        py = "/icislab/volume2/wyy/conda/envs/rfdetr/bin/python"
        torchrun_cmd = [
            py, "-m", "torch.distributed.run",
            "--nproc_per_node", str(num_gpus),
            "--master_port", "29527",
            code_dir + "/main_open_world_o1o.py",
        ] + argv
        log(f"[stage {idx}] DDP launch: " + " ".join(torchrun_cmd[:8]) + " ...")
        t0 = time.time()
        rc = subprocess.call(torchrun_cmd, cwd=PROJ)
        log(f"[stage {idx} {a['name']}] DDP DONE rc={rc} in {time.time()-t0:.0f}s")
        return rc

    # ---- 单卡路径：进程内直接调用 M.main ----
    # 清掉可能预导入的 repo 模块，强制跑磁盘上的（含本次 F-B 改动）代码
    _stale = [m for m in list(sys.modules)
              if m in ("engine", "main_open_world_o1o", "fast_slow_scm")
              or m.startswith("models") or m.startswith("datasets")
              or m.startswith("util") or m.startswith("fotg_")]
    for m in _stale:
        del sys.modules[m]

    import main_open_world_o1o as M
    argv = build_argv(a, "cuda:0", resume_override=resume_override,
                      old_ckpt_override=old_ckpt_override,
                      skip_resume=(smoke > 0), batch_size=bs, num_gpus=1,
                      fab_on=fab_on, fab_lambda=fab_lambda, out_root=out_root,
                      lr_scale=lr_scale, pca_fab_on=pca_fab_on, pca_dirs=pca_dirs,
                      pca_basis_pt=pca_basis_pt, pca_fab_lam=pca_fab_lam,
                      pca_fab_every=pca_fab_every,
                      mffc_f_on=mffc_f_on, mffc_f_topk=mffc_f_topk, mffc_f_iou=mffc_f_iou,
                      mffc_f_lam_id=mffc_f_lam_id, mffc_f_lam_fg=mffc_f_lam_fg,
                      mffc_f_lam_loc=mffc_f_lam_loc, mffc_f_lam_fib=mffc_f_lam_fib,
                      mffc_f_w=mffc_f_w, mffc_f_loc_scale=mffc_f_loc_scale,
                      mffc_f_fib_w_fg=mffc_f_fib_w_fg, mffc_f_fib_w_loc=mffc_f_fib_w_loc,
                      mffc_f_temp=mffc_f_temp)
    args = M.get_args_parser().parse_args(argv)

    if smoke > 0:
        args.epochs = 1
        _orig_dl = M.DataLoader

        class _LimitedLoader:
            def __init__(self, loader, n):
                self.loader = loader
                self.n = n

            def __iter__(self):
                for i, b in enumerate(self.loader):
                    if i >= self.n:
                        break
                    yield b

            def __len__(self):
                return self.n

        def _limited_factory(*aa, **kk):
            return _LimitedLoader(_orig_dl(*aa, **kk), smoke)

        M.DataLoader = _limited_factory

        def _noop_eval(*aa, **kk):
            log("[smoke] evaluate skipped")
            return {}, None
        M.evaluate = _noop_eval
        log(f"[smoke] limited to {smoke} batches, eval skipped, epochs=1")

    t0 = time.time()
    rc = M.main(args)
    log(f"[stage {idx} {a['name']}] DONE rc={rc} in {time.time()-t0:.0f}s")
    return 0 if rc is None else int(rc)


def parse_cli():
    p = argparse.ArgumentParser(description="F-A+B task1-4 全链路启动器")
    p.add_argument("--all", action="store_true",
                   help="从 task1_train 起串行跑全部阶段，并自链下一阶段")
    p.add_argument("--stage", type=int, default=0,
                   help="单阶段索引 0..6（0=t1_train,1=t2_train,2=t2_ft,"
                        "3=t3_train,4=t3_ft,5=t4_train,6=t4_ft）")
    p.add_argument("--gpus", default="5",
                   help="逗号分隔的物理 GPU 列表（如 3,5）。单卡传一个即可。")
    p.add_argument("--num-gpus", type=int, default=1,
                   help="DDP 卡数。1=单卡；2=两卡并行（batch 每卡=官方batch/卡数，"
                        "全局 batch 不变）。")
    p.add_argument("--batch-size", type=int, default=None,
                   help="覆盖每卡 batch（默认 None=官方单卡 batch / num_gpus）")
    p.add_argument("--chain", action="store_true",
                   help="内部用：本阶段结束后自链下一阶段")
    p.add_argument("--smoke", type=int, default=0,
                   help="冒烟测试：限制为 N 个 batch 并跳过 eval（仅单卡路径支持）")
    p.add_argument("--resume-override", default="",
                   help="调试用：覆盖 --resume 路径")
    p.add_argument("--old-ckpt-override", default="",
                   help="调试用：覆盖 --fotg_old_ckpt 路径")
    p.add_argument("--fab-on", type=int, default=0,
                   help="F-B 软投影开关（0=关/F-A，1=开/F-B）")
    p.add_argument("--fab-lambda", type=float, default=1e-3,
                   help="F-B nullspace_projection 的 λ（正则）")
    p.add_argument("--lr-scale", type=float, default=1.0,
                   help="lr 缩放因子。全局 batch 从官方 16 降到 4（两卡×2）时，"
                        "按 linear scaling 应设 0.25（lr = 官方 lr × 0.25）。")
    p.add_argument("--pca-fab-on", type=int, default=0,
                   help="新 F-B（PCA-directional Jacobian Protection）开关（0=关，1=开）")
    p.add_argument("--pca-dirs", default="5,9,37",
                   help="冲突方向索引，逗号分隔（默认 5,9,37）")
    p.add_argument("--pca-basis-pt", default="/dev/shm/pca_basis_t2e.pt",
                   help="预计算 PCA 方向基 .pt 路径")
    p.add_argument("--pca-fab-lam", type=float, default=1e-3,
                   help="新 F-B 投影正则 λ")
    p.add_argument("--pca-fab-every", type=int, default=20,
                   help="新 F-B 每 N 步重算 J_K，中间步复用缓存")
    # ---- MFFC-F 开关与超参（本启动器默认开启 MFFC-F-only 口径） ----
    p.add_argument("--mffc-f-on", type=int, default=1,
                   help="MFFC-F 多通道功能纤维约束开关（本启动器默认 1=MFFC-F-only；"
                        "设 0 可与同源 FAB 跑 baseline 对照）。")
    p.add_argument("--mffc-f-topk", type=int, default=5, help="F1 教师 Top-K 旧类竞争类数")
    p.add_argument("--mffc-f-iou", type=float, default=0.5, help="GT 对齐匹配 IoU 阈值")
    p.add_argument("--mffc-f-lam-id", type=float, default=1.0, help="F1 身份保持权重")
    p.add_argument("--mffc-f-lam-fg", type=float, default=1.0, help="F2 前景保持权重")
    p.add_argument("--mffc-f-lam-loc", type=float, default=1.0, help="F3 定位保持权重")
    p.add_argument("--mffc-f-lam-fib", type=float, default=0.0, help="L_fib 局部 Jacobian 功能纤维约束权重（第一组实验=0，纯 F1+F2+F3；后续组可开 0.1）")
    p.add_argument("--mffc-f-w", type=float, default=1.0, help="MFFC-F 总损失整体权重")
    p.add_argument("--mffc-f-loc-scale", type=float, default=1.0, help="定位坐标额外缩放")
    p.add_argument("--mffc-f-fib-w-fg", type=float, default=1.0, help="L_fib 前景通道归一化权重 W_F")
    p.add_argument("--mffc-f-fib-w-loc", type=float, default=1.0, help="L_fib 定位通道归一化权重 W_F")
    p.add_argument("--mffc-f-temp", type=float, default=1.0, help="身份 margin 温度尺度")
    p.add_argument("--out-root", default="",
                   help="覆盖输出根目录（F-B 独立目录隔离），空=用 STAGES 默认 out")
    p.add_argument("--code-dir", default=os.path.dirname(os.path.abspath(__file__)),
                   help="运行代码目录（mffc_f.py/engine.py/main_open_world_o1o.py 所在）。"
                        "默认=本启动脚本所在目录（自包含）。MFFC-F 包直接放 --code-dir 即可。")
    return p.parse_args()


def main():
    cli = parse_cli()
    idx = cli.stage
    if not (0 <= idx < len(STAGES)):
        log(f"[fatal] --stage {idx} 越界 (0..{len(STAGES)-1})")
        sys.exit(2)

    rc = run_stage(idx, cli.gpus, smoke=cli.smoke,
                   resume_override=cli.resume_override or None,
                   old_ckpt_override=cli.old_ckpt_override or None,
                   num_gpus=cli.num_gpus, batch_size=cli.batch_size,
                   fab_on=cli.fab_on, fab_lambda=cli.fab_lambda,
                   out_root=cli.out_root or None,
                   lr_scale=cli.lr_scale, pca_fab_on=cli.pca_fab_on,
                   pca_dirs=cli.pca_dirs, pca_basis_pt=cli.pca_basis_pt,
                   pca_fab_lam=cli.pca_fab_lam, pca_fab_every=cli.pca_fab_every,
                   mffc_f_on=cli.mffc_f_on, mffc_f_topk=cli.mffc_f_topk, mffc_f_iou=cli.mffc_f_iou,
                   mffc_f_lam_id=cli.mffc_f_lam_id, mffc_f_lam_fg=cli.mffc_f_lam_fg,
                   mffc_f_lam_loc=cli.mffc_f_lam_loc, mffc_f_lam_fib=cli.mffc_f_lam_fib,
                   mffc_f_w=cli.mffc_f_w, mffc_f_loc_scale=cli.mffc_f_loc_scale,
                   mffc_f_fib_w_fg=cli.mffc_f_fib_w_fg, mffc_f_fib_w_loc=cli.mffc_f_fib_w_loc,
                   mffc_f_temp=cli.mffc_f_temp,
                   code_dir=cli.code_dir)

    # 自链：当前阶段成功且不是最后一阶段，则拉起下一阶段
    if rc == 0 and idx < len(STAGES) - 1 and (cli.all or cli.chain) and cli.smoke == 0:
        nxt = idx + 1
        nxt_name = STAGES[nxt]["name"]
        log(f"[chain] 拉起下一阶段 {nxt} ({nxt_name}) on gpus={cli.gpus}")
        cmd = [sys.executable, "-u", os.path.abspath(__file__),
               "--stage", str(nxt), "--gpus", cli.gpus,
               "--num-gpus", str(cli.num_gpus), "--chain"]
        if cli.batch_size is not None:
            cmd += ["--batch-size", str(cli.batch_size)]
        with open(SUPERVISOR_LOG, "a") as f:
            f.write("LAUNCH " + " ".join(cmd) + "\n")
        subprocess.Popen(cmd)
    elif rc == 0 and idx == len(STAGES) - 1 and cli.smoke == 0:
        log("[chain] 全部阶段完成 ✅")
    elif cli.smoke > 0:
        log(f"[smoke] 阶段 {idx} 冒烟结束 rc={rc}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
