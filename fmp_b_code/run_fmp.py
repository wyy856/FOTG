#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FMP task1-4 launcher using the O1O stage schedule.  -- B-run (unk_head) edition.

与原版 run_fmp.py 的差异（仅 4 处，其余逐字一致）：
  1. PROJ 指向 /dev/shm/unkhead/proj（独立代码环境，models 内含 unk_head 编辑版 detector）。
  2. SUPERVISOR_LOG 指向 /dev/shm/unkhead/supervisor.log（不与 A run 的日志混写）。
  3. torchrun master_port 29527 -> 29537（A run 占用 29527）。
  4. build_argv 注入 --unk_head_w 1.0（独立 Unknown/Objectness 头，A run 无此参数）。
"""
import os
import sys
import time
import argparse
import subprocess

# 屏蔽 ~/.local 用户 site（内含损坏的 scipy），强制使用 conda env 的包。
# 注意：PYTHONNOUSERSITE=1 对已启动的进程无效（site 已处理完），
# 必须在任何第三方包被 import 之前直接过滤 sys.path。
sys.path = [p for p in sys.path if "/.local/lib/" not in p]

os.environ.setdefault("TMPDIR", "/dev/shm/tmp")
os.environ["PYTHONNOUSERSITE"] = "1"
os.environ["WANDB_MODE"] = "offline"          # 无 API key 直接崩

PROJ = "/dev/shm/unkhead/proj"
# Checkpoints live in tmpfs; archive them before a machine restart.
SUPERVISOR_LOG = "/dev/shm/unkhead/supervisor.log"

# FMP-only: retain the frozen teacher, disable the legacy FOTG losses, and use
# geometry/function preservation plus hs_last-space new-gradient projection.
FMP_FLAGS = [
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
# resume / old_ckpt are previous-stage checkpoints; epochs are global end epochs.
STAGES = [
    dict(name="task1_train", train_set="owod_t1_train", PREV=0, CUR=20,
         psc=0, csc=6, nsc=13, batch=4, lr=2e-4, epochs=30, lr_drop=17,
         eval_every=1, resume="", old_ckpt="", ft=False,
         pseudo="data/OWOD/pseudo-boxes/TOWOD/mowod_until_t1_normal_pseudo_labels.json",
         out="/dev/shm/fmp_chain/task1_train",
         ex_prev="", ex_cur="learned_owod_t1_ft.txt", max_len=850),
    dict(name="task2_train", train_set="owod_t2_train", PREV=20, CUR=20,
         psc=6, csc=3, nsc=13, batch=4, lr=2e-5, epochs=40, lr_drop=50,
         eval_every=1,
         resume="/icislab/volume1/wyy/O1O-main/exps/o1o_mowod_task1_train/checkpoint0029.pth",
         old_ckpt="/icislab/volume1/wyy/O1O-main/exps/o1o_mowod_task1_train/checkpoint0029.pth", ft=False,
         pseudo="data/OWOD/pseudo-boxes/TOWOD/mowod_until_t2_normal_pseudo_labels.json",
         out="/dev/shm/fmp_chain/task2_train",
         ex_prev="learned_owod_t1_ft.txt", ex_cur="learned_owod_t2_ft.txt", max_len=1743),
    dict(name="task2_ft", train_set="o1o_mowod_exemplars/learned_owod_t2_ft",
         PREV=20, CUR=20, psc=6, csc=3, nsc=13, batch=4, lr=2e-4, epochs=60,
         lr_drop=50, eval_every=5,
         resume="/dev/shm/fmp_chain/task2_train/checkpoint0039.pth",
         old_ckpt="/icislab/volume1/wyy/O1O-main/exps/o1o_mowod_task1_train/checkpoint0029.pth", ft=True,
         pseudo="data/OWOD/pseudo-boxes/TOWOD/mowod_until_t2_normal_pseudo_labels.json",
         out="/dev/shm/fmp_chain/task2_ft", ex_prev="", ex_cur="", max_len=0),
    dict(name="task3_train", train_set="owod_t3_train", PREV=40, CUR=20,
         psc=9, csc=2, nsc=13, batch=4, lr=2e-5, epochs=70, lr_drop=80,
         eval_every=1,
         resume="/dev/shm/fmp_chain/task2_ft/checkpoint0059.pth",
         old_ckpt="/dev/shm/fmp_chain/task2_ft/checkpoint0059.pth", ft=False,
         pseudo="data/OWOD/pseudo-boxes/TOWOD/mowod_until_t3_normal_pseudo_labels.json",
         out="/dev/shm/fmp_chain/task3_train",
         ex_prev="learned_owod_t2_ft.txt", ex_cur="learned_owod_t3_ft.txt", max_len=2361),
    dict(name="task3_ft", train_set="o1o_mowod_exemplars/learned_owod_t3_ft",
         PREV=40, CUR=20, psc=9, csc=2, nsc=13, batch=4, lr=2e-4, epochs=90,
         lr_drop=80, eval_every=5,
         resume="/dev/shm/fmp_chain/task3_train/checkpoint0069.pth",
         old_ckpt="/dev/shm/fmp_chain/task3_train/checkpoint0069.pth", ft=True,
         pseudo="data/OWOD/pseudo-boxes/TOWOD/mowod_until_t3_normal_pseudo_labels.json",
         out="/dev/shm/fmp_chain/task3_ft", ex_prev="", ex_cur="", max_len=0),
    dict(name="task4_train", train_set="owod_t4_train", PREV=60, CUR=20,
         psc=11, csc=1, nsc=13, batch=4, lr=2e-5, epochs=100, lr_drop=110,
         eval_every=1,
         resume="/dev/shm/fmp_chain/task3_ft/checkpoint0089.pth",
         old_ckpt="/dev/shm/fmp_chain/task3_ft/checkpoint0089.pth", ft=False,
         pseudo="data/OWOD/pseudo-boxes/TOWOD/mowod_until_t4_normal_pseudo_labels.json",
         out="/dev/shm/fmp_chain/task4_train",
         ex_prev="learned_owod_t3_ft.txt", ex_cur="learned_owod_t4_ft.txt", max_len=2749),
    dict(name="task4_ft", train_set="owod_t4_train", PREV=60, CUR=20,
         psc=11, csc=1, nsc=13, batch=4, lr=2e-4, epochs=120,
         lr_drop=110, eval_every=5,
         resume="/dev/shm/fmp_chain/task4_train/checkpoint0099.pth",
         old_ckpt="/dev/shm/fmp_chain/task4_train/checkpoint0099.pth", ft=True,
         pseudo="data/OWOD/pseudo-boxes/TOWOD/mowod_until_t4_normal_pseudo_labels.json",
         out="/dev/shm/fmp_chain/task4_ft", ex_prev="", ex_cur="", max_len=0),
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
               fmp_on=1, fmp_topk=5, fmp_lam_geo=1.0,
               fmp_lam_id=1.0, fmp_lam_fg=1.0, fmp_lam_loc=1.0,
               fmp_w=1.0, fmp_project_on=1, fmp_geo_same_weight=1.0,
               fmp_geo_cross_weight=1.0, fmp_geo_scale_sigma=1.5,
               fmp_projection_rtol=1e-4):
    a = stage
    out = (a["out"] if out_root is None else
           out_root.rstrip("/") + "/" + a["out"].split("/")[-1])
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
            # 崩溃续训：若本 stage 自己的 output_dir 已有 checkpoint.pth（上次跑到一半），
            # 优先从它续（否则每次都从 a["resume"] 的上一任务 ckpt 重头跑，崩溃=白跑）。
            # 仅对非 ft 的 train stage 生效；ft 的 a["resume"] 是上游 train 的 checkpoint0039，不适用。
            self_ckpt = out + "/checkpoint.pth"
            if (not a.get("ft", False)) and os.path.isfile(self_ckpt):
                r = self_ckpt
            elif out_root is not None and r.startswith("/dev/shm/fmp_chain/"):
                r = out_root.rstrip("/") + "/" + r.split("/")[-2] + "/" + r.split("/")[-1]
            argv += ["--resume", r]
    if old_ckpt_override:
        argv += ["--fotg_old_ckpt", old_ckpt_override]
    elif a["old_ckpt"]:
        o = a["old_ckpt"]
        if out_root is not None and o.startswith("/dev/shm/fmp_chain/"):
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
    argv += FMP_FLAGS
    # ---- B-run 专用：独立 Unknown/Objectness 头（A run 无此参数）----
    argv += ["--unk_head_w", "1.0"]
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
    if fmp_on:
        argv += [
            "--fmp_on", "1",
            "--fmp_project_on", str(fmp_project_on),
            "--fmp_topk", str(fmp_topk),
            "--fmp_lam_geo", str(fmp_lam_geo),
            "--fmp_lam_id", str(fmp_lam_id),
            "--fmp_lam_fg", str(fmp_lam_fg),
            "--fmp_lam_loc", str(fmp_lam_loc),
            "--fmp_w", str(fmp_w),
            "--fmp_geo_same_weight", str(fmp_geo_same_weight),
            "--fmp_geo_cross_weight", str(fmp_geo_cross_weight),
            "--fmp_geo_scale_sigma", str(fmp_geo_scale_sigma),
            "--fmp_projection_rtol", str(fmp_projection_rtol),
        ]
    return argv


def run_stage(idx, gpus, smoke=0, resume_override=None, old_ckpt_override=None,
              num_gpus=1, batch_size=None, fab_on=0, fab_lambda=1e-3,
              out_root=None, lr_scale=1.0, pca_fab_on=0, pca_dirs="", pca_basis_pt="", pca_fab_lam=1e-3, pca_fab_every=20,
              fmp_on=1, fmp_topk=5, fmp_lam_geo=1.0, fmp_lam_id=1.0,
              fmp_lam_fg=1.0, fmp_lam_loc=1.0, fmp_w=1.0,
              fmp_project_on=1,
              fmp_geo_same_weight=1.0, fmp_geo_cross_weight=1.0,
              fmp_geo_scale_sigma=1.5, fmp_projection_rtol=1e-4,
              code_dir=os.path.dirname(os.path.abspath(__file__))):
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
    os.environ["FOTG_CKPT_EVERY"] = "2"   # 每 2 轮存编号 ckpt，抗间歇崩溃
    os.environ["WANDB_MODE"] = "offline"
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.environ["TMPDIR"] = "/dev/shm/tmp"
    # FMP scripts run from code_dir; O1O-main supplies model/data utility packages.
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
                          fmp_on=fmp_on, fmp_topk=fmp_topk,
                          fmp_lam_geo=fmp_lam_geo, fmp_lam_id=fmp_lam_id,
                          fmp_lam_fg=fmp_lam_fg, fmp_lam_loc=fmp_lam_loc,
                          fmp_w=fmp_w, fmp_project_on=fmp_project_on,
                          fmp_geo_same_weight=fmp_geo_same_weight,
                          fmp_geo_cross_weight=fmp_geo_cross_weight,
                          fmp_geo_scale_sigma=fmp_geo_scale_sigma,
                          fmp_projection_rtol=fmp_projection_rtol)
        # smoke 模式额外注入：限制 batch + 跳 eval（通过环境变量，由子进程 monkeypatch 不可行；
        # 这里改为直接不改，DDP smoke 用 --smoke 走单卡已覆盖。DDP 冒烟仅验证能否启动+训几步。）
        py = "/icislab/volume2/wyy/conda/envs/rfdetr/bin/python"
        # --- 崩溃诊断（只改日志落盘方式，不影响训练语义）---
        # torchrun 默认会把子进程(rank)的异常吞掉，只在父进程留一个 ChildFailedError 外壳，
        # 真实 Python traceback 拿不到。这里：
        #   --log-dir  : 每个 rank 的 stdout/stderr 落到 <out_dir>/tb/<run>/attempt_0/<rank>/
        #   --redirects 3: stdout+stderr 重定向到上述文件
        #   --tee 3    : 同时回显到父进程 stdout，保证 tee 到 launch.log 的训练进度日志不丢
        tb_dir = os.path.join(out_dir, "tb")
        os.makedirs(tb_dir, exist_ok=True)
        torchrun_cmd = [
            py, "-m", "torch.distributed.run",
            "--nproc_per_node", str(num_gpus),
            "--master_port", "29537",
            "--log-dir", tb_dir,
            "--redirects", "3",
            "--tee", "3",
            code_dir + "/main_open_world_o1o.py",
        ] + argv
        log(f"[stage {idx}] DDP launch: " + " ".join(torchrun_cmd[:8]) + " ...")
        t0 = time.time()
        # 抗 RTX3090(无ECC) 偶发 CUDA 故障：expandable_segments 减少 allocator 碎片。
        # 不启用 CUDA_LAUNCH_BLOCKING（会拖慢~10-30%）；若需定位根因，临时设该 env=1 重跑。
        _cuda_env = dict(os.environ)
        _cuda_env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        rc = subprocess.call(torchrun_cmd, cwd=PROJ, env=_cuda_env)
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
                      fmp_on=fmp_on, fmp_topk=fmp_topk,
                      fmp_lam_geo=fmp_lam_geo, fmp_lam_id=fmp_lam_id,
                      fmp_lam_fg=fmp_lam_fg, fmp_lam_loc=fmp_lam_loc,
                      fmp_w=fmp_w, fmp_project_on=fmp_project_on,
                      fmp_geo_same_weight=fmp_geo_same_weight,
                      fmp_geo_cross_weight=fmp_geo_cross_weight,
                      fmp_geo_scale_sigma=fmp_geo_scale_sigma,
                      fmp_projection_rtol=fmp_projection_rtol)
    args = M.get_args_parser().parse_args(argv)

    if smoke > 0:
        args.epochs = 1
        # smoke 只跑 N 个 batch，exemplar 选择在全量得分缺失时会对空张量 min() 崩溃，
        # 故 smoke 下直接关闭 exemplar 选择（真实训练不受影响）。
        args.exemplar_replay_selection = False
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
    p = argparse.ArgumentParser(description="FunctionalManifoldPreservation task1-4 launcher")
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
    p.add_argument("--fmp-on", type=int, default=1, help="Enable FunctionalManifoldPreservation.")
    p.add_argument("--fmp-topk", type=int, default=5, help="Old identity competitors per matched instance.")
    p.add_argument("--fmp-lam-geo", type=float, default=1.0, help="Geometry loss weight.")
    p.add_argument("--fmp-lam-id", type=float, default=1.0, help="Identity function loss weight.")
    p.add_argument("--fmp-lam-fg", type=float, default=1.0, help="Foreground function loss weight.")
    p.add_argument("--fmp-lam-loc", type=float, default=1.0, help="Localization function loss weight.")
    p.add_argument("--fmp-w", type=float, default=1.0, help="Overall FMP loss multiplier.")
    p.add_argument("--fmp-project-on", type=int, default=1,
                   help="Enable only the hs_last new-gradient nullspace update rule.")
    p.add_argument("--fmp-geo-same-weight", type=float, default=1.0)
    p.add_argument("--fmp-geo-cross-weight", type=float, default=1.0)
    p.add_argument("--fmp-geo-scale-sigma", type=float, default=1.5)
    p.add_argument("--fmp-projection-rtol", type=float, default=1e-4)
    p.add_argument("--out-root", default="",
                   help="覆盖输出根目录（F-B 独立目录隔离），空=用 STAGES 默认 out")
    p.add_argument("--code-dir", default=os.path.dirname(os.path.abspath(__file__)),
                   help="Directory containing this FMP module and its training scripts.")
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
                   fmp_on=cli.fmp_on, fmp_topk=cli.fmp_topk,
                   fmp_lam_geo=cli.fmp_lam_geo, fmp_lam_id=cli.fmp_lam_id,
                   fmp_lam_fg=cli.fmp_lam_fg, fmp_lam_loc=cli.fmp_lam_loc,
                   fmp_w=cli.fmp_w, fmp_project_on=cli.fmp_project_on,
                   fmp_geo_same_weight=cli.fmp_geo_same_weight,
                   fmp_geo_cross_weight=cli.fmp_geo_cross_weight,
                   fmp_geo_scale_sigma=cli.fmp_geo_scale_sigma,
                   fmp_projection_rtol=cli.fmp_projection_rtol,
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
        cmd += [
            "--fmp-on", str(cli.fmp_on),
            "--fmp-project-on", str(cli.fmp_project_on),
            "--fmp-topk", str(cli.fmp_topk),
            "--fmp-lam-geo", str(cli.fmp_lam_geo),
            "--fmp-lam-id", str(cli.fmp_lam_id),
            "--fmp-lam-fg", str(cli.fmp_lam_fg),
            "--fmp-lam-loc", str(cli.fmp_lam_loc),
            "--fmp-w", str(cli.fmp_w),
            "--fmp-geo-same-weight", str(cli.fmp_geo_same_weight),
            "--fmp-geo-cross-weight", str(cli.fmp_geo_cross_weight),
            "--fmp-geo-scale-sigma", str(cli.fmp_geo_scale_sigma),
            "--fmp-projection-rtol", str(cli.fmp_projection_rtol),
        ]
        if cli.out_root:
            cmd += ["--out-root", cli.out_root]
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
