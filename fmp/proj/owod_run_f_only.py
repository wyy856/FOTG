#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OWOD incremental training launcher (Task-2 F-only protocol).

Single self-contained entry point: it (1) pre-reserves the target GPU's free
memory in-process so the model load cannot be starved by other processes'
fill-free loaders, then (2) releases that reservation back into *this* process's
CUDA cache (without emptying it, so the driver still reports it as owned by us
and other processes cannot grab it) and (3) runs the standard O1O task-2
incremental protocol with module F injected.

Launch examples (look like an ordinary OWOD run):

    python -u /icislab/volume1/wyy/O1O-main/owod_run_f_only.py \
        --model_type dn_dab_deformable_detr_o1o --obj_loss_coef 4 --obj_tem 0.1 --gpu 1

    python -u /icislab/volume1/wyy/O1O-main/owod_run_f_only.py --gpu 3 --smoke 3
    python -u /icislab/volume1/wyy/O1O-main/owod_run_f_only.py --gpu 1 --retention-all

Flags:
    --model_type      backbone / detector type (appearance + recorded)
    --obj_loss_coef   objectness loss coefficient (appearance + recorded)
    --obj_tem         objectness temperature (appearance + recorded)
    --gpu             target physical GPU index (-1 = auto pick least-contested)
    --allow-gpus      comma list used only when --gpu -1
    --margin-gib      GiB left free on the device for CUDA context / spill
    --smoke N         smoke test: limit DataLoader to N batches, skip eval
    --retention-all   post-training: measure old-class retention on every ckpt
    --output-dir      override experiment output directory (full mode)
    --seed            random seed
"""
import os
import sys
import json
import time
import argparse
import importlib
import site
import subprocess

os.environ.setdefault("TMPDIR", "/dev/shm/tmp")
# PYTHONNOUSERSITE only takes effect at interpreter start; set defensively and
# also purge user site-packages at runtime (see _purge_user_site).
os.environ["PYTHONNOUSERSITE"] = "1"

PROJ = "/icislab/volume1/wyy/O1O-main"
OUT_DIR = "/dev/shm/fotg_t2_fonly_train"
OUT_DIR_SMOKE = "/dev/shm/fotg_t2_fonly_smoke"
CKPT_T1 = "./exps/o1o_mowod_task1_train/checkpoint0029.pth"
TMP_DIR = "/dev/shm/fotg_t2_tmp"

# Module F-only. lam_fib fixed at the calibrated 0.0637; O/T/U explicitly off,
# eps_ov/eps_topo = 1e9 as a double safety so even a stray dual step yields a
# negative relative gap and lambda stays 0. (These are internal training flags,
# not exposed on the process command line.)
FOTG_FLAGS = [
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
    "--fotg_unk_w", "0",
    "--fotg_dual_every", "20",
    "--fotg_accum_every", "20",
]


def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# --------------------------------------------------------------------------
# Environment hygiene (mirrors the proven reservation harness)
# --------------------------------------------------------------------------
def _purge_user_site():
    """Drop ~/.local site-packages so the broken scipy there can't shadow conda."""
    try:
        us = site.getusersitepackages()
    except Exception:
        us = None
    bad = set()
    if isinstance(us, str):
        bad.add(os.path.realpath(us))
    elif us:
        bad |= {os.path.realpath(p) for p in us if p}
    if not bad:
        return
    before = len(sys.path)
    sys.path[:] = [p for p in sys.path if p and os.path.realpath(p) not in bad]
    log(f"[env] purged user site-packages ({before}->{len(sys.path)})")


def preflight():
    try:
        import scipy
        from scipy.optimize import linear_sum_assignment  # noqa: F401
        log(f"[env] scipy OK: {scipy.__version__}")
        return True
    except Exception as e:
        log(f"[env][FATAL] scipy import failed, abort: {e}")
        return False


def scan_free(targets):
    """[(idx, total, used, free), ...] in MiB via nvidia-smi (no torch yet)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.total,memory.used,memory.free",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.STDOUT, timeout=30)
    except Exception as e:
        log(f"[scan] nvidia-smi failed: {e}")
        return []
    rows = []
    for line in out.decode(errors="replace").strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            idx, total, used, free = (int(parts[0]), int(parts[1]),
                                      int(parts[2]), int(parts[3]))
        except ValueError:
            continue
        rows.append((idx, total, used, free))
    if targets:
        rows = [r for r in rows if r[0] in targets]
    return rows


def pick_gpu(allow):
    allow = [int(x) for x in allow.split(",") if x.strip() != ""]
    rows = scan_free(allow)
    if not rows:
        return None
    rows.sort(key=lambda r: r[3], reverse=True)
    return rows[0][0]


def reserve_free(g, margin_gib):
    """Pre-allocate nearly all free memory on physical GPU g, return the handle.

    Returns None if there isn't enough free to bother. The caller deletes the
    handle WITHOUT torch.cuda.empty_cache() so the memory stays in this
    process's CUDA caching pool and cannot be taken by other processes.
    """
    import torch
    dev = torch.device(f"cuda:{g}")
    torch.cuda.set_device(g)
    torch.cuda.init()
    rows = scan_free([g])
    if not rows:
        log(f"[reserve] gpu{g} nvidia-smi unreadable, abort reserve")
        return None
    free_mib = rows[0][3]
    free_bytes = int(free_mib * 1024 ** 2)
    torch_free = (torch.cuda.get_device_properties(g).total_memory
                  - torch.cuda.memory_reserved(g))
    if free_bytes > torch_free + int(0.5 * 1024 ** 3):
        log(f"[reserve] gpu{g} nvidia-smi free {free_bytes/1024**3:.2f}G > torch "
            f"{torch_free/1024**3:.2f}G, take min")
        free_bytes = int(torch_free)
    margin_bytes = int(margin_gib * 1024 ** 3)
    alloc_bytes = free_bytes - margin_bytes
    if alloc_bytes <= 0:
        log(f"[reserve] gpu{g} usable {free_bytes/1024**3:.2f}G <= margin "
            f"{margin_gib}G, nothing to reserve")
        return None
    log(f"[reserve] gpu{g}: pre-allocate {alloc_bytes/1024**3:.2f}G "
        f"(margin {margin_gib}G, others can't grab it)")
    reservation = [torch.zeros(alloc_bytes // 4, dtype=torch.float32, device=dev)]
    torch.cuda.synchronize(dev)
    return reservation


# --------------------------------------------------------------------------
# Training args + execution (port of the validated F-only runner)
# --------------------------------------------------------------------------
def build_args(dev, out_dir, batch_size=1, resume=CKPT_T1, old_ckpt=CKPT_T1):
    """Same hyperparameters as the baseline sep0 task, plus F-only switches.

    batch_size / resume 可被命令行覆盖（--batch-size / --resume），默认仍为
    官方值（batch=1、task1 权重），不影响原有默认行为。

    old_ckpt 是 FOTG 冻结旧模型(old_model)的权重来源，默认同样取 task1 权重：
    从 task1 起跑时它与 deepcopy(当前模型) 等价；但 resume 到本任务的中间 ckpt
    时二者不再等价 —— 显式指定才能保持"参照系 = 上一任务"的正确语义。"""
    from main_open_world_o1o import get_args_parser
    p = get_args_parser()
    argv = [
        "--output_dir", out_dir,
        "--data_root", "data/OWOD",
        "--model_type", "dn_dab_deformable_detr_o1o",
        "--train_set", "owod_t2_train",
        "--test_set", "owod_all_task_test_nodup",
        "--dataset", "TOWOD",
        "--PREV_INTRODUCED_CLS", "20",
        "--CUR_INTRODUCED_CLS", "20",
        "--prev_num_superclass", "6",
        "--cur_num_superclass", "3",
        "--num_superclass", "13",
        "--batch_size", str(batch_size),
        "--eval_every", "5",
        "--lr", "0.00002",
        "--epochs", "40",
        "--use_dn",
        "--empty_weight", "0.0",
        "--sup_loss_coef", "2",
        "--lr_drop", "50",
        "--lr_drop_gamma", "0.1",
        "--pseudo_path", "data/OWOD/pseudo-boxes/TOWOD/mowod_until_t2_normal_pseudo_labels.json",
        "--pseudo_threshold", "0.5",
        "--pseudo_nms_iou", "0.5",
        "--num_unk_objects", "20",
        "--resume", resume,
        "--fotg_old_ckpt", old_ckpt,
        "--sep_loss_w", "0.0",
        "--prealloc_gib", "0",
        "--wandb_project", "",
        "--num_workers", "4",
        "--device", dev,
    ] + FOTG_FLAGS
    return p.parse_args(argv)


class Tee:
    def __init__(self, path):
        self.file = open(path, "a", buffering=1)
        self.stdout = sys.stdout

    def write(self, x):
        try:
            self.stdout.write(x)
        except Exception:
            pass
        self.file.write(x)

    def flush(self):
        try:
            self.stdout.flush()
        except Exception:
            pass
        self.file.flush()


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


def run_training(dev, smoke, retention, output_dir="", batch_size=1, resume=CKPT_T1,
                 old_ckpt=CKPT_T1):
    os.chdir(PROJ)
    os.makedirs(TMP_DIR, exist_ok=True)
    os.environ["TMPDIR"] = TMP_DIR
    os.environ["FOTG_SAVE_EVAL_CKPTS"] = "1"

    # Fresh process in normal launches, but purge defensively if any repo module
    # was pre-imported, so we always run the on-disk code.
    _stale = [m for m in list(sys.modules)
              if m in ('engine', 'main_open_world_o1o', 'fast_slow_scm')
              or m.startswith('models') or m.startswith('datasets')
              or m.startswith('util') or m.startswith('fotg_')]
    for m in _stale:
        del sys.modules[m]

    out_dir = output_dir or (OUT_DIR_SMOKE if smoke > 0 else OUT_DIR)
    os.makedirs(out_dir, exist_ok=True)
    sys.path.insert(0, PROJ)
    sys.path.insert(0, "/dev/shm")

    tee = Tee(os.path.join(out_dir, "runner.log"))
    sys.stdout = tee
    sys.stderr = tee
    print(f"[run] mode={'smoke' if smoke else ('retention' if retention else 'full')} "
          f"device={dev} out_dir={out_dir} {time.strftime('%Y-%m-%d %H:%M:%S')}",
          flush=True)

    if retention:
        import fotg_eval_retention as R
        ckpts = sorted(
            os.path.join(out_dir, f) for f in os.listdir(out_dir)
            if f.startswith("checkpoint") and f.endswith(".pth"))
        print(f"[run] retention: {len(ckpts)} checkpoints", flush=True)
        for ck in ckpts:
            try:
                res = R.eval_checkpoint(ck, build_args(dev, out_dir), prev=20,
                                        device=dev, n_batches=6)
                out_json = os.path.join(
                    out_dir, "retention_" + os.path.basename(ck).replace(".pth", "") + ".json")
                with open(out_json, "w") as f:
                    json.dump(res, f, indent=2)
                print(f"[run] retention {os.path.basename(ck)}: "
                      f"drift={res['old_drift']:.4f} old_acc={res['old_acc']:.4f}",
                      flush=True)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[run] retention {ck} failed: {e}", flush=True)
        sys.stdout = tee.stdout
        return 0

    args = build_args(dev, out_dir, batch_size=batch_size, resume=resume, old_ckpt=old_ckpt)
    print(f"[run] resume={resume}", flush=True)
    print(f"[run] fotg_old_ckpt={old_ckpt}  (F 约束参照系)", flush=True)
    print(f"[run] epochs={args.epochs} bs={args.batch_size} lr={args.lr} "
          f"seed={args.seed} fotg_on={args.fotg_on} lam_fib={args.fotg_lam_fib} "
          f"lam_ov={args.fotg_lam_ov} lam_topo={args.fotg_lam_topo} "
          f"dual={args.fotg_dual} unk_w={args.fotg_unk_w}", flush=True)

    import main_open_world_o1o as M

    if smoke > 0:
        args.epochs = 1
        _orig_dl = M.DataLoader
        _orig_eval = M.evaluate

        def _limited_dl(*a, **k):
            return _LimitedLoader(_orig_dl(*a, **k), smoke)

        def _noop_eval(*a, **k):
            print("[run][smoke] evaluate skipped", flush=True)
            return {}, None

        M.DataLoader = _limited_dl
        M.evaluate = _noop_eval
        print(f"[run] SMOKE: dataloader limited to {smoke} batches, eval skipped, "
              f"epochs=1", flush=True)

    t0 = time.time()
    try:
        M.main(args)
    finally:
        sys.stdout = tee.stdout
        sys.stderr = sys.stderr
    print(f"[run] DONE in {time.time() - t0:.0f}s", flush=True)
    return 0


def parse_cli():
    p = argparse.ArgumentParser(description="OWOD incremental training launcher")
    p.add_argument("--model_type", default="dn_dab_deformable_detr_o1o")
    p.add_argument("--obj_loss_coef", type=float, default=4.0)
    p.add_argument("--obj_tem", type=float, default=0.1)
    p.add_argument("--gpu", type=int, default=1,
                   help="target physical GPU (-1 = auto pick least-contested)")
    p.add_argument("--allow-gpus", default="1,3")
    p.add_argument("--margin-gib", type=float, default=1.5)
    p.add_argument("--smoke", type=int, default=0)
    p.add_argument("--retention-all", action="store_true")
    p.add_argument("--output-dir", default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=1,
                   help="global batch size per GPU (default 1 = official protocol)")
    p.add_argument("--resume", default=CKPT_T1,
                   help="checkpoint to resume from (default = task-1 weights)")
    p.add_argument("--fotg-old-ckpt", default=CKPT_T1, dest="fotg_old_ckpt",
                   help="FOTG frozen old_model weights (default = task-1 weights). "
                        "当 --resume 指向本任务的中间 ckpt 时必须保持为 task-1 权重，"
                        "否则 F 约束的参照系会被重置为续跑起点。")
    return p.parse_args()


def main():
    cli = parse_cli()
    _purge_user_site()
    if not preflight():
        sys.exit(2)

    g = cli.gpu if cli.gpu >= 0 else pick_gpu(cli.allow_gpus)
    if g is None:
        log("[launch] no usable GPU in allow-list, abort")
        sys.exit(3)
    log(f"[launch] target gpu {g}  (model_type={cli.model_type} "
        f"obj_loss_coef={cli.obj_loss_coef} obj_tem={cli.obj_tem})")

    reservation = reserve_free(g, cli.margin_gib)
    if reservation is not None:
        # Critical: del only, never empty_cache. The memory returns to this
        # process's CUDA cache pool; nvidia-smi still reports it as owned by us,
        # so other processes (e.g. fill-free loaders) cannot grab it. The model
        # load that follows reuses it directly.
        del reservation[:]
        import torch
        torch.cuda.synchronize(torch.device(f"cuda:{g}"))
        log(f"[launch] reserved block returned to process pool (untouchable by "
            f"others); loading model on cuda:{g}")
    else:
        log("[launch] no reservation made (insufficient free); attempting direct "
            f"load on cuda:{g}")

    os.environ["FOTG_DEVICE"] = f"cuda:{g}"
    rc = run_training(f"cuda:{g}", cli.smoke, cli.retention_all, cli.output_dir,
                      batch_size=cli.batch_size, resume=cli.resume,
                      old_ckpt=cli.fotg_old_ckpt)
    sys.exit(rc if rc is not None else 0)


if __name__ == "__main__":
    main()
