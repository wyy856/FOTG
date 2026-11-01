#!/usr/bin/env bash
# =============================================================================
# launch_mffc_chain.sh — MFFC-F 全链路对照启动器（task2_train → task2_ft）
# -----------------------------------------------------------------------------
# 设计目标：逐项镜像 gpu3/4 上正在跑的 PCA-FAB 实验，只把 F 模块从 pca_fab 换成
#           MFFC-F（其余 fotg 约束全部置 0，干净 MFFC-F-only），其余配置 1:1 对齐：
#
#   FAB (gpu3/4)               |  MFFC-F (本脚本)
#   --------------------------- |  ---------------------------
#   task2_train: lr 5e-6 (2e-5×0.25) |  task2_train: lr 5e-6  (lr_scale=0.25)
#   task2_ft:   lr 5e-5 (2e-4×0.25) |  task2_ft:   lr 5e-5  (lr_scale=0.25)
#   batch 每卡 2 / 全局 4       |  batch 每卡 2 / 全局 4 (num_gpus=2)
#   pca_fab_on 1 (train+ft)    |  mffc_f_on 1   (train+ft, F1+F2+F3, L_fib=0)
#   resume task1/ckpt0029       |  resume task1/ckpt0029
#   epochs 40 / 50              |  epochs 40 / 50
#   输出 /dev/shm/pca_fab_chain |  输出 /dev/shm/mffc_f_chain
#
# 第一组实验配置（用户指定）：
#   F-A (fotg_on)           = 0 关闭（MFFC_FLAGS 置零，engine 里 _fa_on gate 不执行）
#   旧 Jacobian-F-B (fab_on) = 0 关闭
#   PCA-F-B (pca_fab_on)     = 0 关闭
#   MFFC-F                   = 1，仅 F1+F2+F3（--mffc-f-lam-fib 0，L_fib 不开）
#   margin_on                = 0 关闭
#   Detection loss           = 正常开启（不动原损失）
#
# 注意：ft 与 train 的 lr 不一样（5e-5 vs 5e-6），本脚本通过 lr_scale=0.25 在
#       两个 STAGES 默认 lr 上分别得到，避免误用单一 lr。
#
# 用法：
#   bash /dev/shm/mffc_f/launch_mffc_chain.sh
#   GPUS=0,1 bash /dev/shm/mffc_f/launch_mffc_chain.sh   # 指定其它卡
# =============================================================================
set -u

PROJ="/icislab/volume1/wyy/O1O-main"
CODE="/dev/shm/mffc_f"                       # MFFC-F 运行代码目录（engine/main/run_mffc/mffc_f）
PY="/icislab/volume2/wyy/conda/envs/rfdetr/bin/python"
T1CKPT="/icislab/volume1/wyy/O1O-main/exps/o1o_mowod_task1_train/checkpoint0029.pth"
OUT="/dev/shm/mffc_f_chain"
GPUS="${GPUS:-3,4}"                          # 默认 gpu3,4（与你 FAB 同槽）；可 GPUS=0,1 指定
NGPU=2
LR_SCALE=0.25                                # → train 5e-6, ft 5e-5（FAB 实测 lr）

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$OUT/chain.log"; }

# ---- 环境（避坑：坏 user-site scipy 会让 import 崩；cwd 让 data_root 解析） ----
export PYTHONNOUSERSITE=1
export WANDB_MODE=offline
export TMPDIR=/dev/shm/tmp
mkdir -p "$OUT" /dev/shm/tmp
cd "$PROJ" || { echo "[FATAL] cd $PROJ 失败"; exit 1; }

# 前置断言
[ -f "$CODE/run_mffc.py" ] || { echo "[FATAL] 缺少 $CODE/run_mffc.py"; exit 1; }
[ -f "$T1CKPT" ]          || { echo "[FATAL] 缺少 teacher ckpt $T1CKPT"; exit 1; }

log "=== MFFC-F 全链路启动：GPU=$GPUS NGPU=$NGPU LR_SCALE=$LR_SCALE OUT=$OUT ==="

# -----------------------------------------------------------------------------
# 阶段 1：MFFC-F task2_train（lr=5e-6, 40 epoch, resume task1/ckpt0029, mffc_f_on）
# -----------------------------------------------------------------------------
log "=== [1/2] MFFC-F task2_train (lr=5e-6, 40ep) ==="
$PY -u "$CODE/run_mffc.py" \
    --stage 1 --gpus "$GPUS" --num-gpus $NGPU \
    --lr-scale $LR_SCALE --mffc-f-on 1 --mffc-f-lam-fib 0 \
    --out-root "$OUT" 2>&1 | tee -a "$OUT/train.log"
rc1=${PIPESTATUS[0]}
if [ $rc1 -ne 0 ]; then log "[FATAL] task2_train 失败 rc=$rc1"; exit 1; fi

TRAIN_CKPT="$OUT/task2_train/checkpoint0039.pth"
[ -f "$TRAIN_CKPT" ] || { log "[FATAL] 未找到 $TRAIN_CKPT（task2_train 应产出 checkpoint0039）"; exit 1; }
log "=== task2_train 完成 -> $TRAIN_CKPT ==="

# -----------------------------------------------------------------------------
# 阶段 2：MFFC-F task2_ft（lr=5e-5, 50 epoch, resume 自有 train ckpt, mffc_f_on）
# -----------------------------------------------------------------------------
log "=== [2/2] MFFC-F task2_ft (lr=5e-5, 50ep) ==="
$PY -u "$CODE/run_mffc.py" \
    --stage 2 --gpus "$GPUS" --num-gpus $NGPU \
    --lr-scale $LR_SCALE --mffc-f-on 1 --mffc-f-lam-fib 0 \
    --out-root "$OUT" \
    --resume-override "$TRAIN_CKPT" \
    --old-ckpt-override "$T1CKPT" 2>&1 | tee -a "$OUT/ft.log"
rc2=${PIPESTATUS[0]}
if [ $rc2 -ne 0 ]; then log "[FATAL] task2_ft 失败 rc=$rc2"; exit 1; fi

log "=== MFFC-F 全链路完成 ✅ 输出: $OUT/task2_train + $OUT/task2_ft ==="
log "=== 对照 FAB baseline: PK~58.1 / CK~36.9 (task2_ft) 来自 /dev/shm/pca_fab_chain/task2_ft/log.txt ==="
