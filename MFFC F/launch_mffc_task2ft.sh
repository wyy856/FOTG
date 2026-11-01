#!/usr/bin/env bash
# =====================================================================
# MFFC-F  task2_ft 启动脚本
# 配置严格对齐 gpu3/4 上正在跑的 FAB（同 batch_size=2 / lr=5e-5 / 2卡 / 同 resume 与 old_ckpt）
# 仅把 F 模块从 PCA-FAB 换成 MFFC-F（其余 fotg_lam_* 置 0，只留 MFFC-F 生效）
#
# 前置条件：gpu3/4 上的 FAB 已跑完或已停止。
#   CUDA_VISIBLE_DEVICES=3,4 让 torchrun(--nproc_per_node 2) 占 3/4。
# 启动：
#   bash /dev/shm/mffc_f/launch_mffc_task2ft.sh
# =====================================================================
set -eo pipefail

export CUDA_VISIBLE_DEVICES=3,4
export PYTHONNOUSERSITE=1            # 关键：否则 user-site 里坏的 scipy 会让 import 直接崩
PY=/icislab/volume2/wyy/conda/envs/rfdetr/bin/python
CODE=/dev/shm/mffc_f/main_open_world_o1o.py
CWD=/icislab/volume1/wyy/O1O-main    # data_root data/OWOD 在此解析
OUT=/dev/shm/mffc_f_chain/task2_ft
mkdir -p "$OUT"

cd "$CWD"
echo "[launch] CWD=$CWD  CODE=$CODE  OUT=$OUT  GPUS=$CUDA_VISIBLE_DEVICES"

$PY -m torch.distributed.run --nproc_per_node 2 --master_port 29527 "$CODE" \
  --output_dir "$OUT" \
  --data_root data/OWOD \
  --model_type dn_dab_deformable_detr_o1o \
  --train_set o1o_mowod_exemplars/learned_owod_t2_ft \
  --test_set owod_all_task_test_nodup \
  --dataset TOWOD \
  --PREV_INTRODUCED_CLS 20 \
  --CUR_INTRODUCED_CLS 20 \
  --prev_num_superclass 6 \
  --cur_num_superclass 3 \
  --num_superclass 13 \
  --batch_size 2 \
  --eval_every 5 \
  --lr 5e-05 \
  --epochs 60 \
  --use_dn \
  --empty_weight 0.0 \
  --sup_loss_coef 2 \
  --lr_drop 50 \
  --lr_drop_gamma 0.1 \
  --pseudo_path data/OWOD/pseudo-boxes/TOWOD/mowod_until_t2_normal_pseudo_labels.json \
  --pseudo_threshold 0.5 \
  --pseudo_nms_iou 0.5 \
  --num_unk_objects 20 \
  --device cuda \
  --num_workers 4 \
  --sep_loss_w 0.0 \
  --wandb_project "" \
  --num_gpus 2 \
  --resume /dev/shm/pca_fab_chain/task2_train/checkpoint0039.pth \
  --fotg_old_ckpt /icislab/volume1/wyy/O1O-main/exps/o1o_mowod_task1_train/checkpoint0029.pth \
  --fotg_on 0 \
  --fotg_dim 256 \
  --fotg_k_sub 8 \
  --fotg_w_threshold 0.0 \
  --fotg_w_topk 10 \
  --fotg_w_soft 1 \
  --fotg_lam_fib 0 --fotg_lam_ov 0 --fotg_lam_topo 0 --fotg_dual 0 \
  --fotg_eps_fib 1e-3 --fotg_eps_ov 1e9 --fotg_eps_topo 1e9 \
  --fotg_lam_route 0 --fotg_eps_route 0.05 --fotg_lam_unit_route 1.0 \
  --fotg_dual_lr_route 1.0 --fotg_lam_max_route 1e3 \
  --fotg_unk_w 0 --fotg_dual_every 20 --fotg_accum_every 20 \
  --fab_on 0 --fab_lambda 0.001 \
  --pca_fab_on 0 \
  --mffc_f_on 1 \
  --mffc_f_topk 5 --mffc_f_iou 0.5 \
  --mffc_f_lam_id 1.0 --mffc_f_lam_fg 1.0 --mffc_f_lam_loc 1.0 --mffc_f_lam_fib 0.1 \
  2>&1 | tee "$OUT/train.log"
