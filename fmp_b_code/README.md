# FMP B run（FMP + 独立 unk_head）代码整理

来源：服务器 172.29.2.188 `/dev/shm/unkhead/`（fmp/ + proj/models），即 B run
（`/dev/shm/unkhead_out/task2_train/`，unk_head_w=1.0）实际加载的代码。
打包时间：2026-11-03 15:03。

## 目录结构

```
fmp_b_code/
├── main_open_world_o1o.py      # 训练入口（FMP 版 wrapper，--mode 驱动 task1/task2）
├── run_fmp.py                  # FMP 运行辅助
├── fmp.py                      # FMP 核心（feature-map probe / 蒸馏几何项）
├── fotg_module.py              # FOTG 模块
├── fotg_losses.py              # FOTG 损失
├── engine.py                   # 训练/评估循环（与原版完全一致，0 diff）
└── proj/models/
    └── dn_dab_deformable_detr_o1o.py   # 检测器模型（唯一改动的 proj 文件，含 unk_head）
```

注：`proj/` 其余所有文件（datasets / util / open_world_eval 等）与
volume1 原始 O1O-main **逐字节一致**，未包含在本包内。

## 与原始版本的三份 diff（diffs/）

| 文件 | 内容 | 行数 |
|---|---|---|
| `dn_dab__unkhead_edits_vs_orig.diff` | **unk_head 专属改动**（vs 修改前备份 .orig）——看这个最快 | 209 |
| `dn_dab_deformable_detr_o1o__vs_volume1.diff` | 模型 vs volume1 O1O-main 当前版（含更早的改动累积） | 819 |
| `main_open_world_o1o__vs_orig.diff` | 入口脚本相对原版的改动（resume / old_model 加载逻辑等） | 53 |
| `engine__vs_orig.diff` | 空（engine 未改动） | 0 |

## B run 启动命令（服务器实测）

```bash
cd /dev/shm/unkhead/proj
CUDA_VISIBLE_DEVICES=3,4 torchrun --nproc_per_node=2 \
  /dev/shm/unkhead/fmp/main_open_world_o1o.py \
  --output_dir /dev/shm/unkhead_out/task2_train \
  --data_root data/OWOD --model_type dn_dab_deformable_detr_o1o \
  --train_set owod_t2_train --test_set owod_all_task_test_nodup \
  --dataset TOWOD --PREV_INTRODUCED_CLS 20 --CUR_INTRODUCED_CLS 20 \
  --prev_num_superclass 6 --cur_num_superclass 3 --num_superclass 13 \
  --batch_size 2 --eval_every 1 --lr 5e-06 --epochs 40 --use_dn \
  --empty_weight 0.0 --sup_loss_coef 2 --lr_drop 50 \
  --pseudo_path data/OWOD/pseudo-boxes/TOWOD/mowod_until_t2_normal_pseudo... \
  --unk_head_w 1.0        # B run 关键 flag（A run 为 0.0，走残差 unk 路径）
```

（完整命令以 stdout.log 首行 Namespace 为准）

## 关键机制速记（供 review 对照）

1. **unk_head**：独立 BCE 头（`unk_head_w>0` 启用），与分类头共享 hs，
   Hungarian matching 到 unknown GT 的 query 设 tgt=1。
2. **已知坑**：`unk_idx(80) == no_object_index(80)` 标签冲突——分类头把
   unknown 正样本当 no-object 监督，unk_head 只是叠加分离信号（additive），
   未移除错误监督。
3. **resume 结构**：task1 在 `exps/o1o_mowod_task1_train`（ep29 止），
   task2_train = epochs 30–39；启动时 `start_epoch→30`，
   `old_model`（蒸馏/冻结参考）从 task1 ckpt0029 加载。
4. **A run 的 1e22 伪影**：`train_tgt_loss_sup_ce_4`（DN superclass CE），
   B run 无此问题（tgt_loss_sup_ce=0.11 正常）。
