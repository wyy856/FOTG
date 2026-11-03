# FMP B run 代码包（unk_head 修复版）

**来源**：服务器 `/dev/shm/unkhead/`（B run 实际加载的代码，逐字节一致）。
**模型文件 md5**：`e65158910e08ae710ef5cd4fab1baf2f`（修复版；被 DN 错位 bug 污染的旧版备份在服务器 `/dev/shm/unkhead/proj/models/dn_dab_deformable_detr_o1o.py.bug_dn_misalign`）。

## 目录结构

```
fmp_b_code/
├── README.md                                  # 本文件
├── main_open_world_o1o.py                     # 训练入口（FMP wrapper）
├── run_fmp.py / fmp.py                        # FMP 运行辅助 + 核心几何项
├── fotg_module.py / fotg_losses.py            # FOTG 模块与损失
├── engine.py                                  # 训练/评估循环（与原版 0 diff，未改）
├── proj/models/dn_dab_deformable_detr_o1o.py  # 唯一改动的模型文件（含 unk_head + DN 裁剪修复）
└── diffs/
    ├── dn_dab__unkhead_edits_vs_orig.diff     # 222 行：unk_head 全部改动 + DN 错位 BUGFIX（重点看）
    ├── main_open_world_o1o__vs_orig.diff      # 53 行：入口 resume/old_model 逻辑
    ├── engine__vs_orig.diff                   # 0 行：engine 未改
    └── dn_dab__vs_volume1_original.diff       # 831 行：相对 volume1 原始 O1O-main 的全部差异
```

## 关键 BUGFIX：pred_unk 的 DN query 错位（2026-11-03）

**位置**：`proj/models/dn_dab_deformable_detr_o1o.py` forward 内 `dn_post_process` 调用处（diff 62–73 行）。

**问题**：DN queries 位于 query 序列**前** `pad_size` 个位置，`dn_post_process` 对 class/coord/superclass 做了裁剪，但 `outputs_unk` 未裁剪。`loss_unk` 的 matcher 索引在裁剪后坐标系里，写入未裁剪的 `tgt` 时全部错位——实测（错位探针）**unk 正样本 135/135 全落在 DN query 区（100%），normal query 收到的全是弱负监督**。unk_head 学到的是"DN query→1"（DN 区 median 0.616 vs normal 区 0.017），推理时无 DN query → 全输出先验低分。含 pseudo-unknown 的图 pad_size=105–115 > Q_normal=100，错位是系统性的。

**修复**：对 `outputs_unk` 做与其他 head 完全相同的 DN 裁剪 + shape 断言：

```python
if outputs_unk is not None:
    _pad = mask_dict.get('pad_size', 0) if mask_dict else 0
    if _pad > 0:
        outputs_unk = outputs_unk[:, :, _pad:, :]
    assert outputs_unk.shape[1:3] == outputs_class.shape[1:3], ...
```

**验证**：修复后探针 `pred_unk Q=100 == pred_logits Q=100`，断言全 batch 通过。旧 checkpoint 的 unk_head 权重已被污染（ep30/31 训练全程错位），因此 **B run 已从 task1 ep29 clean restart**（GPU3/4，`/dev/shm/unkhead_out/task2_train`）。

## 机制速记

- **unk_head**：`Linear(hidden_dim,1)`，每层 decoder 输出各接一个；监督 = 匹配 unknown GT 的 query 正样本(w=1)、匹配 known GT 强负(w=1)、未匹配背景弱负(w=0.1)。`unk_head_w=1.0`。
- **训练期官方指标（eval_mode=False）不用 pred_unk**：unknown 预测来自分类头 channel 80（伪标签路径）。unk_head 只在 eval_mode=True 生效 → A/B 同口径可比，但 unk_head 独立效果需专门用 eval_mode=True 评估。
- **DN 路径**：`prepare_for_dn`（`proj/models/dn_components_o1o.py:67`，硬编码 `.cuda()`）；推理传 `dn_args=num_patterns`(int)。

## B run 启动命令（当前 run，GPU3/4）

```bash
# /dev/shm/unkhead/relaunch_b.sh（完整 132 参数存于 relaunch_cmd.txt）
export PYTHONPATH=/dev/shm/unkhead/proj
export CUDA_VISIBLE_DEVICES=3,4
# torch.distributed.run --nproc_per_node 2 --master_port 29537 ... main_open_world_o1o.py
#   --output_dir /dev/shm/unkhead_out/task2_train --epochs 40 --use_dn --unk_head_w 1.0
#   --resume .../o1o_mowod_task1_train/checkpoint0029.pth (+fmp_* 全套开关)
```

实时日志：`tail -f /dev/shm/unkhead_out/task2_train/tb/*/attempt_0/0/stdout.log | grep --line-buffered -E 'Epoch: \[|test_metrics'`

## 看代码建议顺序

1. `diffs/dn_dab__unkhead_edits_vs_orig.diff` — unk_head 全部改动 + BUGFIX；
2. `fmp.py` + `fotg_module.py` — FMP 核心机制；
3. `diffs/main_open_world_o1o__vs_orig.diff` — 入口改动。
