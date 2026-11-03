# FMP 独立 Unknown/Objectness Head — 最小改动方案 + 预注册闸门

> 目标：known mAP 高 **且** unknown recall(U_R50) 不掉。
> 根因（已确认）：当前 `unk_prob = 1 − Σp_known` 是 known 置信度的残差，FMP 锐化 known 必然压低 unk → 数学上无法两者兼得。解法是把 unknown 从 known 残差里**解耦**成独立打分通道。
> 原则：known 分支 + FMP 投影**原样不动**（additive 改动），仅新增 unk 通道与 unk 监督；flag 门控，对当前/baseline run 字节级兼容。

---

## 0. 关键事实（服务器实查，2026-11-03）

1. **unknown 监督信号已存在**：`O1O-main/datasets/torchvision_datasets/open_world_o1o.py`
   - `merge_pseudo_instances_nms` (L389–449) 把 pseudo unknown 框以 `category_id = unk_idx = len(CLASS_NAMES)-1` 合入训练 target（非 replay 图：先删 known/unk 实例，再 merge 回 pseudo-unk；replay 图保留自身实例）。
   - 即 `targets['labels']` 在训练时**含有 unk_idx 框** → 独立 unk head 可直接用现有 pseudo 标签监督，**无需新数据/新标注管线**。
2. **当前 unk 被当成 no-object**：`unk_idx == num_classes-1 == 无物体索引`（SetCriterion `target_classes` 默认填 `num_classes-1`）。所以已知类 matcher 把 unk 框当背景，不参与已知类训练——这也正是残差公式 `unk_prob = 背景概率 = 1−Σp_known` 生效的原因。
3. **detector 与 evaluator FMP/O1O-main 完全共用同一份**（`dn_dab_deformable_detr_o1o.py` / `open_world_eval.py`），FMP 无本地副本 → 改 detector 需 flag 门控，保证 baseline 字节级不变。
4. **现存 `fotg_unk` 损失路径**（`engine.py` L354–360，gated by `fotg_unk_w>0`）用几何记忆(`compute_unk`)做 unk 监督，但当前 `fotg_on=0` 不触发。本方案不复用它，改为更干净的独立 head（与已知类解耦更彻底）。

---

## 1. 精确插入点（文件 : 行号，服务器已核对）

### 文件 A：`O1O-main/models/dn_dab_deformable_detr_o1o.py`（FMP 经 sys.path 共用）

**(A1) 头定义 — `__init__` 紧接 `self.class_embed` (L107)**
```python
self.class_embed = nn.Linear(hidden_dim, num_classes)      # 现有，不动
self.unk_embed   = nn.Linear(hidden_dim, 1)                # 新增：独立 unk 通道
```
并镜像 decoder 级克隆（L179–193）：
```python
self.unk_embed = _get_clones(self.unk_embed, num_pred)     # 或 ModuleList，与 class_embed 同构
...
self.transformer.decoder.unk_embed = self.unk_embed        # L197 同款
```

**(A2) `forward` 每层输出 — L275–321 循环内**
```python
outputs_class = self.class_embed[lvl](hs[lvl])             # 现有
outputs_unk   = self.unk_embed[lvl](hs[lvl])              # 新增
...
out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1],
       'pred_unk': outputs_unk[-1], ...}                  # 加 pred_unk
# aux_loss：中间层也 emit pred_unk（保持与 class_embed 一致）
```

**(A3) `PostProcess.forward` eval_mode 分支 — L567–577（**向后兼容**）**
```python
if 'pred_unk' in outputs:
    unk_prob = outputs['pred_unk'].sigmoid()               # 解耦：独立通道
else:                                                       # 旧 run 无 pred_unk 时回退残差
    msp_wcls = out_logits_sigmoid.sum(-1)
    unk_prob = (1 - msp_wcls).unsqueeze(-1)
unk_prob = torch.where(unk_prob < eval_threshold, 0.0, unk_prob)
out_logits_sigmoid = torch.cat([out_logits_sigmoid, unk_prob], dim=-1)
```
→ `pred_unk` 存在即用独立通道；不存在则原残差逻辑，**当前/baseline run 字节级不变**。

### 文件 B：`O1O-main/models/dn_dab_deformable_detr_o1o.py` SetCriterion（或新建 `unk_head_loss` 模块）

新增 unk 监督损失，**不碰已知类 matcher**：
- **B1（推荐，无第二 matcher）**：IoU 阈值密集分配。对每张图，用 `outputs['pred_boxes']` 与各 unk_idx GT 框算 IoU；某 query 的 `unk_target=1` 当且仅当 `max IoU(该 query 预测框, 任意 unk GT) ≥ 0.5`。
  ```python
  L_unk = sigmoid_focal_loss(pred_unk.view(-1), unk_target.view(-1), num_boxes, ...)  # 复用 L46 的 focal
  ```
- **B2（备选，匈牙利）**：单独对 `(pred_unk, unk_boxes)` 跑 1 类二分匹配。更标准但改动略大。

### 文件 C：`FMP/engine.py`（L354–360 `fotg_unk` 块之后，并行新增）
```python
if getattr(args, 'unk_head_w', 0.0) > 0 and 'pred_unk' in outputs:
    _lu = compute_unk_head_loss(outputs, targets)          # 见 B1
    loss_dict['unk_head'] = args.unk_head_w * _lu
    losses = losses + loss_dict['unk_head']
# 已知类 loss + FMP loss 此处完全不变
```

### 文件 D：`FMP/main_open_world_o1o.py`（L282 `fotg_unk_w` 参数附近）
```python
parser.add_argument('--unk_head_w', default=0.0, type=float,
                    help='独立 unk head 的 focal/bce 权重；0=关闭(字节级兼容旧 run)')
```

---

## 2. 改动范围与风险（诚实清单）

| 项 | 是否改动 | 说明 |
|---|---|---|
| 已知类 `class_embed` / 已知类 loss | ❌ 不动 | known mAP 由它决定，原样保留 |
| FMP projection（geo/id/fg/loc） | ❌ 不动 | 抗遗忘核心，原样保留 |
| decoder `hs` 特征 | ⚠️ 共享 | unk head 与 known head 共用 `hs` → unk 梯度回传可能微扰共享特征（**唯一真实风险**，见闸门） |
| detector 文件 | ✅ 改（flag 门控） | 影响 FMP 与 O1O-main 共用文件；flag 关时字节级兼容 |
| 数据/标注 | ❌ 不动 | pseudo unk GT 已现成 |

**风险缓释**：
- unk head 仅新增参数 `unk_embed`（+1 线性/层），容量开销可忽略；
- FMP 投影把 known 特征钉在 task1 流形，构成对共享特征漂移的稳定器；
- 预注册闸门显式捕捉 known 退化。

---

## 3. 预注册闸门（GO / NO-GO，严格按你惯用 discipline）

在 **FMP task2 eval（对照当前最佳 ep36 及之后）** 上判定，三项**全过**才 GO：

| 指标 | 通过阈值 | 依据 |
|---|---|---|
| **PK_AP50** | `≥ 42.0` | 抗遗忘核心收益，不可退化（当前 ~43.6） |
| **CK_AP50** | `≥ 39.0` | 当前任务学习不掉（当前 ep36=35.2，ep39 baseline=39.7） |
| **U_R50** | `≥ 50.0` | 目标：从 ~38 拉回 baseline ~55 水平 |

次级护栏：`K_AP50` 不跌破 ~39；`WI` 不劣于 ~0.024。

**STOP / 决策分支**：
- `PK < 42` → **NO-GO**，unk head 伤了 known 保持 → 回退，不再推进；
- `PK/CK` 达标但 `U_R50` 未升 → **中性**，说明解耦未生效（检查 unk_target 是否正确、unk_head_w 是否够大、pseudo 框是否过少 `num_unk_objects=20`）→ 调权重或换 B2，不直接判失败；
- 三项全过 → **GO**，升格为完整 40-epoch run。

---

## 4. 对照 run 设计（最小干预、单变量）

- **基底**：复制当前 run 全部 args（取自 `/proc/1053679/cmdline`），仅增 `--unk_head_w <w>`（先扫 `1.0`、`2.0`）。保留 `--fmp_on 1 --fmp_project_on 1`，所有 `fotg_*`、`margin`、`fab` 保持 0。
- **新 `output_dir`**：`/dev/shm/fmp_t2_unkhead/`，**不覆盖**当前 run（让其跑完 ep37–40，作为干净 FMP-vs-baseline 对照）。
- **启动前 sanity check（你定的规矩）**：
  1. dry-run 1 step，确认 `pred_unk` 出现在 `outputs` 且 `loss_dict['unk_head']` 出现；
  2. 梯度检查：`unk_embed` 有梯度；对比 `class_embed` 梯度范数与关闭 unk head 时一致（确认 known 分支梯度未被改）；
  3. 确认 `unk_target` 非空（pseudo unk 框确有进入）。
- **顺序**：先跑一个 `unk_head_w` 值 → ep36 当量评估 → 上闸门 → 仅当 GO 再扫下一档。
- **绝不**动 FMP / known head / decoder 结构（严禁 broad unfreeze）。

---

## 5. 待你确认后才能落地的事项

1. detector 是 FMP/O1O-main **共用文件**，改前需 `cp` 备份（按你 md5 习惯）；是否同意我先做备份 + flag 门控改动？
2. unk 监督用 **B1（IoU 密集，无第二 matcher）** 还是 **B2（匈牙利 1 类匹配）**？我倾向 B1（最小改动）。
3. 首档 `unk_head_w` 用 `1.0` 还是直接 `2.0`？
4. 是否同意在独立 `output_dir` 起对照 run（不动当前 ep37–40 训练）？

> 当前在跑的 FMP run（ep37，GPU3/4）**保持不动**，本方案与其严格解耦，仅作后续对照。
