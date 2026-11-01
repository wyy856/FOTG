# MFFC-F：多通道功能纤维约束（新版 F 模块）

> 本目录是 **新建代码**，不改动 O1O-main 源文件。仅把本目录整体复制到运行节点即可启动。

---

## 1. 方法定位

OWOD/O1O 持续学习中，旧 F 模块（F-A 功能保持 / F-B Jacobian 软投影 / PCA-FAB）
都是**直接约束共享 backbone θ（或特征空间梯度）**，等价于"冻结旧类表征"。
这必然与"新类需要 backbone 自由度"冲突 → PK↑ 以 CK↓ 为代价（已实测：
PCA-FAB task2_train PK +20.89 / CK −3.82；old fab task2_ft PK +2.41 / CK −1.65）。

**MFFC-F（Multi-channel Functional Fibre Constraint）的红线：保护旧类 *功能*，不冻结旧类 *表征*。**
即在"旧类输入存在"的子空间上保持旧类功能，其余方向允许表征自由漂移以学新类。

---

## 2. 三通道功能约束 + L_fib

| 通道 | 名称 | 约束内容 | 是否需旧类输入 |
|------|------|----------|----------------|
| **F1** | Identity 身份 | 学生/教师对同一旧类的 **Top-K 旧类竞争类 margin** 对齐（蒸馏旧类判别结构） | 是 |
| **F2** | Foreground 前景 | 学生/教师 **旧类 vs no-object margin** 对齐（保持"这是前景旧类"的判定） | 是 |
| **F3** | Localization 定位 | 学生/教师在 **GT 对齐的同一预测框** 上 bbox SmoothL1（保持旧类定位能力） | 是 |
| **L_fib** | 局部 Jacobian 功能纤维 | 惩罚学生特征沿"在教师处改变旧类功能"的方向漂移（不做固定 PCA 方向硬投影） | 否（几何正则） |

总损失：
```
L_F = λ_id·L_id + λ_fg·L_fg + λ_loc·L_loc + λ_fib·L_fib
```
教师侧全部 `.detach()`，梯度只回流学生；**表征不被冻结**，仅旧类功能被保持。

### 三个实现边界（来自方法定义，已落实）
1. **需旧类输入**：仅在"已匹配到旧类 GT 的 query"上计算；旧类样本缺失时该通道=0（不引入噪声）。
2. **DETR 按 GT 对齐**：学生与教师各自独立匹配到同一 GT（`gt_aligned_pairs`，IoU≥阈值），
   再比较对应预测，**不依赖固定 query slot**。
3. **三通道独立归一化**：分类 margin 与 bbox 坐标尺度不同，先在 `L_fib` 内按 `W_F`（fib_w_*）独立归一化，
   再拼成功能向量求 Jacobian，不把异尺度量直接拼成一个平方误差。

---

## 3. 文件清单

| 文件 | 作用 | 是否改动源 |
|------|------|-----------|
| `mffc_f.py` | MFFC-F 核心模块（`MFFCFConstraint` + `gt_aligned_pairs` + `_iou_1toN`） | 新建 |
| `engine.py` | O1O-main `engine.py` 副本，仅在 FOTG 块内 **MARGIN 之后** 插入 MFFC-F 调用（flag 门控 `--mffc_f_on`） | 副本+插入 |
| `main_open_world_o1o.py` | O1O-main `main_open_world_o1o.py` 副本，新增 `--mffc_f_*` 参数；并在 `--mffc_f_on` 时一并构建 teacher（criterion.fotg） | 副本+插入 |
| `run_mffc.py` | 启动器（基于 `owod_run_fab.py`），新增 `--mffc_f_*` CLI，默认 **MFFC-F-only**（原 fotg 约束全置 0） | 副本+插入 |
| `README.md` | 本文件 | 新建 |

源：`/icislab/volume1/wyy/O1O-main` 保持纯净，仅作 `models/util/datasets/fotg_module/fotg_losses` 等依赖的参照（经 PYTHONPATH 复用）。

---

## 4. 部署

```bash
# 备份（不改动源）
mkdir -p /icislab/volume1/wyy/mffc_f
cp mffc_f.py engine.py main_open_world_o1o.py run_mffc.py README.md /icislab/volume1/wyy/mffc_f/

# 启动副本（/dev/shm 为 tmpfs，跑前确保有空间；长跑需每阶段后归档）
mkdir -p /dev/shm/mffc_f
cp mffc_f.py engine.py main_open_world_o1o.py run_mffc.py README.md /dev/shm/mffc_f/
```

---

## 5. 启动

```bash
cd /dev/shm/mffc_f
# 单阶段（如 task2_train，索引 1）
python -u run_mffc.py --stage 1 --gpus 5

# 全链路自链（task1_train -> task4_ft）
python -u run_mffc.py --all --gpus 5

# 多卡（DDP）
python -u run_mffc.py --stage 1 --gpus 3,5 --num-gpus 2

# 冒烟（限 N 个 batch，跳过 eval，验证能否启动+跑几步）
python -u run_mffc.py --stage 1 --gpus 5 --smoke 5
```

启动器自动把 `--code-dir` 设为本脚本所在目录，运行时从 `code_dir` 加载
`main_open_world_o1o.py / engine.py / mffc_f.py`，其余模块从 `PROJ`（O1O-main）加载。

### 默认 MFFC-F-only 口径
`run_mffc.py` 默认 `--mffc_f_on 1`，并注入 `MFFC_FLAGS`（原 fotg 的
`lam_fib/ov/topo/route/unk` 全置 0、`dual=0`、`accum_every=0`），
使原 fotg 自身损失归零、只有 MFFC-F 生效；F-B / PCA-FAB 投影关闭。
teacher 由本 task 的 `old_ckpt`（上一任务权重）加载，提供正确旧类功能参照系。

### 关键超参（可用 CLI 调）
`--mffc-f-lam-id/-fg/-loc`（默认 1.0/1.0/1.0）、`--mffc-f-lam-fib`（默认 0.1）、
`--mffc-f-w`（总权重，默认 1.0）、`--mffc-f-topk`（5）、`--mffc-f-iou`（0.5）、
`--mffc-f-temp`（1.0）、`--mffc-f-loc-scale`（1.0）、`--mffc-f-fib-w-fg/-loc`（1.0）。

---

## 6. 监控指标

每个 step 会在 `loss_dict` 记录：
- `mffc_L_id / mffc_L_fg / mffc_L_loc / mffc_L_fib`：各通道分量（detach 后）
- `mffc_n_pairs`：本 step 参与计算的旧类 GT 对齐对数量（=0 说明该 step 无旧类 GT，MFFC-F 为 0）

**健康闸门**：`mffc_n_pairs` 在 task2+ 训练中应稳定 > 0（exemplar replay 保证旧类样本在场）；
若长期为 0，检查 GT 标签口径（旧类标签 < prev）与 IoU 阈值。

---

## 7. 验证计划（与 F-A/F-B/PCA-FAB 对照）

1. **复现 baseline 口径**：用同源 `--mffc-f-on 0`（或 FAB 配置）跑 task2_train / task2_ft，
   锚定 PK/CK/U 三项（与已测得 PCA-FAB / old fab 数字对齐）。
2. **MFFC-F-only**：同数据同 seed 跑，比较 task2_train→task4 的 PK（保持度）/ CK（新类损失）/ U（未知发现）。
3. **判定准则（pre-registered）**：
   - GO：PK 保持接近 FAB 水平 **且** CK 下降明显小于 FAB（CK drop 收敛到 ≤ ~1.0，理想持平）；
   - NO-GO：CK 仍与 FAB 同等幅度下降（说明"功能保持"仍挤占新类学习）→ 回到 F1/F2/F3 权重配比或放宽定位通道。
4. **消融**：逐个关闭 F1/F2/F3（置对应 λ=0），定位各通道对 PK/CK 的贡献。

---

## 8. 已知设计取舍 / 后续可细化

- **L_fib 的 `g_loc` 采用 `bbox_embed(z)` 的原始 delta 输出**，而非最终 `pred_boxes`（绝对框）。
  二者同属"旧类定位功能"但坐标系不同；L_fib 是几何正则（λ_fib 默认 0.1，小权重），
  同一 head_fn 自洽，不影响 F3（F3 用绝对框 SmoothL1）。若需严格对齐 F3 空间，
  可在 `head_fn` 中再叠加 reference-point 反变换（待验证收益后决定）。
- **`head_fn` 使用当前学生读头 `class_embed/bbox_embed[-1]`**：L_fib 度量"当前读头下沿特征方向的旧类功能变化"，
  与 F1/F2/F3 一致。teacher head 不用于 L_fib 的 Jacobian（避免旧 head 与新 head 错位）。
- **teacher 前向额外一次**：与 FAB 同构（每 step 多一次冻结旧模型前向），开销可接受。
