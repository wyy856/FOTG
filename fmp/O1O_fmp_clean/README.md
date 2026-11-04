# O1O + FMP 精简代码包

DAB-Deformable-DETR（O1O 框架）训练/评测代码，含 FMP（功能流形保持 + 新类梯度投影）。
已移除 unk_head / fg_head 相关全部代码（argparse、模型头、损失、优化器分组）。

## 文件组成
- `main_open_world_o1o.py`  训练/评测入口，`--eval` 走单跑评测
- `engine.py`               `train_one_epoch` / `evaluate`，含 FMP 调用点与 teacher 前向
- `fmp.py`                  FMP 实现（几何损失 / 功能 Jacobian / 零空间投影）
- `fotg_losses.py` `fotg_module.py`  FOTG 约束集，默认由 `--fotg_on 0` 关闭，仅作 teacher 支持
- `run_fmp.py`              四阶段（task1→4）链路启动器；`run_fmp_ctrl.py` 为同构变体（仅端口/日志路径不同）
- `models/` `datasets/` `util/`  检测器、数据管线、工具
- `models/ops/`             deformable attention 的 CUDA 扩展源码

## 运行前需要自行补齐（包内未含，避免体积）
1. 数据：`<项目根>/data/OWOD/`（TOWOD，含 JPEGImages / Annotations / ImageSets）
2. 骨干权重：`<项目根>/models/dino_resnet50_pretrain.pth`
3. CUDA 扩展：`cd models/ops && sh make.sh`
4. 运行目录默认为 `./data/OWOD`（`--data_root` 可改）

## 需要改成本机路径的常量
`run_fmp.py` 与 `run_fmp_ctrl.py` 顶部：
- `PROJ`（训练工作目录）、`SUPERVISOR_LOG`
- `STAGES[*].out`（输出目录）、`STAGES[*].resume` / `old_ckpt`（上阶段 ckpt）、`STAGES[*].pseudo`（伪标签 json）

## 怎么跑
唯一的权威启动参数就是 `run_fmp.py` 里 `STAGES` + `build_argv()` 拼出来的那串：
单卡调试直接读 `build_argv(idx, "0")` 的返回列表，逐项传给 `main_open_world_o1o.py`。
DDP 走 `torchrun --nproc_per_node N --master_port <port> main_open_world_o1o.py <同上参数>`。

## 备注
- 包内文件与服务器运行副本逐字节一致，校验值见 `MD5SUMS.txt`。
- `engine.py` 里 `FOTG_PROBE=1` 时会 import `fotg_fa_probe`（探针用，未随包提供）；不设该环境变量不会触发。
