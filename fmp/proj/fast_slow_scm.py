"""
fast_slow_scm.py
================
CLS-inspired Fast-Slow dual visual encoder + Selective Consolidation Module (SCM)
for OWOD / O1O (DAB-Deformable-DETR family).

设计动机（与论文框架对应）:
    * Slow Network  (慢速通路): 继承 Task-1 权重，保持高稳定性（冻结/小学习率 + 特征蒸馏）。
                              输出 F_slow = 稳定旧类表征 + 渐进巩固得到的"慢速残差"。
    * Fast Network  (快速通路): 高可塑性，只有 Adapter 接收梯度，快速拟合 Task-2 新类与 Unknown。
                              输出 F_fast = 同一冻结基 + 自由学习的"快速残差"。
    * SCM          (位于 Fast 与 Slow 之间): 用响应打分/不确定性估计 M 筛选 Fast 中高置信 Novel 特征，
                              通过 Cross-Attention 把 Novel 知识渐进写入 Slow 空间（表征级巩固），
                              并通过 EMA 把 Fast adapter 权重渐进巩固至 Slow adapter（权重级巩固）。
    * 特征融合: F_final = F_slow + alpha * (M * (F_fast - F_slow))   [share_base=True，默认]
                = F_slow + alpha * (M * F_fast)                       [share_base=False，即论文原始写法]

接口契约（保证可嵌入 O1O，不改 checkpoint 格式）:
    * 本文件提供两个独立 nn.Module:
        - FastCNNBackbone : 单条通路 = 冻结 CNN 基 + 逐层 Bottleneck Adapter（可塑部分）。
        - SelectiveConsolidationModule : 单层 SCM（打分 + Cross-Attention + 融合）。
    * 模型侧仍保留原始 Joiner 作为 self.backbone（Slow 基 + pos embed + strides/num_channels），
      只新增 self.fast_path (FastCNNBackbone) 与 self.scm (ModuleList[SCM])。
      因此 checkpoint 键 `backbone.0.*` 仍原样载入 Slow 基，新增模块随机/零初始化。

张量约定:
    B = batch,  C = 通道,  H/W = 特征图空间尺寸,  N = H*W (展平后 token 数),
    L = 特征层级数 (ResNet-50 默认 3: C3/C4/C5, 通道 512/1024/2048)。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# O1O 的 NestedTensor 来自 util.misc；嵌入代码库时此导入应已可用。
try:
    from util.misc import NestedTensor
except Exception:  # pragma: no cover - 仅本地冒烟测试兜底
    from typing import Optional, Tuple

    class NestedTensor(object):
        """最小兼容实现：仅用于脱离 O1O 仓库时的本地形状测试。"""
        def __init__(self, tensors: torch.Tensor, mask: Optional[torch.Tensor]):
            self.tensors = tensors
            self.mask = mask

        def decompose(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
            return self.tensors, self.mask


# ---------------------------------------------------------------------------
# 1) FastCNNBackbone: 单条 Fast/Slow 通路
#    = 冻结 CNN 基（共享的 ImageNet / Task-1 权重）+ 逐层 Bottleneck Adapter。
#    可塑部分严格限制在 Adapter，符合"只有 Novel-slot 对应的参数或 Adapter 接收梯度"。
# ---------------------------------------------------------------------------
class FastCNNBackbone(nn.Module):
    def __init__(
        self,
        cnn: nn.Module,                       # 冻结 ResNet 基（如 Joiner[0] / build_backbone(...)[0]）
        levels: Tuple[int, ...] = (512, 1024, 2048),  # 各特征层通道数 (ResNet-50)
        bottleneck: int = 64,                 # Adapter 瓶颈维度
        freeze_base: bool = True,             # 基是否冻结（Fast/Slow 共用，均冻结 -> 可塑性只在 Adapter）
        init_zero: bool = True,               # Adapter 末层零初始化 -> 起点 = 基线（无残差）
    ):
        super().__init__()
        self.cnn = cnn
        self.levels = levels
        if freeze_base:
            for p in self.cnn.parameters():
                p.requires_grad_(False)
            self.cnn.eval()  # 冻结基用 running stats，推理/训练一致、稳定

        # 逐层 Bottleneck Adapter: x -> x + scale * (W_up(GELU(W_down(LN(x)))))
        # init-0 残差保证：在训练第一步，本通路输出 == 冻结基输出（baseline parity）。
        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(C),
                nn.Linear(C, bottleneck),
                nn.GELU(),
                nn.Linear(bottleneck, C),
            )
            for C in levels
        ])
        if init_zero:
            for m in self.adapters:
                nn.init.zeros_(m[-1].weight)
                nn.init.zeros_(m[-1].bias)
        # 每层一个可学习标量，控制该层残差幅度（起点 0 -> 无残差）。
        self.scale = nn.Parameter(torch.zeros(len(levels)))

    def forward(self, samples) -> list:
        """
        Args:   samples : NestedTensor 或能被 self.cnn 接受的输入
        Returns: feats   : List[NestedTensor]，长度 = len(levels)
                 feats[l].tensors 形状 = [B, C_l, H_l, W_l]
        说明: 冻结基始终 eval（用 running stats，不被训练 BN 漂移）；基前向在 no_grad 下完成，
              梯度只流向 Adapter（Fast 通路可塑，Slow 通路 adapter 无梯度 -> 仅 EMA 巩固）。
        """
        self.cnn.eval()
        with torch.no_grad():
            base_feats = self.cnn(samples)    # List[NestedTensor], 长度 = len(levels)
        return self.apply_adapter(base_feats)

    def apply_adapter(self, base_feats: list) -> list:
        """给定已算好的冻结基多层级特征（List[NestedTensor]），叠加本通路的 adapter 残差。
        模型侧复用同一份 base_feats（Fast/Slow 共享冻结基，只算一次基），避免重复前向。
        """
        out = []
        for l, feat in enumerate(base_feats):
            x, mask = feat.decompose()        # x: [B, C_l, H_l, W_l]; mask: [B, H_l, W_l] or None
            B, C, H, W = x.shape
            # 把空间特征当 token 过 Adapter: [B, C, H, W] -> [B*H*W, C] -> residual -> 还原
            flat = x.permute(0, 2, 3, 1).reshape(-1, C)          # [B*H*W, C]
            r = self.adapters[l](flat)                           # [B*H*W, C]
            r = r.reshape(B, H, W, C).permute(0, 3, 1, 2)        # [B, C, H, W]
            out.append(NestedTensor(x + self.scale[l] * r, mask))
        return out


# ---------------------------------------------------------------------------
# 2) SelectiveConsolidationModule: 单层 SCM
#    输入: F_slow [B,C,H,W], F_fast [B,C,H,W]
#    输出: F_final [B,C,H,W] + 诊断字典 {M, f_fast_sel, f_ca, f_delta, alpha}
# ---------------------------------------------------------------------------
class SelectiveConsolidationModule(nn.Module):
    def __init__(
        self,
        C: int,                               # 该层通道数
        num_heads: int = 8,
        share_base: bool = True,              # True: 融合用 delta(F_fast-F_slow); False: 直接加 F_fast
        alpha_init: float = 0.1,              # 融合权重初值（小 -> 起点接近纯 Slow）
    ):
        super().__init__()
        self.C = C
        self.share_base = share_base

        # (1) 响应打分 / 不确定性估计 -> 空间置信度图 M
        #     输入 = concat(F_fast, |F_fast - F_slow|)，即"Fast 响应"与"Fast/Slow 分歧"共同决定 Novel 候选。
        #     注意: 这里 M 是自包含的可学习 proxy；若已有 O1O unknown-head 的 U 图，可在此替换/拼接。
        self.scorer = nn.Sequential(
            nn.Conv2d(2 * C, C, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(C, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),                      # M in [0,1], 形状 [B,1,H,W]
        )

        # (2) Cross-Attention 渐进巩固: Slow 空间作为 Q，被 M 筛选的 Fast 作为 K/V
        #     把 "Novel 知识" 以 Slow 的坐标系写回 -> 表征级 consolidation。
        self.norm_s = nn.LayerNorm(C)
        self.norm_f = nn.LayerNorm(C)
        self.cross_attn = nn.MultiheadAttention(C, num_heads, batch_first=True)

        # (3) 融合权重 alpha（可学习标量，按层独立）
        self.alpha = nn.Parameter(torch.tensor(alpha_init))

        # (4) cross-attn 开关: scm_loss_w==0 时 consolidation_loss 不计入 loss，
        #     故 f_ca 这个 [B, heads, N, N] 注意力矩阵（高分辨率 C3 可达数 GiB）纯属浪费。
        #     默认 True（保留独立冒烟测试行为）；训练脚本在 scm_loss_w==0 时将其置 False 以省显存。
        self.use_cross_attn = True

    def forward(self, f_slow: torch.Tensor, f_fast: torch.Tensor,
                mask_slow: torch.Tensor = None) -> Tuple[torch.Tensor, dict]:
        """
        f_slow : [B, C, H, W]  Slow 通路特征（稳定 + 已巩固残差）
        f_fast : [B, C, H, W]  Fast 通路特征（含最新 Novel 残差）
        mask_slow: [B, H, W] bool（True=padding）或 None

        Returns:
            f_final : [B, C, H, W]  融合后特征，送入 O1O 的 input_proj / Detection Head
            info    : dict 诊断量，供 SCM 辅助 loss 与可视化
        """
        B, C, H, W = f_slow.shape

        # --- (1) 置信度 / Novel 打分 ---
        resid = (f_fast - f_slow).abs()                       # [B, C, H, W]  Fast/Slow 分歧
        M = self.scorer(torch.cat([f_fast, resid], dim=1))    # [B, 1, H, W]  空间 Novel 置信度

        # --- (2) 筛选高置信 Novel 的 Fast 特征 ---
        f_fast_sel = M * f_fast                              # [B, C, H, W]  仅高置信处保留 Fast

        # --- (3) Cross-Attention 巩固 (Slow 坐标系写回 Novel 知识) ---
        if self.use_cross_attn:
            qs = self.norm_s(f_slow.flatten(2).transpose(1, 2))  # [B, N=H*W, C]
            kf = self.norm_f(f_fast_sel.flatten(2).transpose(1, 2))  # [B, N, C]
            # 构造 key padding mask（True=忽略）。query 与 key 来自同一图像、同一 padding 分布，
            # 故仅用 key_padding_mask 即可（query 的 padded 位置输出本就被丢弃）。
            # 注: 不使用 query_key_padding_mask（部分 torch 版本无该参数）。
            if mask_slow is not None:
                pad = mask_slow.flatten(1)                        # [B, N] bool
                f_ca, _ = self.cross_attn(qs, kf, kf, key_padding_mask=pad)
            else:
                f_ca, _ = self.cross_attn(qs, kf, kf)
            f_ca = f_ca.transpose(1, 2).view(B, C, H, W)         # [B, C, H, W]
        else:
            # scm_loss_w==0 时 consolidation_loss 不进 loss，f_ca 无任何消费者。
            # 跳过整个 cross-attn 前向 -> 省下 [B, heads, N, N] 注意力矩阵显存
            # （高分辨率 C3 上 N≈H*W 可达 ~1.6e4，矩阵约 8 GiB/B，是 OOM 元凶）。
            f_ca = None

        # --- (4) 特征融合 ---
        if self.share_base:
            # 共享冻结基: 只把 Fast 相对 Slow 的 Novel 偏差按 M 加回去（避免重复叠加稳定部分）
            f_delta = M * (f_fast - f_slow)                   # [B, C, H, W]
            f_final = f_slow + self.alpha * f_delta
        else:
            # 独立基: 直接采用论文写法 F_final = F_slow + alpha * (M * F_fast)
            f_final = f_slow + self.alpha * f_fast_sel

        info = {
            "M": M,                # [B,1,H,W]         Novel 置信度图
            "f_fast_sel": f_fast_sel,  # [B,C,H,W]      筛选后的 Fast 特征
            "f_ca": f_ca,          # [B,C,H,W]         Cross-Attention 巩固特征
            "f_delta": (f_fast - f_slow) if self.share_base else f_fast_sel,
            "alpha": self.alpha,   # 标量               融合权重
        }
        return f_final, info


# ---------------------------------------------------------------------------
# 3) 辅助损失（可选，按论文需要取舍；均不引入新数据/新 head）
# ---------------------------------------------------------------------------
def slow_distillation_loss(slow_feats: list, teacher_feats: list) -> torch.Tensor:
    """
    Slow 稳定性锚定: 用冻结的 Task-1 基（teacher）约束 Slow 通路特征，防旧类遗忘。
    slow_feats / teacher_feats: List[NestedTensor]，长度一致。
    """
    loss, n = 0.0, 0
    for sf, tf in zip(slow_feats, teacher_feats):
        s, _ = sf.decompose()      # [B, C, H, W]
        t, _ = tf.decompose()
        loss = loss + F.mse_loss(s, t.detach())
        n += 1
    return loss / max(n, 1)


def consolidation_loss(f_ca: torch.Tensor, f_fast_sel: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """
    SCM 巩固一致性: 在高置信 Novel 区域，要求 Cross-Attention 输出逼近（被 stop-grad 的）筛选 Fast 特征。
    f_ca, f_fast_sel: [B, C, H, W];  M: [B, 1, H, W]
    """
    diff = (f_ca - f_fast_sel.detach()) ** 2          # [B, C, H, W]
    return (M * diff).mean()


def novel_gate_budget_loss(M: torch.Tensor, budget: float = 0.30) -> torch.Tensor:
    """
    防止 SCM 过度开放: 限制平均门控比例不超过 budget（鼓励稀疏、选择性）。
    """
    return F.relu(M.mean() - budget)


# ---------------------------------------------------------------------------
# 4) 权重级渐进巩固: EMA 把 Fast adapter 移入 Slow adapter
#    Slow adapter 不接收梯度，只由 EMA 缓慢吸收 Fast 的 Novel 残差 -> 高稳定性 + 渐进巩固。
#    在训练循环每个 step（或每 K step）调用一次。
# ---------------------------------------------------------------------------
@torch.no_grad()
def ema_consolidate(slow_path: FastCNNBackbone, fast_path: FastCNNBackbone,
                    rho: float = 0.01, confidence_gate: float = None) -> None:
    """
    Slow adapter 参数 <- (1-rho)*Slow + rho*Fast
    rho 小 -> Slow 是 Fast 的慢速移动平均（稳定但持续巩固）。
    confidence_gate: 若提供，则用 (rho * confidence_gate) 调制，仅在平均 Novel 置信度高时加速巩固。
    """
    rho_eff = rho * confidence_gate if confidence_gate is not None else rho
    for sa, fa in zip(slow_path.adapters, fast_path.adapters):
        for psa, pfa in zip(sa.parameters(), fa.parameters()):
            psa.mul_(1.0 - rho_eff).add_(rho_eff * pfa.detach())


# ---------------------------------------------------------------------------
# 5) 构造辅助: 从 Slow 基复制权重初始化 Fast 基 + Fast/Slow adapter（Task-2 起点一致）
#    在载入 Task-1 checkpoint 之后调用一次。
# ---------------------------------------------------------------------------
@torch.no_grad()
def init_fast_from_slow(slow_cnn: nn.Module, fast_cnn: nn.Module,
                        slow_path: FastCNNBackbone, fast_path: FastCNNBackbone) -> None:
    # 复制冻结基权重（两者起点完全相同 = Task-1 / ImageNet 权重）
    fast_cnn.load_state_dict(slow_cnn.state_dict())
    # adapter 均为 init-0，这里显式对齐确保起点零残差（baseline parity）
    fast_path.adapters.load_state_dict(slow_path.adapters.state_dict())
    fast_path.scale.copy_(slow_path.scale)


# ---------------------------------------------------------------------------
# 本地冒烟测试入口（脱离 O1O 仓库也能验证形状 / 梯度 / EMA 语义）
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    # 假 CNN: 输出 3 层固定形状特征（模拟 ResNet-50 C3/C4/C5）
    class FakeCNN(nn.Module):
        def __init__(self, levels=(512, 1024, 2048)):
            super().__init__()
            self.levels = levels
            self.conv = nn.ModuleList([nn.Conv2d(3, c, 3, padding=1) for c in levels])
        def forward(self, samples):
            x = samples.tensors if hasattr(samples, "tensors") else samples   # [B,3,H,W]
            out = []
            for i, c in enumerate(self.levels):
                h = self.conv[i](x)                    # [B, c, H, W]（仅测形状，不真实级联）
                out.append(NestedTensor(h, None))       # 模拟 backbone 输出 List[NestedTensor]
            return out

    slow_cnn = FakeCNN()
    fast_cnn = FakeCNN()
    slow_path = FastCNNBackbone(slow_cnn, freeze_base=True, init_zero=True)
    fast_path = FastCNNBackbone(fast_cnn, freeze_base=True, init_zero=True)
    # 真实集成中 slow adapter 是 EMA 目标 -> 不收梯度；这里显式模拟该设定。
    for p in slow_path.adapters.parameters():
        p.requires_grad_(False)
    scm = nn.ModuleList([SelectiveConsolidationModule(C) for C in (512, 1024, 2048)])

    # 构造假 NestedTensor 输入 [2,3,64,64]
    inp = NestedTensor(torch.randn(2, 3, 64, 64), None)

    # 注: scale 初值为 0（init-zero 残差 = baseline parity），首步 adapter 梯度天然为 0；
    # 这里把 fast scale 置 0.5 以验证"当 scale>0 时梯度确实流向 fast adapter"这条路径存在。
    fast_path.scale.data.fill_(0.5)

    slow_feats = slow_path(inp)   # List[NestedTensor]
    fast_feats = fast_path(inp)
    fused, infos = [], []
    for l in range(len(slow_feats)):
        fs, _ = slow_feats[l].decompose()
        ff, _ = fast_feats[l].decompose()
        f_final, info = scm[l](fs, ff)
        fused.append(NestedTensor(f_final, None))
        infos.append(info)

    # 断言 1: 融合特征形状与原 backbone 输出一致 [B,C,H,W]
    for l, f in enumerate(fused):
        t, _ = f.decompose()
        assert t.shape == (2, (512, 1024, 2048)[l], 64, 64), t.shape
    print("[OK] fused feature shapes match O1O backbone output:", [f.decompose()[0].shape for f in fused])

    # 断言 2: fast adapter 有梯度、slow adapter 无梯度（可塑性只在 Fast）
    fast_loss = sum(inf["f_delta"].sum() for inf in infos)
    fast_loss.backward()
    fa_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in fast_path.adapters.parameters())
    sa_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in slow_path.adapters.parameters())
    print(f"[OK] fast adapter grad={fa_has_grad}, slow adapter grad={sa_has_grad}")
    assert fa_has_grad and not sa_has_grad

    # 断言 3: EMA consolidate 把 fast adapter 权重移入 slow adapter
    before = [p.detach().clone() for p in slow_path.adapters[0].parameters()]
    ema_consolidate(slow_path, fast_path, rho=0.5)
    after = list(slow_path.adapters[0].parameters())
    moved = any(not torch.allclose(b, a) for b, a in zip(before, after))
    print(f"[OK] ema_consolidate moved slow adapter weights: {moved}")
    assert moved
    print("\nALL SMOKE TESTS PASSED")
