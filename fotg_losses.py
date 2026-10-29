#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FOTG 核心损失模块：功能纤维 (F) + 动态重叠预算 (O)。

只依赖 torch，不依赖模型具体结构，可独立单测。

模块 F — 功能纤维约束：
    L_fib = E_{z in A_old} || J_old(z) · (z_theta - z_ref) ||_2^2
    J_old(z) = d g_old(z) / dz   (旧输出对特征的雅可比)
    g_old  = 旧类分类读出（含 superclass 门控），定位读出可选。

模块 O — 动态重叠预算：
    旧-旧：L_old = sum w_ij * rho(O_ij^(t) - O_ij^(t-1))
    新-旧：L_new = sum rho(O_ij - O_ij*)
    O_ij* = clip(O_0 + alpha*S_ij - beta*C_ij, O_min, O_max)
    O_ij  = (1/k) ||U_i^T U_j||_F^2   (子空间重叠，主角度余弦平方均值)

设计原则：默认权重 0 时不改变原训练行为，仅在显式启用时激活。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================================
# 模块 F：功能纤维约束
# ============================================================

def readout_jacobian(z, Wc, bc, Ws, bs, prev, cur, sup_idx_of_cls):
    """
    计算旧类读出 g_old(z) 对特征 z 的雅可比（向量化的解析式，不依赖 autograd）。

    真实读出（复现 PostProcess，先分别 softmax 再逐元素乘）：
        s_c = p_c * g_sigma(c)          # p = softmax(class_logits 的 known 部分)
                                        # g = softmax(superclass_logits 的 known 部分)
        g_old = sum_{c in old} s_c      # 旧类总分（或取每个旧类的 s_c 向量）

    敏感方向（已从代码推导，无统计假设）：
        h_c = s_c * [(w_c - w_bar) + (v_sigma(c) - v_bar)]
        其中 w_bar = sum_c p_c w_c, v_bar = sum_j g_j v_j

    返回：
        J : (N, C_old, D)  每个旧类的敏感方向（每类一个 D 维雅可比行向量）
        具体：J[n, c] = h_c[n]   （对 c in [0, prev)）
        g_old 是"每个旧类各自的分类分数"，不是它们的总和——
        这样才能保护每个旧类的分类边界，而非仅保护旧类总分。
    """
    num_classes = Wc.shape[0]           # 81
    N = z.shape[0]
    device = z.device

    # class logits (屏蔽 invalid)
    cls_logits = z @ Wc.t() + bc.unsqueeze(0)          # (N, 81)
    invalid = list(range(prev + cur, num_classes - 1))
    if invalid:
        cls_logits[:, invalid] = -1e10
    p = torch.softmax(cls_logits[:, :-1], dim=-1)       # (N, 80)

    # superclass logits
    sup_logits = z @ Ws.t() + bs.unsqueeze(0)           # (N, 13)
    g = torch.softmax(sup_logits[:, :-1], dim=-1)       # (N, 12)

    # 门控：s_c = p_c * g_sigma(c)
    g_of_cls = g[:, sup_idx_of_cls]                     # (N, 80)
    s = p * g_of_cls                                    # (N, 80)

    # 加权平均权重
    w_bar = p @ Wc[:-1]                                 # (N, D)
    v_bar = g @ Ws[:-1]                                 # (N, D)

    Wc_known = Wc[:80]
    Ws_of_cls = Ws[sup_idx_of_cls]                      # (80, D)

    # h_c = s_c * [(w_c - w_bar) + (v_sigma(c) - v_bar)]
    direction = (Wc_known.unsqueeze(0) - w_bar.unsqueeze(1)) + \
                (Ws_of_cls.unsqueeze(0) - v_bar.unsqueeze(1))   # (N, 80, D)
    h = s.unsqueeze(-1) * direction                      # (N, 80, D)

    # 每个旧类的敏感方向：保留 c in [0, prev)，不求和
    J = h[:, :prev, :]                                   # (N, prev, D)
    return J


def functional_fibre_loss(z_theta, z_ref, J_old, prev, reduction="mean"):
    """
    功能纤维约束（逐类保护分类边界，而非仅保护总分）：
        L_fib = E_z [ Σ_{c in old} || J_c(z_ref) · (z_theta - z_ref) ||_2^2 ]

    参数：
        z_theta : (N, D) 当前模型 RoI 特征
        z_ref   : (N, D) 旧模型同输入 RoI 特征
        J_old   : (N, C_old, D) 每个旧类的雅可比敏感方向（readout_jacobian 输出）
        prev    : 旧类数 C_old

    关键：J_old 是逐类的 (N, C_old, D)，每个旧类各自算输出变化，
    再对所有旧类求和。这保护了每个旧类的分类边界（防止类别间混淆被总分抵消掩盖）。

    注意：z_ref 与 J_old 应 detach（旧模型冻结，不反传）。
    """
    dz = z_theta - z_ref.detach()          # (N, D)
    J = J_old.detach()                     # (N, C_old, D)
    # 每个旧类的一阶输出变化：delta[n, c] = J_c[n] · dz[n]
    delta = torch.einsum("ncd,nd->nc", J, dz)   # (N, C_old)
    # 逐类平方和（保护每个旧类边界），再对样本求和/平均
    loss = (delta ** 2).sum(dim=1)         # (N,)
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


def nullspace_projection(J_old, lam=1e-3, alpha=1.0):
    """
    近似零空间投影（F-B 软投影，已接入 engine.py backward hook 主路径；alpha 控软硬）：
        P_fib = I - J^T (J J^T + lam I)^{-1} J
    其中 J 是完整雅可比矩阵 (C_old, D)（每个旧类一行），
    与用户公式一致：JJ^T 是 (C_old, C_old) 矩阵。

    返回 (D, D) 投影矩阵（针对单个样本的 J）。
    J_old : (C_old, D) 或 (N, C_old, D)。若 3D 则对每个样本返回 (N, D, D)。
    """
    J = J_old.detach()
    if J.dim() == 2:
        C, D = J.shape
        JJt = J @ J.t()                      # (C, C)
        reg = JJt + lam * torch.eye(C, device=J.device)
        # P = I - J^T (JJ^T + lam I)^{-1} J
        P = torch.eye(D, device=J.device) - alpha * J.t() @ torch.linalg.solve(reg, J)
        return P
    elif J.dim() == 3:
        # (N, C, D)：逐样本
        N, C, D = J.shape
        P = torch.empty(N, D, D, device=J.device)
        for n in range(N):
            P[n] = nullspace_projection(J[n], lam=lam, alpha=alpha)
        return P
    else:
        raise ValueError(f"J_old 维度应为 2 或 3，得到 {J.dim()}")


# ============================================================
# 模块 F v2：修订读出（门控前 logits + 边界框）+ query 加权
# ------------------------------------------------------------
# v1 的问题：J_c ∝ s_c = p_c * g_sigma(c)，而 100 个 query 里绝大多数是背景，
# s_c 极小 => L_fib 被压到 1e-9 量级，约束形同虚设。
# v2 的修正（三步方案）：
#   1) query 加权 w_q：只在承载旧类知识的 query 上求平均，背景 query 不稀释。
#   2) 主读出改为「门控前 logits」+「边界框输出」：
#      logits = W z + b 对 z 是线性的，雅可比恒等于 W_old，根本不存在门控压低。
#   3) 只在有效 query 上归一化，并按输出维度（C_old / 4）分别求平均。
# ============================================================

def gated_scores(z, Wc, bc, Ws, bs, prev, cur, sup_idx_of_cls):
    """复现 PostProcess 的门控分数 s_c = p_c * g_sigma(c)。

    仅用于构造 query 权重 / 诊断（不用于主约束），返回 s : (N, num_known)。
    """
    num_classes = Wc.shape[0]
    cls_logits = z @ Wc.t() + bc.unsqueeze(0)              # (N, C+1)
    invalid = list(range(prev + cur, num_classes - 1))
    if invalid:
        cls_logits[:, invalid] = -1e10
    p = torch.softmax(cls_logits[:, :-1], dim=-1)          # (N, num_known)
    sup_logits = z @ Ws.t() + bs.unsqueeze(0)              # (N, 13)
    g = torch.softmax(sup_logits[:, :-1], dim=-1)          # (N, 12)
    return p * g[:, sup_idx_of_cls]                        # (N, num_known)


def class_logits_jacobian(Wc, prev):
    """旧类 logits 对 z 的雅可比。

    logits = W z + b 对 z 线性 => J = W_old，**精确成立且与 z 无关**。
    这正是 v2 选择 logits 作主读出的原因：不存在被 s_c 压低的可能。

    Wc : (num_classes_total, D)，取前 prev 行。返回 (prev, D)。
    """
    return Wc[:prev]


def bbox_jacobian(z, bbox_head):
    """旧模型边界框输出对 z 的雅可比，返回 (N, 4, D)。

    bbox_head 对每个 query 独立作用（无跨 query 交互），因此对 b[:, j].sum()
    求 z 的梯度，得到的 (N, D) 第 q 行正是 db_qj / dz_q —— 4 次反传拿全雅可比。
    """
    z = z.detach().requires_grad_(True)
    b = bbox_head(z)                                       # (N, 4)
    rows = []
    for j in range(b.shape[-1]):
        (g,) = torch.autograd.grad(b[:, j].sum(), z, retain_graph=True,
                                   allow_unused=True)
        rows.append(g if g is not None else torch.zeros_like(z))
    return torch.stack(rows, dim=1)                        # (N, 4, D)


def query_weight_old_class(s, prev, threshold=0.0, topk=10, num_queries=None,
                           soft=True):
    """构造旧类 query 权重 w_q（三步方案第一步）。

    规则：
      - 对 s[:, :prev] 取最大值作为「旧类置信度」，天然要求 argmax 落在旧类；
      - soft=True：直接用置信度当权重（背景 query 自动接近 0，无需硬阈值）；
        soft=False：0/1 掩码，此时 threshold 才有意义；
      - 每张图最多保留 topk 个候选，避免个别图刷屏。

    【v2 修正】默认 threshold 从 0.3 改为 0.0（即不设绝对阈值），原因是实测：
      s = p_cls · g_sup 是两个 softmax 相乘，量级天然被压到 ~1e-2
      （80 类 softmax 的最大值常在 0.1~0.5，13 类 superclass softmax 常在 0.1~0.9，
       乘积的期望量级远低于任一因子）。标定脚本在 30 个真实 batch、6000 个 query 上
      阈值 0.3 命中数为 0，L_fib 恒等于 0——约束形同虚设。
      正确做法是【按每图 top-k 相对排序】而不是绝对阈值：DETR 里
      绝大多数 query 是背景，绝对阈值取决于类别数这一与语义无关的因素，
      而"本图中最像旧类的 k 个 query"才是稳定、跨数据集可迁移的定义。
      量级差异交给 soft 权重和后续分母归一化（Σ_q w_q）处理。

      若调用方确实想过滤背景，请给一个按实测分位数定出的 threshold，不要拍 0.3。

    s : (N, num_known)，N = B*Q；num_queries = Q（按图分组做 topk 用）。
    返回 (N,) 权重。
    """
    s_old = s[:, :prev]
    conf, _ = s_old.max(dim=-1)                            # (N,)
    if soft:
        w = torch.where(conf >= threshold, conf.clamp(min=0.0),
                        torch.zeros_like(conf))
    else:
        w = (conf >= threshold).to(conf.dtype)
    if topk is not None and topk > 0 and num_queries is not None:
        B = s.shape[0] // num_queries
        w2 = w.view(B, num_queries)
        conf2 = conf.view(B, num_queries)
        k = min(topk, num_queries)
        _, idx = conf2.topk(k, dim=1)                      # (B, k)
        mask = torch.zeros_like(w2)
        mask.scatter_(1, idx, 1.0)
        w = (w2 * mask).reshape(-1)
    return w


def functional_fibre_loss_v2(z_theta, z_ref, W_old, J_box, w_q=None, eta=1.0,
                             eps=1e-6, return_parts=False):
    """修订版功能纤维约束：

        L_fib = [ Σ_q w_q ( ‖d_q^cls‖² / C_old + η ‖d_q^box‖² / 4 ) ] / (Σ_q w_q + ε)

        d_q^cls = W_old · Δz_q    （门控前 logits，雅可比 = W_old）
        d_q^box = J_box_q · Δz_q  （旧模型边界框输出）

    参数：
        z_theta : (N, D) 当前模型特征（保留梯度）
        z_ref   : (N, D) 旧模型同输入特征（detach）
        W_old   : (C_old, D) 旧类 logits 权重
        J_box   : (N, 4, D) 旧模型 bbox 雅可比
        w_q     : (N,) query 权重（None 则全 1）
    """
    dz = z_theta - z_ref.detach()                          # (N, D)
    W_old = W_old.detach()
    J_box = J_box.detach()
    C_old = W_old.shape[0]

    d_cls = dz @ W_old.t()                                 # (N, C_old)
    d_box = torch.einsum("nod,nd->no", J_box, dz)          # (N, 4)

    cls_term = (d_cls ** 2).sum(-1) / max(C_old, 1)        # (N,)
    box_term = (d_box ** 2).sum(-1) / 4.0                  # (N,)
    per_q = cls_term + eta * box_term                      # (N,)

    if w_q is None:
        w_q = torch.ones_like(per_q)
    w = w_q.detach().to(per_q.dtype)
    denom = w.sum() + eps
    loss = (w * per_q).sum() / denom

    if return_parts:
        return loss, {
            "cls_term": float(((w * cls_term).sum() / denom).item()),
            "box_term": float((eta * (w * box_term).sum() / denom).item()),
            # w.sum() 是"权重质量"（soft 时等于置信度之和），n_pos 才是真正参与约束
            # 的 query 个数——诊断 n_valid=0 这类问题时必须看 n_pos，不能只看 w.sum()。
            "w_sum": float(w.sum().item()),
            "n_pos": int((w > 0).sum().item()),
            "n_valid": int((w > 0).sum().item()),
            "w_max": float(w.max().item()) if w.numel() else 0.0,
            "w_mean_pos": float(w[w > 0].mean().item()) if int((w > 0).sum()) else 0.0,
            "per_query": per_q.detach(),
        }
    return loss


def _route_gate(cls_logits, sup_logits, sup_idx_of_cls, mode="blend"):
    """统一 Route gate：给定 class/superclass logits，返回 (N, 80) 路由分数。

    本函数是 Route 模块内**唯一**构造 gate 分数的入口，确保 Route 损失内部
    和 old_scores()（query 权重来源）使用**同一公式**。

    mode:
      "blend"    s = softmax(cls) × (0.5 + 0.5 × softmax(sup))
                 —— 与 old_scores() 默认 mode 一致。softmax 竞争 + blend 调制，
                   superclass 退化为 [0.5, 1] 区间权重而非乘法衰减因子。
                   优点：softmax 让不同类互相竞争，topk query 选择更稳定；
                         blend 让 superclass 不把概率完全压没，保留分类分数信息。
      "multiply" s = sigmoid(cls) × sigmoid(sup)
                 —— 原始 train 约定。两个独立概率相乘，量级偏小（~0.07-0.56）。
    """
    dev = cls_logits.device
    sup_idx = sup_idx_of_cls.to(dev)
    if mode == "blend":
        p = torch.softmax(cls_logits[..., :-1].float(), dim=-1)
        g = torch.softmax(sup_logits[..., :-1].float(), dim=-1)
        return p * (0.5 + 0.5 * g[..., sup_idx])
    # multiply
    p = torch.sigmoid(cls_logits[..., :-1])
    g = torch.sigmoid(sup_logits[..., :-1])
    return p * g[..., sup_idx]


def route_aware_preservation(out_cur, out_old, sup_idx_of_cls, prev, w_q=None,
                             gate_mode="blend", eps=1e-6):
    """路由感知功能保持损失：约束当前 decoder 对旧类的最终路由分数
    相对冻结旧模型的漂移。

    gate_mode 默认 "blend"，与 FOTGModule.w_mode（old_scores() 默认）一致。
    统一后 Route 损失内部的 gate 与 old_scores()（w_q 来源）使用**完全相同的公式**，
    三处统一：query 选谁（w_q）、Route 损失中 s_cur、Route 损失中 s_ref。

    out_cur : 当前模型 outputs（含 pred_logits / pred_super_logits，保留梯度）
    out_old : 冻结旧模型 outputs（detach，作参考目标）
    sup_idx_of_cls : (80,) 每类 superclass 索引（long）
    prev : 旧类数
    gate_mode : "blend"（统一 softmax×blend，默认）| "multiply"（原始 sigmoid×sigmoid）
    eps : 聚合分母的数值稳定项
    返回标量损失（对旧类 query 加权平均）。
    """
    dev = out_cur["pred_logits"].device
    lc = out_cur["pred_logits"].reshape(-1, out_cur["pred_logits"].shape[-1])
    sc = out_cur["pred_super_logits"].reshape(-1, out_cur["pred_super_logits"].shape[-1])
    s_cur = _route_gate(lc, sc, sup_idx_of_cls, mode=gate_mode)    # (N, 80)

    lo = out_old["pred_logits"].detach().reshape(-1, out_old["pred_logits"].shape[-1])
    so = out_old["pred_super_logits"].detach().reshape(-1, out_old["pred_super_logits"].shape[-1])
    s_ref = _route_gate(lo, so, sup_idx_of_cls, mode=gate_mode)    # (N, 80)

    d = (s_cur[..., :prev] - s_ref[..., :prev]) ** 2                 # (N, prev)
    per_q = d.sum(dim=-1)                                           # (N,)
    if w_q is None:
        w_q = torch.ones_like(per_q)
    w = w_q.detach().to(per_q.dtype)
    denom = w.sum() + eps
    return (w * per_q).sum() / denom


# ============================================================
# 模块 O：动态重叠预算
# ============================================================

def subspace_overlap(U_i, U_j):
    """
    O_ij = (1/k) ||U_i^T U_j||_F^2
    U_i, U_j: (D, k) 正交基（列正交）。返回标量。
    """
    cross = U_i.t() @ U_j                  # (k, k)
    k = U_j.shape[1]
    return (cross ** 2).sum() / max(k, 1)


def overlap_old_loss(O_cur, O_prev, w=None, rho="l2"):
    """
    旧-旧：保留已有几何关系。
        L_old = sum_{i,j in C_old} w_ij * rho(O_ij^(t) - O_ij^(t-1))
    参数：
        O_cur, O_prev : (K_old, K_old) 重叠矩阵（当前 vs 上一阶段）
        w : (K_old, K_old) 可选权重
        rho : "l2" | "l1" | "huber"
    """
    diff = O_cur - O_prev
    if rho == "l1":
        r = diff.abs()
    elif rho == "huber":
        r = F.huber_loss(diff, torch.zeros_like(diff), reduction="none", delta=0.1)
    else:
        r = diff ** 2
    if w is not None:
        r = r * w
    return r.sum()


def calibrate_overlap_prior(O_old, O_min_floor=0.0):
    """从【实测的旧-旧重叠矩阵】反推 O_0 / O_min / O_max，而不是拍脑袋给 0.5/0.2/0.8。

    实测（M-OWODB task3 checkpoint，class 权重方向，D=256）：
        旧-旧重叠 均值 0.121、最大值 0.406
        新-旧重叠 均值 0.008、最大值 0.068
    也就是说"类别权重方向"之间的余弦平方天然就在 0.01 量级，和 O_0=0.5 差了 40 倍。
    拿 0.5 当目标会让 100% 的新-旧对都判为"违背"（实测 violation_frac=1.0），
    约束退化成一个与几何无关的常数惩罚，等于没有约束。
    正确做法：让目标落在【本模型真实几何】的尺度上——
        O_0   = 旧-旧重叠的中位数（"正常共存"的基准）
        O_max = 旧-旧重叠的分位数上限（"过分纠缠"的界）
        O_min = 旧-旧重叠的下分位（允许新类几乎正交）
    """
    O = torch.as_tensor(O_old, dtype=torch.float32)
    n = O.shape[0]
    if n < 2:
        return {"O_0": 0.5, "O_min": 0.2, "O_max": 0.8}
    off = O[~torch.eye(n, dtype=torch.bool)]
    if off.numel() == 0:
        return {"O_0": 0.5, "O_min": 0.2, "O_max": 0.8}
    o0 = float(torch.median(off))
    q_hi = float(torch.quantile(off, 0.90))
    q_lo = float(torch.quantile(off, 0.10))
    return {"O_0": max(o0, 1e-6),
            "O_min": max(min(q_lo, o0), O_min_floor),
            "O_max": max(q_hi, 2.0 * o0, 1e-4)}


def overlap_target(S_ij, C_ij, O_0=0.5, alpha=0.1, beta=0.1, O_min=0.2, O_max=0.8):
    """
    新-旧：动态重叠目标。
        O_ij* = clip(O_0 + alpha*S_ij - beta*C_ij, O_min, O_max)
    S_ij : 可迁移共享视觉因素 (0~1 或任意尺度)
    C_ij : 功能冲突 (0~1 或任意尺度)
    """
    raw = O_0 + alpha * S_ij - beta * C_ij
    return torch.clamp(raw, O_min, O_max)


def overlap_new_loss(O_cur, O_target, rho="l2"):
    """
    新-旧：让实际重叠逼近动态目标。
        L_new = sum_{i in new, j in old} rho(O_ij - O_ij*)
    O_cur, O_target : (K_new, K_old) 实际重叠 vs 目标重叠
    """
    diff = O_cur - O_target
    if rho == "l1":
        return diff.abs().sum()
    elif rho == "huber":
        return F.huber_loss(diff, torch.zeros_like(diff), reduction="sum", delta=0.1)
    return (diff ** 2).sum()


def transfer_value(sup_idx_new, sup_idx_old, feat_sim=None):
    """
    迁移价值 S_ij 代理：
        - superclass 共享度：同 superclass = 1，否则 0（或可用语义层次）。
        - 可选叠加特征质心相似度 feat_sim。
    返回 (K_new, K_old) 矩阵。
    """
    K_new, K_old = len(sup_idx_new), len(sup_idx_old)
    S = torch.zeros(K_new, K_old)
    for i in range(K_new):
        for j in range(K_old):
            S[i, j] = 1.0 if sup_idx_new[i] == sup_idx_old[j] else 0.0
    if feat_sim is not None:
        S = S + feat_sim
    return S


def conflict_value(J_new_proj_on_old):
    """
    功能冲突 C_ij 代理：
        新类读出雅可比在旧类敏感方向上的投影幅度（越大越冲突）。
    输入 J_new_proj_on_old : (K_new, K_old) 已投影的冲突矩阵。
    """
    return J_new_proj_on_old


# ============================================================
# 模块 T：拓扑维护（持久同调签名 + Wasserstein 距离）
# ------------------------------------------------------------
# 关键约束（你的第四节）：
#   - 不把 betti1_max 直接当惩罚项（受采样数/过滤尺度影响太大）
#   - 保护的是「同一个旧类别」的持久图在阶段间的变化
#   - 必须在【同一批、固定数量、类别平衡】的旧类锚点上比较，否则拓扑变化
#     可能只是样本数变化的结果
#
# 实现说明：gudhi/ripser 在本机装不上（根分区满 + scipy 被污染），
# 这里用纯 numpy 实现：
#   PD_0：VR 过滤下 0 维持久图 = 最小生成树边权（birth 全为 0）—— 这是精确的。
#   PD_1：正确的 VR H_1 —— 边在「同分量内闭边」时诞生 1-环（birth=边权），
#         三角形（2-单纯形）出现时把所在分量中 birth 最小的存活环配对消亡
#         （death=三角形三边最大值）。之前版本误把「分量合并」当环消亡（无三角形填充），
#         导致 PD_1 堆满长命环、几乎全是噪声，已在 rips_pd1 中修正。
#         纯 numpy 实现；如需更快/更精确可换 ripser（本环境已可安装）。
# ============================================================

def _pairwise_dist(X):
    """X: (n, d) -> (n, n) 欧氏距离矩阵（numpy）。"""
    X = X.detach().cpu().numpy() if hasattr(X, "detach") else X
    X = np.asarray(X, dtype=np.float64)
    sq = (X * X).sum(1)
    D2 = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    np.fill_diagonal(D2, 0.0)
    return np.sqrt(np.maximum(D2, 0.0))


def _mst_edge_weights(D):
    """Prim 最小生成树，返回 MST 的 n-1 条边权（升序）。"""
    n = D.shape[0]
    if n <= 1:
        return np.zeros(0)
    in_tree = np.zeros(n, dtype=bool)
    best = np.full(n, np.inf)
    best[0] = 0.0
    weights = []
    for _ in range(n - 1):
        # 选未入树中 best 最小的点
        cand = np.where(~in_tree, best, np.inf)
        u = int(np.argmin(cand))
        if not np.isfinite(cand[u]):
            break
        weights.append(float(best[u]))
        in_tree[u] = True
        best = np.minimum(best, D[u])
        best[in_tree] = 0.0
    return np.array(sorted(weights))


def rips_pd0(X):
    """0 维持久图 PD_0：birth 全 0，death = MST 边权。返回 (n-1, 2) float64。"""
    D = _pairwise_dist(X)
    w = _mst_edge_weights(D)
    if w.size == 0:
        return np.zeros((0, 2))
    return np.column_stack([np.zeros_like(w), w])


def rips_pd1(X, max_scale=None):
    """1 维持久图 PD_1（H1 区间），以 ripser 为权威实现。

    拓扑语义（第二优先级核验的结论）：
        对每类特征点云构造 VR 复形，H1 区间 [birth, death] 表示：
          - birth  = 闭合该 1-环（loop）的第一条边的入滤尺度；
          - death  = 填充该环的【二维单形（三角形）】的入滤尺度
                    （三角形权 = 其三边长的最大值）。
        即 ripser 的标准 VR 复形约简中，H1 消亡尺度天然来自三角形填充，
        不存在"分量合并"式近似。零模型对照实验（probe_cache/verify_pd1.json）
        确认真实类点云的 H1 持续性显著高于打乱/高斯零模型（t1/t2）。

    为什么不用自写边界矩阵约简：
        自写的 VR 2-骨架 mod-2 列约简会把大量三角形误判为死亡配对，
        对随机 9 点云也会吐出 56 个伪 H1 环（而 ripser 给出 H1=0），
        且 H0 顶点-边配对语义整体错位；手工修补配对规则无法对齐 ripser。
        故直接委托 ripser（权威 C++ 实现，maxdim=1），保证 death 尺度语义正确。

    注意（部署）：本函数依赖 ripser 包（服务器 watcher 进程已剔除 ~/.local，
        其中曾有残废 scipy；ripsser 仅在 ~/.local，故 T 模块在 watcher 同进程
        运行时不可用——F-only 训练不调用本函数，不受影响）。
        若需无 ripser 的可微近似（T 模块拓扑保持 loss），应另行实现基于
        H1 persistence 的可微代理，而非依赖本精确但非可微的实现。

    返回 (m, 2) 的 (birth, death)；persistence = death − birth。无环时返回 (0,2) 空。
    """
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    if n < 3:
        return np.zeros((0, 2))
    from scipy.spatial.distance import pdist, squareform
    D = squareform(pdist(X))
    max_scale = float(D.max()) if max_scale is None else float(max_scale)
    try:
        from ripser import ripser as _ripser
    except Exception as e:  # pragma: no cover - 部署环境必须安装 ripser
        raise RuntimeError(
            "rips_pd1 需要可用的 ripser 包。H1 的 death 尺度语义依赖 ripser 的标准"
            " VR 复形约简，自写 2-骨架约简会产生伪环（已实测：随机点云上多算 56 个）。"
        ) from e
    dgm = _ripser(D, distance_matrix=True, maxdim=1, thresh=max_scale)["dgms"][1]
    dgm = dgm[np.isfinite(dgm[:, 1])]
    if dgm.size == 0:
        return np.zeros((0, 2))
    death = np.minimum(dgm[:, 1], max_scale)
    P = np.column_stack([dgm[:, 0], death])
    out = P[(P[:, 1] - P[:, 0]) > 0]
    return out if out.size else np.zeros((0, 2))


def _diag_dist(p):
    """点到对角线 (b=d) 的距离 = |d-b|/sqrt(2)。p: (...,2)"""
    return np.abs(p[..., 1] - p[..., 0]) / np.sqrt(2.0)


def wasserstein_pd(P, Q):
    """两个持久图之间的 1-Wasserstein 距离（带对角线投影）。
    P, Q: (n,2) / (m,2) numpy。未匹配的点投影到对角线计入代价。
    """
    P = np.asarray(P, dtype=np.float64).reshape(-1, 2)
    Q = np.asarray(Q, dtype=np.float64).reshape(-1, 2)
    if P.size == 0 and Q.size == 0:
        return 0.0
    if P.size == 0:
        return float(_diag_dist(Q).sum())
    if Q.size == 0:
        return float(_diag_dist(P).sum())
    # 代价矩阵：匹配代价 + 到对角线的代价
    C = np.linalg.norm(P[:, None, :] - Q[None, :, :], axis=-1)   # (n, m)
    dp = _diag_dist(P)[:, None]
    dq = _diag_dist(Q)[None, :]
    # 补齐成方阵：多出的行列表示「投影到对角线」
    n, m = C.shape
    N = n + m
    big = np.zeros((N, N))
    big[:n, :m] = C
    big[:n, m:] = dp
    big[n:, :m] = dq
    try:
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(big)
        return float(big[r, c].sum())
    except Exception:
        # 无 scipy 时退化为贪心（上界）
        idx = np.argsort(big, axis=1)[:, 0]
        return float(sum(big[i, idx[i]] for i in range(N)))


def filter_pd(P, top_n=None, min_persistence=0.0):
    """把持久图裁到"值得当签名用"的那部分。

    Vietoris–Rips 滤流在 H_1 上会产生【成百上千】个短命类（点云里每三个邻近点
    就临时围出一个环）。实测 48 点、40 步滤流的 PD_1 有 535 个特征，几乎全是噪声，
    直接算 W_1 得到的是"噪声条数的差"而不是"形状的差"。
    标准做法是按持久时间 (death−birth) 排序只留最显著的若干个。
    """
    P = np.asarray(P, dtype=np.float64).reshape(-1, 2)
    if P.size == 0:
        return P
    if min_persistence > 0:
        P = P[(P[:, 1] - P[:, 0]) >= min_persistence]
    if top_n is not None and P.shape[0] > top_n:
        pers = P[:, 1] - P[:, 0]
        P = P[np.argsort(-pers)[:top_n]]
    return P


def topology_loss(pd_cur, pd_prev, eta=1.0, min_persistence=0.0, return_parts=False,
                  top_n=None):
    """模块 T 损失：Σ_c [ W1(PD_0^t, PD_0^{t-1}) + η W1(PD_1^t, PD_1^{t-1}) ]
    pd_cur / pd_prev : list of dict {'pd0':..., 'pd1':...}，逐【同一个旧类别】比较。
    min_persistence : 丢弃持久时间 (death-birth) 小于该值的特征，避免保护噪声结构。
    return_parts    : 同时返回 {"pd0": 平均 0 维距离, "pd1": 平均 1 维距离}，便于诊断。
    """
    def filt(P):
        return filter_pd(P, top_n=top_n, min_persistence=min_persistence)

    total, n, s0, s1 = 0.0, 0, 0.0, 0.0
    for a, b in zip(pd_cur, pd_prev):
        d0 = wasserstein_pd(filt(a.get("pd0")), filt(b.get("pd0")))
        d1 = wasserstein_pd(filt(a.get("pd1")), filt(b.get("pd1")))
        total += d0 + eta * d1
        s0 += d0
        s1 += d1
        n += 1
    n = max(n, 1)
    val = total / n
    if return_parts:
        return val, {"pd0": s0 / n, "pd1": s1 / n, "n": n}
    return val


# ============================================================
# 模块 U：未知识别（局部切空间残差）
# ------------------------------------------------------------
# 判断物体能否被已有类别流形解释，而不只是看分类置信度低不低。
#   r_c(z) = || (I - U_c U_c^T)(z - mu_c) ||_{Sigma_c^{-1}}
#   r(z)   = min_c r_c(z)
#   a(z)   = 归一化邻域距离（缺乏任何已知类局部支持的程度）
#   s_unk  = p_obj * sigmoid( alpha*r + beta*log(1+a) + gamma*(1-p_max) - b )
# ============================================================

def local_tangent_residual(z, mu, U, sigma_inv):
    """z: (N, D)；mu: (D,)；U: (D, k) 列正交局部 PCA 基；sigma_inv: (D, D) 或 (D,) 逆协方差（对角）。
    返回 (N,) 每类的残差 r_c。
    """
    dev = z - mu.unsqueeze(0)                        # (N, D)
    P = torch.eye(U.shape[0], device=z.device) - U @ U.t()   # 正交补投影
    orth = dev @ P.t()                               # (N, D)
    if sigma_inv.dim() == 1:
        w = orth * sigma_inv.unsqueeze(0)
    else:
        w = orth @ sigma_inv
    return (orth * w).sum(-1).clamp(min=0).sqrt()     # Mahalanobis 意义下的模


def normalized_neighbor_distance(z, anchors, k=5):
    """a(z)：到 anchors 的第 k 近邻距离 / 该 anchor 集合的典型尺度（中位数近邻距离）。"""
    D = torch.cdist(z, anchors)                       # (N, M)
    dk = D.kthvalue(min(k, D.shape[1]), dim=1).values  # (N,)
    with torch.no_grad():
        AA = torch.cdist(anchors, anchors)
        scale = AA.kthvalue(min(k + 1, AA.shape[1]), dim=1).values.median().clamp(min=1e-6)
    return dk / scale


def fit_unknown_norm(z_known, class_stats, anchors=None, k=5):
    """在【已知类样本】上标定 r 与 log(1+a) 的中心与尺度。

    为什么必须标定：r 是 Mahalanobis 残差，其绝对值取决于特征维度 D 与该类的
    协方差尺度，典型值可以随便落在 1~100 之间（实测 D=256 时已知类中位 ≈8.4、
    未知中位 ≈14.3）。把 α·r 直接送进 sigmoid 会让 σ(αr) 在 0.9997 附近饱和，
    未知和已知的分数都塌成 1.0（实测未知/已知中位数比 = 1.000），
    分数退化成一个常数，AUROC 完全由噪声决定。

    标定方式：中位数作中心、1.4826×MAD 作尺度（对离群点稳健，不用标准差），
    使已知类样本的归一化分量 ~ N(0,1)，于是 α=β=γ=1 才是"等权"的真实含义。
    """
    if anchors is None:
        anchors = torch.stack([st["mu"].to(z_known.device) for st in class_stats], 0)
    rs = []
    for st in class_stats:
        rs.append(local_tangent_residual(z_known, st["mu"].to(z_known.device),
                                         st["U"].to(z_known.device),
                                         st["sigma_inv"].to(z_known.device)))
    r = torch.stack(rs, 0).min(0).values if rs else torch.zeros(z_known.shape[0])
    a = torch.log1p(normalized_neighbor_distance(z_known, anchors.to(z_known.device), k=k))
    out = {}
    for name, x in (("r", r), ("a", a)):
        med = torch.median(x)
        mad = torch.median((x - med).abs())
        scale = 1.4826 * mad
        if not torch.isfinite(scale) or float(scale) < 1e-6:
            scale = x.std().clamp(min=1e-6)
        out[f"{name}_med"] = float(med)
        out[f"{name}_scale"] = float(scale)
    return out


def unknown_score(z, class_stats, p_obj, p_max, alpha=1.0, beta=1.0,
                  gamma=1.0, b=0.0, k=5, anchors=None, norm=None):
    """s_unk = p_obj * sigmoid(alpha*r + beta*log(1+a) + gamma*(1-p_max) - b)
    class_stats : list of dict {'mu','U','sigma_inv'} 每个已知类一份。
    anchors     : 所有已知类锚点拼接 (M, D)，用于算 a(z)；None 则用类均值拼接。
    norm        : fit_unknown_norm 的返回值。给了就把 r 与 log(1+a) 先做
                  (x − med)/scale 标准化；不给则直接用原始量（会饱和，见 fit 文档）。
    返回 (N,) 未知分数。
    """
    device = z.device
    rs = []
    for st in class_stats:
        rs.append(local_tangent_residual(z, st["mu"].to(device), st["U"].to(device),
                                         st["sigma_inv"].to(device)))
    r = torch.stack(rs, dim=0).min(dim=0).values if rs else torch.zeros(z.shape[0], device=device)
    if anchors is None:
        anchors = torch.stack([st["mu"].to(device) for st in class_stats], dim=0)
    a = normalized_neighbor_distance(z, anchors.to(device), k=k)
    la = torch.log1p(a)
    if norm is not None:
        rn = (r - norm["r_med"]) / max(norm["r_scale"], 1e-6)
        an = (la - norm["a_med"]) / max(norm["a_scale"], 1e-6)
    else:
        rn, an = r, la
    logits = alpha * rn + beta * an + gamma * (1.0 - p_max) - b
    return p_obj * torch.sigmoid(logits), {"r": r, "a": a, "r_norm": rn, "a_norm": an}


def box_iou_cxcywh(a, b):
    """a: (N,4) b: (M,4)，cxcywh -> (N,M) IoU"""
    def to_xyxy(x):
        cx, cy, w, h = x.unbind(-1)
        return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)
    A, B = to_xyxy(a), to_xyxy(b)
    lt = torch.max(A[:, None, :2], B[None, :, :2])
    rb = torch.min(A[:, None, 2:], B[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    aa = (A[:, 2] - A[:, 0]) * (A[:, 3] - A[:, 1])
    ab = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-8)


@torch.no_grad()
def greedy_match_queries(q_boxes, gt_boxes, iou_thr=0.5):
    """把 query 贪心匹配到 GT（每个 GT 最多一个 query，IoU < thr 不配对）。

    用贪心而不是匈牙利：这里的目的只是给 query 打【噪声标签】用于累积几何记忆，
    不需要最优匹配；贪心在 IoU 上是 2-近似且快得多（没有 O(n^3) 的指派求解）。
    返回 (q_idx, gt_idx)，等长 LongTensor。
    """
    if gt_boxes.shape[0] == 0 or q_boxes.shape[0] == 0:
        return torch.zeros(0, dtype=torch.long, device=q_boxes.device), \
               torch.zeros(0, dtype=torch.long, device=q_boxes.device)
    iou = box_iou_cxcywh(q_boxes, gt_boxes)
    flat = iou.reshape(-1)
    order = torch.argsort(flat, descending=True)
    q_used = torch.zeros(q_boxes.shape[0], dtype=torch.bool, device=q_boxes.device)
    g_used = torch.zeros(gt_boxes.shape[0], dtype=torch.bool, device=q_boxes.device)
    qs, gs = [], []
    G = gt_boxes.shape[0]
    for f in order.tolist():
        if float(flat[f].item()) < iou_thr:
            break
        q, g = f // G, f % G
        if q_used[q] or g_used[g]:
            continue
        q_used[q] = True
        g_used[g] = True
        qs.append(q)
        gs.append(g)
    return (torch.tensor(qs, dtype=torch.long, device=q_boxes.device),
            torch.tensor(gs, dtype=torch.long, device=q_boxes.device))


def _gram_schmidt_basis(W):
    """
    由 class_embed 行向量构造类别子空间正交基（用于 O 模块）。
    W : (K, D) 类别权重行向量。返回 (D, min(K,D)) 正交基。
    对整块做 QR；对单个类别行，退化为单位方向。
    """
    K, D = W.shape
    # 用 QR 分解：W^T = Q R，Q 的列是 span(W 行) 的正交基
    Wt = W.t()                              # (D, K)
    Q, R = torch.linalg.qr(Wt)              # Q: (D, D)
    # 取前 K 列（span 的基）
    basis = Q[:, :min(K, D)]
    return basis


def per_class_basis(W, class_indices):
    """
    给定 class_embed 权重 W (K_all, D)，为每个指定类别构造子空间基。
    单个类别行 => 单位方向 (D,1)。
    返回 list of (D, k) 基（此处每类 k=1，即行向量归一化方向）。
    """
    bases = []
    for c in class_indices:
        w = W[c].unsqueeze(0)               # (1, D)
        w = w / (w.norm(dim=-1, keepdim=True) + 1e-8)
        bases.append(w.t())                 # (D, 1)
    return bases
