#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FOTG 完整实现：模块 F / O / T / U + 几何记忆 + 带约束的统一优化。

=============================================================================
总览
=============================================================================
F  Functional fibre    功能纤维：约束"旧类读出函数"在当前特征上的取值不漂移
                       L_fib = Σ_q w_q [‖d_q^cls‖²/C_old + η‖d_q^box‖²/4] / Σ_q w_q
                       读出口 = 门控前 logits + bbox。logits = Wz+b 对 z 线性，
                       故雅可比恒为 W_old、与 z 无关——v1 里 J_c ∝ s_c 被门控压到
                       1e-9 的结构性缺陷到此消除。
O  Overlap regulation  重叠度调控：用子空间主角度刻画类间重叠
                       O_ij = (1/k)‖U_iᵀ U_j‖_F²
                       旧-旧保持，新-旧按 O* = clip(O_0 + αS_ij − βC_ij, O_min, O_max)
T  Topological growth  拓扑生长：持久同调签名（PD_0/PD_1）跨阶段的 W_1 距离
U  Unknown ID          未知拒识：局部切空间残差 + 归一化近邻距离
                       r_c(z) = ‖(I − U_cU_cᵀ)(z − μ_c)‖_{Σ^{-1}}
                       s_unk  = p_obj · σ(αr + β log(1+a) + γ(1 − p_max) − b)

统一优化（§VI）：
    min_θ  L_det(θ)
    s.t.   C_fib(θ) ≤ ε_fib,  C_ov(θ) ≤ ε_ov,  C_topo(θ) ≤ ε_topo
拉格朗日松弛 + 对偶上升：
    L(θ, λ) = L_det + Σ_i λ_i (C_i − ε_i)
    λ_i ← clamp(λ_i + η_i (C_i − ε_i), 0, λ_max)
λ 初值由【共享参数上的梯度范数比】标定（fotg_calib.py），
使 λ_i‖∇C_i‖ / ‖∇L_det‖ 初始落在 ρ=0.05 量级，避免手工拍系数。

文件职责
--------
  fotg_losses.py   纯 loss 数学（已单测）
  fotg_module.py   本文件：状态、几何记忆、约束计算、对偶更新
  fotg_patch.py    对 engine.py / dn_dab_deformable_detr_o1o.py 的最小侵入接线
  fotg_calib.py    λ 初值标定
  fotg_probe.py    前向结果缓存（离线消融用）
  fotg_ablate.py   离线消融
"""

import os
import json
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- 权威 superclass 映射（与 datasets/torchvision_datasets/open_world_o1o.py 一致） ----
VOC = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
       "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
       "sheep", "sofa", "train", "tvmonitor"]
T2 = ["elephant", "bear", "zebra", "giraffe", "truck", "traffic light", "fire hydrant",
      "stop sign", "parking meter", "bench", "backpack", "umbrella", "handbag", "tie",
      "suitcase", "microwave", "oven", "toaster", "sink", "refrigerator"]
T3 = ["frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
      "baseball glove", "skateboard", "surfboard", "tennis racket", "banana", "apple",
      "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake"]
T4 = ["bed", "toilet", "laptop", "mouse", "remote", "keyboard", "cell phone", "book",
      "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
      "wine glass", "cup", "fork", "knife", "spoon", "bowl"]
CLASS_NAMES = VOC + T2 + T3 + T4                      # 80 类，顺序即 M-OWODB 官方顺序

MOWODB_T4_SUPERCLASSES = {
    "animal": ["bird", "cat", "cow", "dog", "horse", "sheep", "elephant", "bear", "zebra", "giraffe"],
    "vehicle": ["aeroplane", "bicycle", "boat", "bus", "car", "motorbike", "train", "truck"],
    "person": ["person"],
    "furniture": ["chair", "diningtable", "pottedplant", "sofa", "bed", "toilet"],
    "electronic": ["tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone"],
    "kitchen": ["bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl"],
    "accessory": ["backpack", "umbrella", "handbag", "tie", "suitcase"],
    "appliance": ["microwave", "oven", "toaster", "sink", "refrigerator"],
    "outdoor": ["traffic light", "fire hydrant", "stop sign", "parking meter", "bench"],
    "sports": ["frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
               "baseball glove", "skateboard", "surfboard", "tennis racket"],
    "food": ["banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
             "pizza", "donut", "cake"],
    "indoor": ["book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"],
    "unknown": ["unknown"],
}
SUP_LIST = list(MOWODB_T4_SUPERCLASSES.keys())                 # 13 个（含 unknown）
NAME2SUP = {v: k for k in MOWODB_T4_SUPERCLASSES for v in MOWODB_T4_SUPERCLASSES[k]}
CLS2SUP = [SUP_LIST.index(NAME2SUP[c]) for c in CLASS_NAMES]   # 80 维 -> 0..12
SUP_IDX_OF_CLS = CLS2SUP


# =============================================================================
# 几何记忆：跨阶段的类级统计（一/二阶矩流式更新 -> PCA 子空间、锚点、点云、持久图）
# =============================================================================
class GeometricMemory:
    """每个已见类的几何记忆。

    不存原始数据，只维护流式充分统计量：
        n_c, Σ_c = Σ z,  M_c = Σ z zᵀ
    由此可随时解出
        μ_c = Σ_c / n_c
        Cov_c = M_c/n_c − μ_c μ_cᵀ
        U_c = Cov_c 的前 k 个特征向量（局部切空间基）
        σ_c² = diag(Cov_c)  （用于 Σ^{-1} 加权的残差范数）
    """

    def __init__(self, dim, num_classes=80, k=8, max_cloud=128,
                 cov_eps=1e-4, device="cpu"):
        self.dim = dim
        self.num_classes = num_classes
        self.k = k
        self.max_cloud = max_cloud
        self.cov_eps = cov_eps
        self.device = device

        self.register_buffer = dict(
            n=torch.zeros(num_classes, device=device),
            s=torch.zeros(num_classes, dim, device=device),
            m=torch.zeros(num_classes, dim, dim, device=device),
        )
        self.cloud = [None] * num_classes        # FIFO 特征云（拓扑签名用）
        self.pd_sig = dict()                     # 类 -> {"pd0":..., "pd1":...}（上一阶段签名）
        self._basis_cache = {}

    # ---------------- 流式累积 ----------------
    @torch.no_grad()
    def accumulate(self, feats, labels):
        """feats (M, D) float；labels (M,) long，取值为真实类别 id（未知用 num_classes 之外的 id 传 None 过滤）。"""
        if feats.numel() == 0:
            return
        dev = feats.device
        n = self.register_buffer["n"].to(dev)
        s = self.register_buffer["s"].to(dev)
        m = self.register_buffer["m"].to(dev)
        f = feats.detach().to(dev).float()
        lb = labels.detach().to(dev).long()
        keep = (lb >= 0) & (lb < self.num_classes)
        if int(keep.sum()) == 0:
            return
        f, lb = f[keep], lb[keep]
        n.index_add_(0, lb, torch.ones_like(f[:, :1]).expand(-1, 1)[:, 0].float()
                     if False else torch.ones(f.shape[0], device=dev, dtype=n.dtype))
        s.index_add_(0, lb, f)
        # (M, D, D) 外积；M 通常 ~100，D=256 -> 6.5M 元素，训练时每 accum_every 步一次可接受
        outer = f.unsqueeze(2) * f.unsqueeze(1)
        m.index_add_(0, lb, outer)
        self.register_buffer["n"], self.register_buffer["s"], self.register_buffer["m"] = n, s, m

        # FIFO 特征云
        for c in lb.unique().tolist():
            cf = f[lb == c]
            cur = self.cloud[c]
            cur = cf if cur is None else torch.cat([cur, cf], dim=0)
            if cur.shape[0] > self.max_cloud:
                cur = cur[-self.max_cloud:]
            self.cloud[c] = cur
        self._basis_cache.clear()

    # ---------------- 类级统计量 ----------------
    @torch.no_grad()
    def stats(self, c):
        """返回 dict(mu, U, sigma_inv, n)。U: (D, k) 列正交；sigma_inv: (D,) 各坐标 1/σ。"""
        if c in self._basis_cache:
            return self._basis_cache[c]
        dev = self.device
        n = float(self.register_buffer["n"][c].item())
        if n < 2:
            out = {"mu": torch.zeros(self.dim, device=dev),
                   "U": torch.zeros(self.dim, 0, device=dev),
                   "sigma_inv": torch.ones(self.dim, device=dev),
                   "n": n}
            self._basis_cache[c] = out
            return out
        mu = self.register_buffer["s"][c] / n
        cov = self.register_buffer["m"][c] / n - torch.outer(mu, mu)
        cov = 0.5 * (cov + cov.t())
        evals, evecs = torch.linalg.eigh(cov)
        order = torch.argsort(evals, descending=True)
        k = min(self.k, self.dim)
        U = evecs[:, order[:k]]
        var = torch.clamp(torch.diagonal(cov), min=self.cov_eps)
        sigma_inv = 1.0 / torch.sqrt(var)
        out = {"mu": mu.to(dev), "U": U.to(dev), "sigma_inv": sigma_inv.to(dev), "n": n}
        self._basis_cache[c] = out
        return out

    @torch.no_grad()
    def all_stats(self, classes):
        return [self.stats(int(c)) for c in classes]

    # ---------------- 模块 O：重叠矩阵 ----------------
    @torch.no_grad()
    def overlap_matrix(self, classes):
        """O_ij = (1/k)‖U_iᵀ U_j‖_F²，对称、对角=1。"""
        from fotg_losses import subspace_overlap
        K = len(classes)
        O = torch.eye(K, device=self.device)
        Us = [self.stats(int(c))["U"] for c in classes]
        for i in range(K):
            for j in range(i + 1, K):
                v = subspace_overlap(Us[i], Us[j])
                O[i, j] = v
                O[j, i] = v
        return O

    # ---------------- 模块 T：持久同调签名 ----------------
    @torch.no_grad()
    def persistence_signature(self, classes, max_points=64):
        """对每个类的特征云算 (PD_0, PD_1)。点云做均衡采样，避免类间样本数差异污染。"""
        from fotg_losses import rips_pd0, rips_pd1
        sig = {}
        for c in classes:
            cloud = self.cloud[int(c)]
            if cloud is None or cloud.shape[0] < 8:
                continue
            X = cloud.detach().float().cpu().numpy()
            if X.shape[0] > max_points:
                idx = torch.linspace(0, X.shape[0] - 1, max_points).long().numpy()
                X = X[idx]
            sig[int(c)] = {"pd0": rips_pd0(X), "pd1": rips_pd1(X)}
        return sig

    def store_signature(self, classes):
        """阶段结束时固化签名，作为下一阶段拓扑约束的参照。"""
        self.pd_sig = self.persistence_signature(classes)
        return self.pd_sig

    # ---------------- 序列化 ----------------
    def state_dict(self):
        return {k: v.detach().cpu() for k, v in self.register_buffer.items()}, \
               {k: (v.detach().cpu() if v is not None else None)
                for k, v in enumerate(self.cloud)}, \
               {k: {"pd0": v["pd0"], "pd1": v["pd1"]} for k, v in self.pd_sig.items()}

    def load_state_dict(self, sd):
        buf, cloud, sig = sd
        for k, v in buf.items():
            self.register_buffer[k] = v.to(self.device)
        for k, v in cloud.items():
            self.cloud[int(k)] = None if v is None else v.to(self.device)
        self.pd_sig = {int(k): v for k, v in sig.items()}
        self._basis_cache.clear()


# =============================================================================
# FOTG 主模块
# =============================================================================
class FOTGModule(nn.Module):
    """挂在训练循环上的 FOTG 状态机。

    用法（见 fotg_patch.py）：
        fotg = FOTGModule(model, old_model, prev, cur, cfg)
        out  = model(samples, ...)                 # 当前模型前向
        z_cur = hs[-1]                             # (B, Q, D)
        with torch.no_grad():
            hs_old, out_old, _ = old_model(samples, ...)
        z_ref = hs_old[-1]
        c = fotg.compute(z_cur, z_ref, out_old, out, targets, step)
        loss = fotg.total_loss(L_det, c, step)
    """

    def __init__(self, model, old_model, prev_cls, cur_cls, cfg=None):
        super().__init__()
        cfg = cfg or {}
        self.prev = int(prev_cls)
        self.cur = int(cur_cls)
        self.num_known = self.prev + self.cur              # 当前已知类数
        # 用 object.__setattr__ 绕过 nn.Module 的属性注册：否则 model / old_model 会被
        # 登记成子模块，一旦 FOTGModule 被挂进 criterion，optimizer 会把【冻结的旧模型
        # 参数】也收进去（虽然 requires_grad=False 不会更新，但会污染参数组与状态字典）。
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "old_model", old_model)
        if old_model is not None:
            self.old_model.eval()
            for p in self.old_model.parameters():
                p.requires_grad_(False)

        # ---- 超参 ----
        self.k_sub = int(cfg.get("k_sub", 8))
        self.eta_box = float(cfg.get("eta_box", 1.0))       # F 中定位项权重
        self.w_threshold = float(cfg.get("w_threshold", 0.0))
        self.w_topk = int(cfg.get("w_topk", 10))
        self.w_soft = bool(cfg.get("w_soft", True))
        self.w_mode = str(cfg.get("w_mode", "blend"))
        self.num_queries = int(cfg.get("num_queries", 100))

        # 重叠调控
        self.O_0 = float(cfg.get("O_0", 0.5))
        self.alpha_sup = float(cfg.get("alpha_sup", 0.1))
        self.beta_conf = float(cfg.get("beta_conf", 0.1))
        self.O_min = float(cfg.get("O_min", 0.2))
        self.O_max = float(cfg.get("O_max", 0.8))
        # True = 用实测旧-旧重叠反推的先验覆盖上面三个绝对值（consolidate 时算出）
        self.use_calibrated_prior = bool(cfg.get("use_calibrated_prior", True))
        self.O_prior = None

        # 拓扑 / 拒识
        self.eta_pd1 = float(cfg.get("eta_pd1", 1.0))
        self.min_persistence = float(cfg.get("min_persistence", 0.0))
        self.topo_every = int(cfg.get("topo_every", 50))
        self.unk_alpha = float(cfg.get("unk_alpha", 1.0))
        self.unk_beta = float(cfg.get("unk_beta", 1.0))
        self.unk_gamma = float(cfg.get("unk_gamma", 1.0))
        self.unk_bias = float(cfg.get("unk_bias", 0.0))
        self.unk_k = int(cfg.get("unk_k", 5))

        # 约束预算 ε（可由 calib 自动标定，见 set_budgets）
        self.eps = {"fib": float(cfg.get("eps_fib", 1e-3)),
                    "ov": float(cfg.get("eps_ov", 1e-2)),
                    "topo": float(cfg.get("eps_topo", 0.5)),
                    "unk": float(cfg.get("eps_unk", 0.2)),
                    "route": float(cfg.get("eps_route", 0.05))}
        # 对偶上升的"单位步长"：更新量按【相对 gap】(C/ε − 1) 计，再乘 lam_unit。
        # 直接用绝对 gap (C−ε) 会让步长完全取决于 ε 的绝对量级（ε~1e-3 时
        # 每次只动 1e-3，等于不更新），必须归一化。
        self.lam_unit = {k: float(cfg.get(f"lam_unit_{k}", 1.0))
                         for k in ("fib", "route", "ov", "topo", "unk")}
        # 对偶上升步长与上限
        self.dual_lr = {"fib": float(cfg.get("dual_lr_fib", 1.0)),
                        "ov": float(cfg.get("dual_lr_ov", 1.0)),
                        "topo": float(cfg.get("dual_lr_topo", 1.0)),
                        "unk": float(cfg.get("dual_lr_unk", 1.0)),
                        "route": float(cfg.get("dual_lr_route", 1.0))}
        self.lam_max = {"fib": float(cfg.get("lam_max_fib", 1e3)),
                        "ov": float(cfg.get("lam_max_ov", 1e3)),
                        "topo": float(cfg.get("lam_max_topo", 1e2)),
                        "unk": float(cfg.get("lam_max_unk", 1e2)),
                        "route": float(cfg.get("lam_max_route", 1e3))}
        # λ 初值（由 fotg_calib.py 的梯度范数比给出；0 表示纯对偶上升从 0 起）
        self.lam = {"fib": float(cfg.get("lam_fib", 0.0)),
                    "ov": float(cfg.get("lam_ov", 0.0)),
                    "topo": float(cfg.get("lam_topo", 0.0)),
                    "unk": float(cfg.get("lam_unk", 0.0)),
                    "route": float(cfg.get("lam_route", 0.0))}

        self.dual_every = int(cfg.get("dual_every", 20))
        self.O_prev = None

        # 几何记忆
        dim = int(cfg.get("dim", 256))
        self.memory = GeometricMemory(dim, num_classes=80, k=self.k_sub,
                                      max_cloud=int(cfg.get("max_cloud", 128)),
                                      device="cpu")
        self.accum_every = int(cfg.get("accum_every", 20))
        self.sup_idx = torch.tensor(SUP_IDX_OF_CLS, dtype=torch.long)

        # 冻结的旧读出（F 的核心：雅可比与 z 无关，故只需权重，不必每步重算）
        self.W_old = None
        self.b_old = None
        self.old_bbox_head = None
        if old_model is not None:
            self._cache_old_readout()

        self.history = []

    # ------------------------------------------------------------------
    def _cache_old_readout(self):
        """缓存旧模型最后一层 class_embed 的旧类权重与 bbox 头。

        logits = W z + b 对 z 是线性的 => ∂logits_c/∂z = W_c，与 z 无关。
        这正是 v2 相对 v1 的结构性修正：v1 用门控后分数 s_c = p_c·g_σ(c)，
        J_c ∝ s_c，一旦某类置信度低雅可比就塌成 0（实测 L_fib ~1e-9）。
        """
        om = self.old_model
        Wc = om.class_embed[-1].weight.detach()
        bc = om.class_embed[-1].bias.detach()
        self.register_buffer("W_old_buf", Wc[:self.prev].clone())
        self.register_buffer("b_old_buf", bc[:self.prev].clone())
        self.W_old = self.W_old_buf
        self.b_old = self.b_old_buf
        self.old_bbox_head = om.bbox_embed[-1]

    @torch.no_grad()
    def old_scores(self, logits_old, sup_logits_old, mode="blend"):
        """旧模型的门控分数 s，仅用于【挑选要保护哪些 query】——不再进入雅可比
        （v1 的 J_c ∝ s_c 正是 L_fib 塌到 1e-9 的根因）。

        mode:
          "product"  s = p_cls · g_sup           原始定义，量级最小（两概率相乘）
          "cls"      s = p_cls                   只用分类置信度
          "blend"    s = p_cls · (0.5 + 0.5 g_sup)（默认）
                     保留 superclass 先验但避免量级被乘没：g_sup 退化成
                     [0.5, 1] 区间的调制因子而不是乘法衰减因子。
        """
        dev = logits_old.device
        sup_idx = self.sup_idx.to(dev)
        p_cls = torch.softmax(logits_old[..., :-1].float(), dim=-1)
        g_sup = torch.softmax(sup_logits_old[..., :-1].float(), dim=-1)
        if mode == "cls":
            return p_cls
        if mode == "product":
            return p_cls * g_sup[..., sup_idx]
        return p_cls * (0.5 + 0.5 * g_sup[..., sup_idx])

    # ---------------- 模块 F ----------------
    def compute_fib(self, z_cur, z_ref, s_old=None):
        """功能纤维约束值 C_fib。

        z_cur / z_ref: (N, D)，N = B*Q，来自当前模型与冻结旧模型的最后一层 query 特征。
        s_old: (N, num_known) 门控分数，用于挑选"旧类 query"。
        """
        from fotg_losses import bbox_jacobian, query_weight_old_class, functional_fibre_loss_v2
        N, D = z_cur.shape
        if self.prev == 0 or self.W_old is None:
            return z_cur.new_zeros(()), {"n_valid": 0}
        if s_old is None:
            w_q = None
        else:
            w_q = query_weight_old_class(
                s_old, self.prev, threshold=self.w_threshold, topk=self.w_topk,
                num_queries=self.num_queries, soft=self.w_soft)
        J_box = bbox_jacobian(z_ref.detach(), self.old_bbox_head)      # (N, 4, D)
        self._J_box = J_box                                           # 缓存给 F-B 复用
        loss, parts = functional_fibre_loss_v2(
            z_cur, z_ref.detach(), self.W_old, J_box, w_q=w_q,
            eta=self.eta_box, return_parts=True)
        return loss, parts

    # ---------------- 模块 B：Jacobian 软投影（事前主动干预更新方向） ----------------
    def old_readout_jacobian(self, z_ref):
        """旧类读出雅可比 J_full : (N, C_old + 4, D)，供 F-B 软投影用。

        每行是一个旧类读出（或 bbox 分量）对 query 特征 z 的雅可比：
          - 前 C_old 行 = W_old（分类读出，对 z 线性，精确 = ∂logits/∂z）；
          - 后 4 行    = J_box（bbox 读出雅可比，autograd 真算）。
        z_ref 来自冻结旧模型（detach），与 z 无关，故只依赖权重与当前输入。
        返回 None 表示无旧类（task1 / prev==0），F-B 退化为恒等投影。
        """
        if self.prev == 0 or self.W_old is None:
            return None
        N = z_ref.shape[0]
        W = self.W_old.detach()                              # (C_old, D)
        J_cls = W.unsqueeze(0).expand(N, -1, -1)            # (N, C_old, D)
        J_box = getattr(self, "_J_box", None)
        if J_box is None or J_box.shape[0] != N:
            from fotg_losses import bbox_jacobian
            J_box = bbox_jacobian(z_ref.detach(), self.old_bbox_head)
        return torch.cat([J_cls, J_box], dim=1)             # (N, C_old + 4, D)

    # ---------------- 模块 F-route：路由感知功能保持（Phase 1） ----------------
    def compute_route(self, out_cur, out_old, s_old=None):
        """路由感知功能保持（Phase 1）：直接约束当前 decoder 对旧类的最终路由分数
        s_c = p_c · q_{g(c)} 不漂移（train 约定 σ(cls)·σ(sup)），而非仅约束 W_old·z
        的线性代理。梯度经 out_cur 回传到当前 cls/sup head → decoder → encoder，
        因此真正进入训练图。参考目标来自冻结旧模型同输入输出（detach）。
        与 compute_fib 的线性代理（W_old·Δz + J_box·Δz）互补，由独立 λ_route 控制。
        """
        if self.prev == 0 or out_cur is None or out_old is None:
            if out_cur is not None:
                return out_cur["pred_logits"].new_zeros(())
            return torch.zeros(())
        w_q = None
        if s_old is not None:
            from fotg_losses import query_weight_old_class
            w_q = query_weight_old_class(
                s_old, self.prev, threshold=self.w_threshold, topk=self.w_topk,
                num_queries=self.num_queries, soft=self.w_soft)
        from fotg_losses import route_aware_preservation
        # gate_mode 与 old_scores() 默认 self.w_mode 保持一致：两者都用 "blend"。
        # 这是 Fix-2 修复：Route 损失内部的 gate 与 query 权重来源（old_scores）
        # 必须使用同一公式，否则被 Route 约束的量和用于选 query 的量不是同一个函数。
        return route_aware_preservation(
            out_cur, out_old, self.sup_idx, self.prev, w_q=w_q,
            gate_mode=self.w_mode)

    # ---------------- 模块 O ----------------
    def compute_overlap(self, class_embed_weight):
        """重叠调控约束值 C_ov。

        旧-旧：‖O_cur − O_prev‖²（保持上一阶段几何关系）
        新-旧：‖O_new_old − O*‖²，O* = clip(O_0 + αS − βC, O_min, O_max)
        """
        from fotg_losses import per_class_basis, subspace_overlap, overlap_target, overlap_new_loss
        W = class_embed_weight
        dev = W.device
        old_ids = list(range(self.prev))
        new_ids = list(range(self.prev, self.num_known))
        if self.prev == 0:
            return W.new_zeros(()), {"old_term": 0.0, "new_term": 0.0}

        old_b = per_class_basis(W, old_ids)          # list of (D,1) 单位方向
        new_b = per_class_basis(W, new_ids)

        # 新-旧重叠 + 语义/冲突先验
        O_no = torch.zeros(self.cur, self.prev, device=dev)
        S = torch.zeros(self.cur, self.prev, device=dev)
        C = torch.zeros(self.cur, self.prev, device=dev)
        sup_idx = self.sup_idx.to(dev)
        for i in range(self.cur):
            for j in range(self.prev):
                O_no[i, j] = subspace_overlap(new_b[i], old_b[j])
                ci = self.prev + i
                same = (sup_idx[ci] == sup_idx[j]).to(O_no.dtype)
                S[i, j] = same
                C[i, j] = 1.0 - same
        # O_0 / O_min / O_max：优先用【从实测旧-旧重叠标定出来的】先验。
        # 理由见 fotg_losses.calibrate_overlap_prior —— 拍脑袋的 O_0=0.5 与实测
        # 类别权重方向重叠（~0.01）差 40 倍，会让 100% 的新-旧对判为"违背"，
        # 约束退化成与几何无关的常数惩罚。
        pr = getattr(self, "O_prior", None)
        if pr is not None and self.use_calibrated_prior:
            o0, omin, omax = pr["O_0"], pr["O_min"], pr["O_max"]
        else:
            o0, omin, omax = self.O_0, self.O_min, self.O_max
        O_star = overlap_target(S, C, O_0=o0, alpha=self.alpha_sup,
                                beta=self.beta_conf, O_min=omin, O_max=omax)
        new_term = overlap_new_loss(O_no, O_star)

        old_term = W.new_zeros(())
        O_prev = getattr(self, "O_prev", None)
        if O_prev is not None:
            # O_prev 是在上一阶段末尾按【当时全部已知类】(prev+cur) 固化的，进入新阶段后
            # prev 变大；这里取左上角 prev×prev 子块对齐。尺寸不足时直接跳过而不是报错。
            O_prev = O_prev.to(dev)
            if O_prev.shape[0] >= self.prev:
                Op = O_prev[:self.prev, :self.prev]
                O_cur = torch.zeros(self.prev, self.prev, device=dev)
                for i in range(self.prev):
                    for j in range(i, self.prev):
                        v = subspace_overlap(old_b[i], old_b[j])
                        O_cur[i, j] = v
                        O_cur[j, i] = v
                old_term = ((O_cur - Op) ** 2).sum() / max(self.prev ** 2, 1)
        return old_term + new_term, {"old_term": float(old_term.item()),
                                     "new_term": float(new_term.item())}

    # ---------------- 模块 T ----------------
    def compute_topo(self, step, classes=None):
        """拓扑生长约束值 C_topo：当前特征云与上一阶段持久图签名之间的 W_1 距离。

        只在 step % topo_every == 0 时计算（持久同调用 numpy，成本高）。
        """
        from fotg_losses import topology_loss
        if not self.memory.pd_sig or step % self.topo_every != 0:
            return None, {}
        classes = classes or list(range(self.num_known))
        sig_cur = self.memory.persistence_signature(classes)
        losses, d0, d1 = [], [], []
        for c, cur in sig_cur.items():
            prev = self.memory.pd_sig.get(c)
            if prev is None:
                continue
            l, p = topology_loss([cur], [prev], eta=self.eta_pd1,
                                 min_persistence=self.min_persistence, return_parts=True)
            losses.append(l)
            d0.append(p.get("pd0", float("nan")))
            d1.append(p.get("pd1", float("nan")))
        if not losses:
            return None, {}
        val = sum(losses) / len(losses)
        return val, {"n_cls": len(losses),
                     "pd0": float(sum(d0) / len(d0)),
                     "pd1": float(sum(d1) / len(d1))}

    # ---------------- 模块 U ----------------
    def compute_unk(self, z_cur, p_obj, p_max, labels=None):
        """未知拒识。labels 给出时返回可训练的 BCE 约束；否则只返回分数。

        机制：s_unk 完全由【冻结的几何记忆】算出，梯度只经 z 回传到 encoder/decoder，
        于是"把未知样本推离所有已知类切空间"变成一个真正的可优化目标。
        """
        from fotg_losses import unknown_score
        classes = list(range(self.num_known))
        stats = self.memory.all_stats(classes)
        anchors = _collect_anchors(self.memory, classes, z_cur.device)
        s_unk, diag = unknown_score(
            z_cur, stats, p_obj, p_max,
            alpha=self.unk_alpha, beta=self.unk_beta, gamma=self.unk_gamma,
            b=self.unk_bias, k=self.unk_k, anchors=anchors)
        if labels is None:
            return s_unk, diag
        # labels: (N,) long；未知 = num_classes(=80) 或 -1
        is_unk = (labels >= 80) | (labels < 0)
        if int(is_unk.sum()) == 0 or int((~is_unk).sum()) == 0:
            return z_cur.new_zeros(()), diag
        tgt = is_unk.to(s_unk.dtype)
        loss = F.binary_cross_entropy(s_unk, tgt)
        return loss, diag

    @torch.no_grad()
    def match_labels_to_queries(self, outputs, targets, iou_thr=0.5):
        """给每个 query 打一个【由 GT 决定的】标签，供几何记忆与模块 U 使用。

        未匹配上的 query 记为 -1（不参与累积、不参与 U 的 BCE）；
        未知 GT 在 O1O 里 label = num_classes-1 = 80（datasets/open_world_eval.py:50），
        伪标签框同样用 80，因此这里不需要额外判断。
        """
        from fotg_losses import greedy_match_queries
        pb = outputs["pred_boxes"]                     # (B, Q, 4)
        B, Q = pb.shape[0], pb.shape[1]
        chunks = []
        for i, t in enumerate(targets):
            labels = torch.full((Q,), -1, dtype=torch.long, device=pb.device)
            gb = t["boxes"]
            if gb is not None and gb.numel() > 0:
                q, g = greedy_match_queries(pb[i].detach(), gb.detach(), iou_thr)
                if q.numel():
                    labels[q] = t["labels"][g].to(labels.dtype)
            chunks.append(labels)
        return torch.cat(chunks, 0)

    # ---------------- 统一入口 ----------------
    @torch.no_grad()
    def maybe_accumulate(self, z_cur, labels, step):
        if step % self.accum_every != 0 or labels is None:
            return
        self.memory.accumulate(z_cur.detach().float().cpu(),
                               labels.detach().cpu().long())

    def compute(self, z_cur, z_ref, out_old, out_cur, targets=None, step=0):
        """算出全部约束值 + 诊断量。不更新 λ（由 dual_step 负责）。"""
        dev = z_cur.device
        N = z_cur.shape[0]
        s_old = None
        if out_old is not None and self.prev > 0:
            # ★batch>1 修复：pred_logits 是 [B, n_reg, C]（已 dn_post_process 去 DN），
            #   flatten 成 [-1, C] = [B*n_reg, C] 正好等于 z_cur 的 N（engine 已按 batch
            #   切到末 n_reg 行再 flatten）。旧写法 reshape(N, -1) 在 batch=2 时把
            #   [2,100,81]=16200 硬塞成 [100,162]，[:-1] 后 161 与 sup_idx 的 80 冲突。
            lg = out_old["pred_logits"].reshape(-1, out_old["pred_logits"].shape[-1])
            sp = out_old["pred_super_logits"].reshape(-1, out_old["pred_super_logits"].shape[-1])
            # 兜底：万一行数与 N 不符（防未来 num_patterns>1 等边界），截断/补到 N 行
            if lg.shape[0] != N:
                lg = lg[-N:] if lg.shape[0] > N else lg
            if sp.shape[0] != N:
                sp = sp[-N:] if sp.shape[0] > N else sp
            s_old = self.old_scores(lg, sp, mode=self.w_mode)

        c_fib, p_fib = self.compute_fib(z_cur, z_ref, s_old)
        c_route = self.compute_route(out_cur, out_old, s_old)
        W = self.model.class_embed[-1].weight
        c_ov, p_ov = self.compute_overlap(W)

        p_obj = torch.sigmoid(out_cur["pred_logits"][..., -1].reshape(N)).detach() \
            if out_cur is not None and out_cur["pred_logits"].shape[-1] > self.num_known \
            else torch.ones(N, device=dev)
        p_max = torch.softmax(out_cur["pred_logits"].reshape(N, -1)[..., :80].detach().float(),
                              dim=-1).max(dim=-1).values \
            if out_cur is not None else torch.zeros(N, device=dev)
        c_unk, p_unk = self.compute_unk(z_cur, p_obj.to(dev), p_max.to(dev), None)
        c_topo, p_topo = self.compute_topo(step)

        c = {"fib": c_fib, "route": c_route, "ov": c_ov, "topo": c_topo, "unk_score": c_unk}
        parts = {"fib": p_fib,
                 "route": float(c_route.item()) if hasattr(c_route, "item") else c_route,
                 "ov": p_ov, "topo": p_topo, "unk": p_unk}
        return c, parts

    def total_loss(self, L_det, c, step=0):
        """L = L_det + Σ λ_i max(0, C_i − ε_i)。C_topo 为 None（未到计算步）时跳过。

        预算门 max(0, ·) 是拉格朗日松弛的正确形式：
        - C > eps：超预算，产生正惩罚项，推动 C 下降；
        - C ≤ eps：预算内，惩罚项 = 0，不施加任何梯度信号（不管 C 多小）。
        无条件 lam*(C-eps) 会在 C < eps 时给负惩罚（奖励），推动训练继续降低 C，
        与"预算内不应有任何约束"的设计语义矛盾。
        """
        loss = L_det
        used = {}
        for k in ("fib", "route", "ov", "topo"):
            v = c.get(k)
            if v is None:
                continue
            lam = self.lam[k]
            if lam <= 0:
                continue
            # ⚠️ 不用 torch.clamp(v-eps, min=0) —— 它在 v=eps 边界处给次梯度 1，
            # 把边界点当成超预算处理。预算满足时边界点应零梯度：
            #   C < eps → 零梯度（不推动继续降低 C）
            #   C = eps → 零梯度（边界也算满足）
            #   C > eps → 正梯度（推动 C 下降）
            # 手动实现 ReLU 门控，让梯度在 v <= eps 时精确为 0：
            diff = v - self.eps[k]
            gate = (diff > 0).to(diff.dtype)   # 只有严格正 diff 才放行梯度
            term = lam * diff * gate
            loss = loss + term
            used[k] = float(term.item())
        return loss, used

    @torch.no_grad()
    def dual_step(self, c):
        """对偶上升：λ_i ← clamp(λ_i + η_i (C_i − ε_i), 0, λ_max_i)。"""
        upd = {}
        for k in ("fib", "route", "ov", "topo"):
            v = c.get(k)
            if v is None:
                continue
            eps = max(self.eps[k], 1e-12)
            gap = float(v.item()) - eps                 # 绝对 gap：写进日志用
            rel = float(v.item()) / eps - 1.0           # 相对 gap：真正驱动步长
            new = self.lam[k] + self.dual_lr[k] * rel * self.lam_unit[k]
            new = max(0.0, min(new, self.lam_max[k]))
            upd[k] = (self.lam[k], new, gap, rel)
            self.lam[k] = new
        return upd

    # ---------------- 阶段切换 ----------------
    @torch.no_grad()
    def consolidate(self, class_embed_weight=None):
        """任务结束时：固化重叠矩阵 O_prev、持久图签名、清空缓存。"""
        if class_embed_weight is None and self.model is not None:
            class_embed_weight = self.model.class_embed[-1].weight
        self.O_prev = self.memory.overlap_matrix(list(range(self.num_known))) \
            if self.num_known > 0 else None
        self.memory.store_signature(list(range(self.num_known)))
        self._cache_old_readout()
        return {"O_prev_shape": None if self.O_prev is None else list(self.O_prev.shape),
                "pd_classes": sorted(self.memory.pd_sig.keys()),
                "O_prior": self.O_prior}


def _supports_parts(fn):
    import inspect
    return "return_parts" in inspect.signature(fn).parameters


def _collect_anchors(memory, classes, device):
    """把各类特征云拼成一个大 anchor 矩阵，供近邻距离使用。"""
    chunks = []
    for c in classes:
        v = memory.cloud[int(c)] if int(c) < len(memory.cloud) else None
        if v is not None and v.numel() > 0:
            chunks.append(v.detach().to(device).float())
    if not chunks:
        return None
    return torch.cat(chunks, dim=0)
