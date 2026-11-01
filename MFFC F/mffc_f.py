# mffc_f.py
# =============================================================================
# MFFC-F : Multi-channel Functional Fibre Constraint  (FOTG 的新版 F 模块)
# -----------------------------------------------------------------------------
# 核心思想：保护旧类 *功能*，不冻结旧类 *表征*。
#   - F1 Identity  : 旧类类别身份保持（Top-K 教师竞争类 margin）
#   - F2 Foreground: 旧类前景属性保持（old-class vs no-object margin）
#   - F3 Localization: 旧类定位能力保持（GT 对齐的 bbox SmoothL1）
#   - L_fib        : 局部 Jacobian 功能纤维约束（教师特征处对功能映射求 J_F，
#                    惩罚学生特征沿"改变旧类功能"的方向漂移；不做固定 PCA 方向硬投影）
#
# 三个实现边界（来自方法定义）：
#   1) 需要旧类输入：仅在"已匹配到旧类 GT 的 query"上计算，旧类样本缺失时该通道为 0。
#   2) DETR 定位按 GT 对齐：教师与学生各自独立匹配到同一 GT，再比较对应预测，
#      不依赖固定 query slot。
#   3) 三通道独立归一化：分类 margin 与 bbox 坐标尺度不同，分别归一化后再进 L_fib，
#      不把异尺度量直接拼成一个平方误差。
# =============================================================================

import torch
import torch.nn.functional as F


def _iou_1toN(box, boxes):
    """单框 box [4] 与 boxes [M,4] (cx,cy,w,h, 归一化) 的 IoU，返回 [M]。"""
    def to_xyxy(b):
        cx, cy, w, h = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
        return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)
    a = to_xyxy(box)                       # [4]
    b = to_xyxy(boxes)                     # [M,4]
    xa = torch.maximum(a[0], b[:, 0])       # [M]
    ya = torch.maximum(a[1], b[:, 1])
    xb = torch.minimum(a[2], b[:, 2])
    yb = torch.minimum(a[3], b[:, 3])
    iw = (xb - xa).clamp(min=0)
    ih = (yb - ya).clamp(min=0)
    inter = iw * ih
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a + area_b - inter
    return inter / union.clamp(min=1e-6)


def gt_aligned_pairs(boxes_pred_s, boxes_pred_t, gt_boxes, gt_labels, prev, iou_thr=0.5):
    """对单张图的旧类 GT，分别用学生/教师预测框做 GT 对齐贪心匹配。

    返回 list[(idx_s, idx_t, y, gt_box)]：
      - idx_s: 学生预测框索引（匹配到该 GT）
      - idx_t: 教师预测框索引（匹配到同一 GT）
      - y    : GT 类别（< prev，旧类）
      - gt_box: GT 框（用于 F3 监控，比较在预测-预测之间进行）
    仅当学生与教师都匹配上同一 GT（IoU>=iou_thr）才保留。
    """
    pairs = []
    if gt_boxes.numel() == 0:
        return pairs
    for gi in range(gt_boxes.shape[0]):
        y = int(gt_labels[gi].item())
        if y >= prev:
            continue  # 只看旧类 GT
        gbox = gt_boxes[gi]
        iou_s = _iou_1toN(gbox, boxes_pred_s)
        iou_t = _iou_1toN(gbox, boxes_pred_t)
        si = int(iou_s.argmax().item()) if iou_s.numel() else -1
        ti = int(iou_t.argmax().item()) if iou_t.numel() else -1
        if si < 0 or ti < 0:
            continue
        if iou_s[si] < iou_thr or iou_t[ti] < iou_thr:
            continue
        pairs.append((si, ti, y, gbox))
    return pairs


class MFFCFConstraint:
    """多通道功能纤维约束（MFFC-F）。

    接口（由 engine 在 FOTG 块内调用）：
        loss, parts = mffc_f(
            z_S, z_T,                      # [Nq_total, D] 学生/教师末层特征（slot 对齐）
            logits_S, logits_T,            # [B, n_reg, C] 学生/教师分类 logits
            boxes_S, boxes_T,              # [B, n_reg, 4] 学生/教师预测框（归一化 cxcywh）
            targets,                       # list[dict]，含 'boxes'(GT) 与 'labels'
            prev, num_known, no_obj_idx,   # 旧类数 / 已知类总数 / no-object 索引(通常=num_known)
            head_fn=None,                  # 可选：z -> (cls_logits, bbox)，用于 L_fib 的 J_F
        )
    parts 为 dict，记录 F1/F2/F3/L_fib 各分量，便于监控。
    """

    def __init__(self, topk=5, iou_thr=0.5,
                 lam_id=1.0, lam_fg=1.0, lam_loc=1.0, lam_fib=0.1,
                 loc_scale=1.0, fib_w_id=None, fib_w_fg=1.0, fib_w_loc=1.0,
                 temp=1.0):
        """
        topk      : F1 使用的教师 Top-K 旧类竞争类数
        iou_thr   : GT 对齐匹配的 IoU 阈值
        lam_*     : 各通道权重（主约束 = L_id+L_fg+L_loc；L_fib 为几何正则）
        loc_scale : 定位坐标的额外缩放（若 boxes 为像素坐标则设大值；归一化坐标用 1.0）
        fib_w_*   : L_fib 中三通道的独立归一化权重 W_F（满足"独立归一化"边界）
        temp      : 身份 margin 的温度尺度（与 fg 在同一 logit 量级，默认 1.0）
        """
        self.topk = topk
        self.iou_thr = iou_thr
        self.lam_id = lam_id
        self.lam_fg = lam_fg
        self.lam_loc = lam_loc
        self.lam_fib = lam_fib
        self.loc_scale = loc_scale
        self.fib_w_id = fib_w_id          # None -> 自动按 sqrt(topk) 归一
        self.fib_w_fg = fib_w_fg
        self.fib_w_loc = fib_w_loc
        self.temp = temp

    # ------------------------------------------------------------------ #
    # 主调用
    # ------------------------------------------------------------------ #
    def __call__(self, z_S, z_T, logits_S, logits_T, boxes_S, boxes_T,
                 targets, prev, num_known, no_obj_idx, head_fn=None):
        B, n_reg, C = logits_S.shape
        device = logits_S.device
        total = 0.0
        parts = {}

        # 收集所有旧类 GT 对齐对（跨 batch 逐图处理）
        s_idx_all, t_idx_all, y_all = [], [], []
        for b in range(B):
            gb = targets[b].get('boxes', None)
            gl = targets[b].get('labels', None)
            if gb is None or gl is None or gb.numel() == 0:
                continue
            pairs = gt_aligned_pairs(
                boxes_S[b], boxes_T[b], gb.float(), gl.long(), prev, self.iou_thr)
            for (si, ti, y, _gbox) in pairs:
                s_idx_all.append(b * n_reg + si)
                t_idx_all.append(b * n_reg + ti)
                y_all.append(y)

        if len(s_idx_all) == 0:
            # 旧类输入缺失：三个功能通道与 L_fib 均为 0（边界 1）
            parts = {k: torch.zeros((), device=device)
                     for k in ('L_id', 'L_fg', 'L_loc', 'L_fib', 'n_pairs')}
            parts['n_pairs'] = torch.zeros((), device=device)
            return torch.zeros((), device=device), parts

        s_idx = torch.tensor(s_idx_all, device=device, dtype=torch.long)
        t_idx = torch.tensor(t_idx_all, device=device, dtype=torch.long)
        y = torch.tensor(y_all, device=device, dtype=torch.long)          # [P]
        P = y.shape[0]

        # ---- F1: Identity preservation（Top-K 教师竞争类 margin 蒸馏） ----
        # 教师在其匹配 query 上挑 Top-K 旧类竞争类（训练中不随学生漂移 -> 稳定）
        logS = logits_S.reshape(B * n_reg, C).float()
        logT = logits_T.reshape(B * n_reg, C).float()
        old_mask = torch.arange(prev, device=device)                        # [prev]
        # 教师 Top-K 竞争类（排除 y 自身）
        teacher_at_t = logT[t_idx]                                          # [P, C]
        comp_scores = teacher_at_t.index_select(1, old_mask)                # [P, prev]
        # 把 y 自身分数压到 -inf，避免被选为竞争者
        y_col = y.unsqueeze(1)                                             # [P,1]
        comp_mask = (old_mask.unsqueeze(0) == y_col)                       # [P,prev] True=自身
        comp_scores = comp_scores.masked_fill(comp_mask, float('-inf'))
        k = min(self.topk, prev - 1)
        topk_vals, topk_idx = torch.topk(comp_scores, k, dim=1)            # [P,k]
        # 竞争类全局索引
        comp_global = old_mask.unsqueeze(0).expand(P, prev).gather(1, topk_idx)  # [P,k]
        # margin: s_y - s_j
        s_y_S = logS[s_idx].gather(1, y_col)                              # [P,1]
        s_y_T = logT[t_idx].gather(1, y_col)                              # [P,1]
        s_j_S = logS[s_idx].gather(1, comp_global)                                 # [P,k]
        s_j_T = logT[t_idx].gather(1, comp_global)                                 # [P,k]
        m_id_S = (s_y_S.expand(-1, k) - s_j_S) / self.temp                # [P,k]
        m_id_T = (s_y_T.expand(-1, k) - s_j_T) / self.temp
        L_id = F.smooth_l1_loss(m_id_S, m_id_T.detach())
        parts['L_id'] = L_id.detach()
        total = total + self.lam_id * L_id

        # ---- F2: Foregroundness preservation（old vs no-object margin） ----
        s_fg_S = logS[s_idx].gather(1, y_col) - logS[s_idx][:, no_obj_idx:no_obj_idx + 1]
        s_fg_T = logT[t_idx].gather(1, y_col) - logT[t_idx][:, no_obj_idx:no_obj_idx + 1]
        L_fg = F.smooth_l1_loss(s_fg_S, s_fg_T.detach())
        parts['L_fg'] = L_fg.detach()
        total = total + self.lam_fg * L_fg

        # ---- F3: Localization preservation（GT 对齐 bbox 蒸馏） ----
        bx_S = boxes_S.reshape(B * n_reg, 4)[s_idx]                       # [P,4]
        bx_T = boxes_T.reshape(B * n_reg, 4)[t_idx]                       # [P,4]
        bx_S = bx_S * self.loc_scale
        bx_T = bx_T * self.loc_scale
        L_loc = F.smooth_l1_loss(bx_S, bx_T.detach())
        parts['L_loc'] = L_loc.detach()
        total = total + self.lam_loc * L_loc

        # ---- L_fib: 局部 Jacobian 功能纤维约束（可选，依赖 head_fn） ----
        L_fib_v = torch.zeros((), device=device)
        if head_fn is not None:
            L_fib_v = self._fib_term(z_S, z_T, head_fn, prev, no_obj_idx)
        parts['L_fib'] = L_fib_v.detach()
        total = total + self.lam_fib * L_fib_v

        parts['n_pairs'] = torch.tensor(float(P), device=device)
        return total, parts

    # ------------------------------------------------------------------ #
    # L_fib: J_F(z_T) (z_S - z_T) 的二范数平方，按 W_F 独立归一化三通道
    # 仅对教师-学生都有的旧类 query 做（用教师 argmax-old 作为该 query 的功能身份）
    # ------------------------------------------------------------------ #
    def _fib_term(self, z_S, z_T, head_fn, prev, no_obj_idx):
        # z_T 需保留图以计算 J_F；z_S - z_T 的方向 stopgrad（仅作投影方向）
        z_T = z_T.detach().requires_grad_(True)
        cls_T, box_T = head_fn(z_T)                 # [N,D]->([N,C],[N,4])
        N = z_T.shape[0]
        # 每 query 的功能身份 y = argmax over 旧类（纯由 z 决定的旧类功能定义）
        old_logits = cls_T[:, :prev]                # [N, prev]
        y = old_logits.argmax(dim=1)                # [N]
        y_col = y.unsqueeze(1)                      # [N,1]
        # 功能向量 g_old(z) 的"身份"部分：y 类对 Top-K 竞争类的 margin
        k = min(self.topk, prev - 1)
        comp = old_logits.clone()
        comp.scatter_(1, y_col, float('-inf'))
        _, topk_idx = torch.topk(comp, k, dim=1)    # [N,k]
        s_y = old_logits.gather(1, y_col)           # [N,1]
        s_j = old_logits.gather(1, topk_idx)                           # [N,k]
        g_id = (s_y.expand(-1, k) - s_j) / self.temp            # [N,k]
        g_fg = (s_y - cls_T[:, no_obj_idx:no_obj_idx + 1])      # [N,1]
        g_loc = box_T                                    # [N,4]
        # 拼接（先按 W_F 独立归一化，满足边界 3）
        w_id = self.fib_w_id if self.fib_w_id is not None else float(k ** 0.5)
        g_id = g_id / w_id
        g_fg = g_fg / self.fib_w_fg
        g_loc = g_loc / self.fib_w_loc
        g = torch.cat([g_id, g_fg, g_loc], dim=1)   # [N, M], M=k+1+4
        M = g.shape[1]

        diff = (z_S - z_T)                          # [N,D] 投影方向：保留 z_S 的图，梯度回流学生特征
        v = torch.zeros(N, M, device=z_T.device)
        for m in range(M):
            grad = torch.autograd.grad(g[:, m].sum(), z_T,
                                       retain_graph=True, allow_unused=True)[0]
            if grad is None:
                grad = torch.zeros_like(z_T)
            v[:, m] = (grad * diff).sum(dim=-1)      # J_F(z_T) 对第 m 功能方向的方向导数
        return (v ** 2).mean()
