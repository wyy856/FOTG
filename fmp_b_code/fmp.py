"""Functional Manifold Preservation losses and feature-space gradient projection."""

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F


@dataclass
class MatchedOldInstances:
    batch: torch.Tensor
    student_query: torch.Tensor
    teacher_query: torch.Tensor
    target: torch.Tensor
    labels: torch.Tensor
    boxes: torch.Tensor

    @property
    def count(self) -> int:
        return int(self.labels.numel())


def match_old_instances(
    student_outputs: Dict[str, torch.Tensor],
    teacher_outputs: Dict[str, torch.Tensor],
    targets: Sequence[Dict[str, torch.Tensor]],
    matcher,
    prev: int,
    student_matches=None,
) -> MatchedOldInstances:
    """Pair student/teacher query indices by the same Hungarian-matched GT."""
    if student_matches is None:
        student_matches = matcher(student_outputs, targets)
    teacher_matches = matcher(teacher_outputs, targets)
    batch_ids: List[int] = []
    student_ids: List[int] = []
    teacher_ids: List[int] = []
    target_ids: List[int] = []
    labels: List[int] = []
    boxes: List[torch.Tensor] = []

    for batch_idx, (s_match, t_match, target) in enumerate(
        zip(student_matches, teacher_matches, targets)
    ):
        s_src, s_tgt = s_match
        t_src, t_tgt = t_match
        s_by_target = {int(t): int(q) for q, t in zip(s_src.tolist(), s_tgt.tolist())}
        t_by_target = {int(t): int(q) for q, t in zip(t_src.tolist(), t_tgt.tolist())}
        for target_idx in sorted(s_by_target.keys() & t_by_target.keys()):
            label = int(target["labels"][target_idx].item())
            if label < 0 or label >= prev:
                continue
            batch_ids.append(batch_idx)
            student_ids.append(s_by_target[target_idx])
            teacher_ids.append(t_by_target[target_idx])
            target_ids.append(target_idx)
            labels.append(label)
            boxes.append(target["boxes"][target_idx])

    device = student_outputs["pred_logits"].device
    return MatchedOldInstances(
        batch=torch.tensor(batch_ids, device=device, dtype=torch.long),
        student_query=torch.tensor(student_ids, device=device, dtype=torch.long),
        teacher_query=torch.tensor(teacher_ids, device=device, dtype=torch.long),
        target=torch.tensor(target_ids, device=device, dtype=torch.long),
        labels=torch.tensor(labels, device=device, dtype=torch.long),
        boxes=(
            torch.stack(boxes).to(device=device, dtype=torch.float32)
            if boxes
            else torch.empty((0, 4), device=device, dtype=torch.float32)
        ),
    )


class FunctionalManifoldPreservation:
    def __init__(
        self,
        topk: int = 5,
        geo_same_class_weight: float = 1.0,
        geo_cross_class_weight: float = 1.0,
        geo_scale_sigma: float = 1.5,
        geo_tau_eps: float = 1e-6,
        lam_geo: float = 1.0,
        lam_id: float = 1.0,
        lam_fg: float = 1.0,
        lam_loc: float = 1.0,
        projection_rtol: float = 1e-4,
    ):
        if topk < 1:
            raise ValueError("topk must be positive")
        if geo_scale_sigma <= 0 or geo_tau_eps <= 0 or projection_rtol <= 0:
            raise ValueError("scale, distance epsilon, and projection tolerance must be positive")
        if min(
            geo_same_class_weight,
            geo_cross_class_weight,
            lam_geo,
            lam_id,
            lam_fg,
            lam_loc,
        ) < 0:
            raise ValueError("geometry weights and FMP loss weights must be non-negative")
        self.topk = topk
        self.geo_same_class_weight = geo_same_class_weight
        self.geo_cross_class_weight = geo_cross_class_weight
        self.geo_scale_sigma = geo_scale_sigma
        self.geo_tau_eps = geo_tau_eps
        self.lam_geo = lam_geo
        self.lam_id = lam_id
        self.lam_fg = lam_fg
        self.lam_loc = lam_loc
        self.projection_rtol = projection_rtol

    def losses(
        self,
        student_z: torch.Tensor,
        teacher_z: torch.Tensor,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        student_boxes: torch.Tensor,
        teacher_boxes: torch.Tensor,
        matches: MatchedOldInstances,
        prev: int,
        no_object_index: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        zero = student_z.sum() * 0.0
        if matches.count == 0:
            return zero, {
                "geo": zero.detach(),
                "id": zero.detach(),
                "fg": zero.detach(),
                "loc": zero.detach(),
                "old_pairs": zero.detach(),
            }

        b, sq, tq, labels = (
            matches.batch,
            matches.student_query,
            matches.teacher_query,
            matches.labels,
        )
        zs = student_z.reshape(-1, student_z.shape[-1])[b * student_z.shape[1] + sq].float()
        zt = teacher_z.reshape(-1, teacher_z.shape[-1])[b * teacher_z.shape[1] + tq].float()
        logits_s = student_logits[b, sq].float()
        logits_t = teacher_logits[b, tq].detach().float()
        boxes_s = student_boxes[b, sq].float()
        boxes_t = teacher_boxes[b, tq].detach().float()

        geo = self._geometry_loss(zs, zt, labels, matches.boxes)
        old_count = min(prev, no_object_index, logits_s.shape[-1], logits_t.shape[-1])
        if old_count < 2:
            identity = zero
        else:
            old_logits_t = logits_t[:, :old_count]
            old_logits_s = logits_s[:, :old_count]
            competitor_scores = old_logits_t.clone()
            competitor_scores.scatter_(1, labels[:, None], float("-inf"))
            k = min(self.topk, old_count - 1)
            competitors = competitor_scores.topk(k, dim=1).indices
            target_s = old_logits_s.gather(1, labels[:, None])
            target_t = old_logits_t.gather(1, labels[:, None])
            margin_s = target_s - old_logits_s.gather(1, competitors)
            margin_t = target_t - old_logits_t.gather(1, competitors)
            identity = F.smooth_l1_loss(margin_s, margin_t)

        if no_object_index < 0 or no_object_index >= logits_s.shape[-1]:
            raise ValueError(
                f"no_object_index {no_object_index} is outside logits dimension "
                f"{logits_s.shape[-1]}"
            )
        target_s = logits_s.gather(1, labels[:, None])
        target_t = logits_t.gather(1, labels[:, None])
        foreground = F.smooth_l1_loss(
            target_s - logits_s[:, no_object_index:no_object_index + 1],
            target_t - logits_t[:, no_object_index:no_object_index + 1],
        )
        localization = F.smooth_l1_loss(boxes_s, boxes_t)
        total = (
            self.lam_geo * geo
            + self.lam_id * identity
            + self.lam_fg * foreground
            + self.lam_loc * localization
        )
        return total, {
            "geo": geo.detach(),
            "id": identity.detach(),
            "fg": foreground.detach(),
            "loc": localization.detach(),
            "old_pairs": student_z.new_tensor(float(matches.count)),
        }

    def _geometry_loss(
        self,
        student_z: torch.Tensor,
        teacher_z: torch.Tensor,
        labels: torch.Tensor,
        boxes: torch.Tensor,
    ) -> torch.Tensor:
        count = student_z.shape[0]
        if count < 2:
            return student_z.sum() * 0.0
        distances_t = torch.cdist(teacher_z.detach().float(), teacher_z.detach().float())
        distances_s = torch.cdist(student_z.float(), student_z.float())
        off_diagonal = ~torch.eye(count, dtype=torch.bool, device=student_z.device)
        tau = distances_t[off_diagonal].mean().clamp_min(self.geo_tau_eps)
        same = labels[:, None] == labels[None, :]
        class_weight = torch.where(
            same,
            student_z.new_tensor(self.geo_same_class_weight),
            student_z.new_tensor(self.geo_cross_class_weight),
        )
        areas = (boxes[:, 2] * boxes[:, 3]).clamp_min(self.geo_tau_eps).log()
        scale_weight = torch.exp(
            -(areas[:, None] - areas[None, :]).abs() / self.geo_scale_sigma
        )
        weights = (class_weight * scale_weight).masked_fill(~off_diagonal, 0.0)
        error = (distances_s / tau - distances_t / tau).square()
        return (weights * error).sum() / float(count * count)

    def functional_jacobian(
        self,
        hs_last: torch.Tensor,
        student_logits: torch.Tensor,
        student_boxes: torch.Tensor,
        teacher_logits: torch.Tensor,
        matches: MatchedOldInstances,
        prev: int,
        no_object_index: int,
    ) -> torch.Tensor:
        """Return rows of d(identity, foreground, localization)/d(hs_last)."""
        if matches.count == 0:
            return hs_last.new_empty((0, hs_last.shape[-1]), dtype=torch.float32)
        b, sq, tq, labels = (
            matches.batch,
            matches.student_query,
            matches.teacher_query,
            matches.labels,
        )
        query_count = student_logits.shape[1]
        logits_s = student_logits[b, sq]
        logits_t = teacher_logits[b, tq].detach()
        boxes_s = student_boxes[b, sq]
        num_old = min(prev, no_object_index, logits_s.shape[-1], logits_t.shape[-1])
        if num_old < 2:
            return hs_last.new_empty((0, hs_last.shape[-1]), dtype=torch.float32)
        competitor_scores = logits_t[:, :num_old].clone()
        competitor_scores.scatter_(1, labels[:, None], float("-inf"))
        competitors = competitor_scores.topk(min(self.topk, num_old - 1), dim=1).indices
        target_logits = logits_s.gather(1, labels[:, None])
        margins = target_logits - logits_s.gather(1, competitors)
        foreground = target_logits - logits_s[:, no_object_index:no_object_index + 1]
        function_values = torch.cat((margins, foreground, boxes_s), dim=1)

        import os as _os
        if _os.environ.get('FMP_DEBUG'):
            print(f"[FMP_J_DEBUG] hs_req={hs_last.requires_grad} hs_gfn={hs_last.grad_fn is not None} "
                  f"logits_req={logits_s.requires_grad} boxes_req={boxes_s.requires_grad} "
                  f"fv_req={function_values.requires_grad} num_old={num_old} nchan={function_values.shape[1]}", flush=True)

        rows = []
        for channel in range(function_values.shape[1]):
            gradient = torch.autograd.grad(
                function_values[:, channel].sum(),
                hs_last,
                retain_graph=True,
                allow_unused=True,
            )[0]
            if _os.environ.get('FMP_DEBUG') and channel == 0:
                print(f"[FMP_J_DEBUG] ch0 grad={'None' if gradient is None else tuple(gradient.shape)}", flush=True)
            if gradient is not None:
                regular_gradient = gradient.reshape(
                    gradient.shape[0], -1, gradient.shape[-1]
                )[:, -query_count:, :]
                rows.append(regular_gradient[b, sq])
        if not rows:
            return hs_last.new_empty((0, hs_last.shape[-1]), dtype=torch.float32)
        return torch.cat(rows, dim=0).detach().float()


def project_to_functional_nullspace(
    new_gradient: torch.Tensor,
    jacobian: torch.Tensor,
    rtol: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project feature gradients using the row-space nullspace of a feature Jacobian."""
    if jacobian.numel() == 0 or new_gradient.numel() == 0:
        ratio = new_gradient.new_tensor(1.0, dtype=torch.float32)
        return new_gradient, ratio
    if new_gradient.shape[-1] != jacobian.shape[-1]:
        raise ValueError(
            f"feature dimensions differ: gradient={new_gradient.shape[-1]}, "
            f"jacobian={jacobian.shape[-1]}"
        )
    j = jacobian.detach().reshape(-1, jacobian.shape[-1]).float()
    _, singular_values, vh = torch.linalg.svd(j, full_matrices=False)
    if singular_values.numel() == 0:
        return new_gradient, new_gradient.new_tensor(1.0, dtype=torch.float32)
    threshold = singular_values.max() * rtol
    basis = vh[singular_values > threshold]
    g = new_gradient.float()
    if basis.numel():
        projected = g - (g @ basis.T) @ basis
    else:
        projected = g
    before = g.square().sum().clamp_min(1e-20)
    ratio = projected.square().sum() / before
    return projected.to(dtype=new_gradient.dtype), ratio.detach()


def new_detection_feature_gradient(
    detection_loss: torch.Tensor,
    outputs: Dict[str, torch.Tensor],
    hs_last: torch.Tensor,
    targets: Sequence[Dict[str, torch.Tensor]],
    prev: int,
    cur: int,
    num_superclass: int,
    student_matches: Sequence[Tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    """Extract only new-class detection gradients at hs_last from the detector loss."""
    pred_logits = outputs["pred_logits"]
    pred_boxes = outputs["pred_boxes"]
    pred_super = outputs.get("pred_super_logits")
    differentiated = [pred_logits, pred_boxes]
    if pred_super is not None:
        differentiated.append(pred_super)
    gradients = torch.autograd.grad(
        detection_loss,
        tuple(differentiated),
        retain_graph=True,
        allow_unused=True,
    )
    grad_logits, grad_boxes = gradients[:2]
    grad_super = gradients[2] if pred_super is not None else None

    if grad_logits is None:
        return torch.zeros_like(hs_last)
    new_end = min(prev + cur, pred_logits.shape[-1])
    class_masked = torch.zeros_like(grad_logits)
    if new_end > prev:
        class_masked[..., prev:new_end] = grad_logits[..., prev:new_end]

    box_masked = torch.zeros_like(grad_boxes) if grad_boxes is not None else None
    superclass_masked = torch.zeros_like(grad_super) if pred_super is not None and grad_super is not None else None
    for batch_idx, (src_indices, target_indices) in enumerate(student_matches):
        src_indices = src_indices.to(pred_logits.device)
        target_indices = target_indices.to(pred_logits.device)
        labels = targets[batch_idx]["labels"].to(pred_logits.device)[target_indices]
        is_new = (labels >= prev) & (labels < prev + cur)
        if not bool(is_new.any()):
            continue
        src_new = src_indices[is_new]
        if box_masked is not None:
            box_masked[batch_idx, src_new] = grad_boxes[batch_idx, src_new]
        if superclass_masked is not None and "superclasses" in targets[batch_idx]:
            supers = targets[batch_idx]["superclasses"].to(pred_super.device)[target_indices[is_new]]
            valid = (supers >= 0) & (supers < num_superclass)
            if bool(valid.any()):
                superclass_masked[batch_idx, src_new[valid], supers[valid]] = \
                    grad_super[batch_idx, src_new[valid], supers[valid]]

    differentiable_outputs = [pred_logits]
    grad_outputs = [class_masked]
    if pred_boxes is not None and box_masked is not None:
        differentiable_outputs.append(pred_boxes)
        grad_outputs.append(box_masked)
    if pred_super is not None and superclass_masked is not None:
        differentiable_outputs.append(pred_super)
        grad_outputs.append(superclass_masked)
    feature_gradient = torch.autograd.grad(
        tuple(differentiable_outputs),
        hs_last,
        grad_outputs=tuple(grad_outputs),
        retain_graph=True,
        allow_unused=True,
    )[0]
    return torch.zeros_like(hs_last) if feature_gradient is None else feature_gradient
