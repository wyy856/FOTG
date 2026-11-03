"""
Unknown Head 分数分布与阈值扫描探针 (B checkpoint, GPU).

三类 query 的 pred_unk.sigmoid() 分布 (用训练同一 Hungarian matcher 划分, 与
loss_unk 监督推导同构):
  A) 匹配 Unknown GT(label==80) 的 query  -> 正样本分数是否足够高
  B) 匹配 Known GT(label 0..39) 的 query  -> 是否错误激活 Unknown
  C) 未匹配背景 query                    -> 是否被正确压低

阈值扫描 (gate 口径, 模拟 eval_mode=True 部署逻辑):
  query 判 unk 若 unk_score > thr, 否则归 known (argmax over 有效 known 0..39,
  sigmoid x sup 乘积, 与官方 eval_mode=False 同口径).
  每个 thr 记录: U_R50 (unknown 检测对 unknown GT 的贪心召回, IoU>=0.5),
  AOSA (known 检测与 unknown GT IoU>=0.5 的检测数), 及 known 侧附带损失.

用法: python unk_head_score_probe.py <N_IMG>
"""
import sys, os, json, argparse
sys.path.insert(0, "/dev/shm/unkhead/fmp")
sys.path.insert(0, "/dev/shm/unkhead/proj")

import torch
import torchvision.ops as tvops
from util.misc import nested_tensor_from_tensor_list
from util.box_ops import box_cxcywh_to_xyxy
from models import build_model
from datasets.torchvision_datasets.open_world_o1o import OWDetection
from datasets.coco import make_coco_transforms

CKPT = "/dev/shm/unkhead_out/task2_train/checkpoint.pth"
DATA_ROOT = "/icislab/volume1/wyy/O1O-main/data/OWOD"
TEST_SET = "owod_all_task_test_nodup"
ANN_DIR = os.path.join(DATA_ROOT, "Annotations")

THRS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n_img", type=int, default=300)
    args_cli = ap.parse_args()

    ck = torch.load(CKPT, map_location="cpu")
    args = ck["args"]
    print("ckpt epoch:", ck.get("epoch"), "unk_head_w:", getattr(args, "unk_head_w", None), flush=True)
    assert getattr(args, "unk_head_w", 0.0) > 0, "B ckpt must have unk_head_w>0"
    args.device = "cuda"
    args.num_gpus = 0
    args.distributed = False
    args.output_dir = "/dev/shm"
    args.data_root = DATA_ROOT
    args.test_set = TEST_SET
    args.eval = True
    args.eval_threshold = 0.0

    device = torch.device("cuda")
    model, criterion, postprocessors, _ = build_model(args, mode=args.model_type)
    model.to(device)
    model.eval()
    criterion.to(device)
    missing, unexpected = model.load_state_dict(ck["model"], strict=True)
    print("loaded strict. missing=%d unexpected=%d" % (len(missing), len(unexpected)), flush=True)

    matcher = criterion.matcher
    unk_idx = args.num_classes - 1
    PREV = int(args.PREV_INTRODUCED_CLS)
    CUR = int(args.CUR_INTRODUCED_CLS)
    CK_LO, CK_HI = PREV, PREV + CUR  # [20, 40)

    ds = OWDetection(args, DATA_ROOT, image_set=TEST_SET, dataset=args.dataset,
                     transforms=make_coco_transforms(TEST_SET))

    # XML 快速预扫描 (不读图): 类名不在前 PREV+CUR 个已知类 -> unknown(80),
    # 与 label_known_class_and_unknown 的映射口径一致.
    import xml.etree.ElementTree as ET
    CLASS_NAMES = ds.CLASS_NAMES
    VOC_COCO = None
    try:
        from datasets.torchvision_datasets.open_world_o1o import (
            VOC_CLASS_NAMES_COCOFIED, BASE_VOC_CLASS_NAMES)
        VOC_COCO = (VOC_CLASS_NAMES_COCOFIED, BASE_VOC_CLASS_NAMES)
    except Exception:
        pass
    known_names = set(CLASS_NAMES[:CK_HI])

    def has_unknown_gt(str_imgid):
        path = os.path.join(ANN_DIR, str_imgid + ".xml")
        if not os.path.exists(path):
            return False
        root = ET.parse(path).getroot()
        for obj in root.findall("object"):
            nm = obj.find("name").text
            if VOC_COCO is not None and nm in VOC_COCO[0]:
                nm = VOC_COCO[1][VOC_COCO[0].index(nm)]
            if nm not in known_names:
                return True
        return False

    valid_idx = [i for i in range(len(ds.str_imgids)) if has_unknown_gt(ds.str_imgids[i])]
    print("images with unknown GT:", len(valid_idx), "of", len(ds), flush=True)
    torch.manual_seed(0)
    perm = torch.randperm(len(valid_idx)).tolist()
    sel = [valid_idx[j] for j in perm[:args_cli.n_img]]
    print("selected", len(sel), "images", flush=True)

    groups = {"unk_pos": [], "known_neg": [], "bg_neg": []}
    unk_pos_iou50 = []          # 正样本中 box 也对 (与 GT IoU>=0.5) 的分数
    scan = {thr: {"u_recall_num": 0, "aosa": 0, "n_known_det": 0, "ck_det": 0, "ck_tp": 0}
            for thr in THRS}
    # 无 gate 基线 (thr=0, 全 known)
    base = {"aosa": 0, "ck_det": 0, "ck_tp": 0}
    n_unk_gt_total = 0
    n_ck_gt_total = 0
    n_img_done = 0

    def greedy_match(det_boxes, det_scores, gt_boxes):
        """按 score 降序贪心, 返回匹配的 GT 索引集合与各 det 的 GT 索引(-1 无)."""
        order = torch.argsort(det_scores, descending=True)
        dev = det_boxes.device
        gt_used = torch.zeros(len(gt_boxes), dtype=torch.bool, device=dev)
        det2gt = -torch.ones(len(det_boxes), dtype=torch.int64, device=dev)
        for d in order.tolist():
            if len(gt_boxes) == 0:
                break
            ious = tvops.box_iou(det_boxes[d:d + 1], gt_boxes)[0]   # [G]
            ious = torch.where(gt_used, torch.zeros_like(ious), ious)
            v, g = ious.max(0)                                       # 0-dim scalars
            if v.item() >= 0.5:
                det2gt[d] = g.item()
                gt_used[g] = True
        return det2gt, gt_used

    with torch.no_grad():
        for i in sel:
            try:
                img, target = ds[i]
                if not isinstance(img, torch.Tensor):
                    continue
                samples = nested_tensor_from_tensor_list([img.to(device)])
                tgt = {"labels": target["labels"].to(device),
                       "boxes": target["boxes"].to(device)}
                out, _ = model(samples, dn_args=args.num_patterns)
                pred_unk = out["pred_unk"].squeeze(-1).sigmoid()[0]          # [Q]
                pred_logits = out["pred_logits"][0]                           # [Q, C]
                pred_boxes = out["pred_boxes"][0]                             # [Q,4] cxcywh norm

                # 训练同款 Hungarian matching
                outputs_without_aux = {"pred_logits": out["pred_logits"],
                                       "pred_boxes": out["pred_boxes"]}
                indices = matcher(outputs_without_aux, [tgt])
                pred_idx, tgt_idx = indices[0]
                pred_idx = pred_idx.to(device)   # matcher 返回 CPU 索引, 统一到 GPU (同 loss_unk 处理)
                tgt_idx = tgt_idx.to(device)
                tgt_labels = tgt["labels"]
                matched_labels = tgt_labels[tgt_idx]

                unk_scores = pred_unk.to(device)

                # ---- 三类 query 分数收集 ----
                matched_pred = torch.zeros(len(unk_scores), dtype=torch.bool, device=device)
                matched_pred[pred_idx] = True
                is_unk_pos = torch.zeros(len(unk_scores), dtype=torch.bool, device=device)
                is_unk_pos[pred_idx[matched_labels == unk_idx]] = True
                is_known_neg = torch.zeros(len(unk_scores), dtype=torch.bool, device=device)
                is_known_neg[pred_idx[matched_labels != unk_idx]] = True
                is_bg = ~matched_pred

                groups["unk_pos"] += unk_scores[is_unk_pos].tolist()
                groups["known_neg"] += unk_scores[is_known_neg].tolist()
                groups["bg_neg"] += unk_scores[is_bg].tolist()

                # 正样本 box 质量分层: 匹配到的 unknown GT 与预测框 IoU>=0.5
                sel_pred = pred_idx[matched_labels == unk_idx]
                sel_tgt = tgt_idx[matched_labels == unk_idx]
                if len(sel_pred) > 0:
                    pb = pred_boxes[sel_pred].to(device)
                    gb = tgt["boxes"][sel_tgt].to(device)
                    ious = tvops.box_iou(pb, gb).diag()
                    unk_pos_iou50 += unk_scores[sel_pred][ious >= 0.5].tolist()

                # ---- 阈值扫描 (gate 口径) ----
                gt_boxes_xyxy = box_cxcywh_to_xyxy(tgt["boxes"])
                size = target["orig_size"].unsqueeze(0).to(device)
                results = postprocessors["bbox"](out, size, eval_mode=False, eval_threshold=0.0)
                res = results[0]
                k_scores = res["scores"].to(device)
                k_labels = res["labels"].to(device)
                k_boxes = res["boxes"].to(device)
                # res 与 query 一一对应? postprocess 取 topk(可能重复同一 query 多 label);
                # gate 需 query 级判定 -> 重建 known 检测: 对非 gate query 取 argmax(0..39)+score.
                valid_known_logits = pred_logits[:, :CK_HI].sigmoid()  # 0..39
                if valid_known_logits.shape[1] > 0:
                    k_score_q, k_label_q = valid_known_logits.max(-1)
                else:
                    k_score_q = torch.zeros(len(unk_scores), device=device)
                    k_label_q = torch.zeros(len(unk_scores), dtype=torch.int64, device=device)

                unk_mask_gt = (tgt_labels == unk_idx)
                ck_mask_gt = (tgt_labels >= CK_LO) & (tgt_labels < CK_HI)
                n_unk_gt_total += int(unk_mask_gt.sum().item())
                n_ck_gt_total += int(ck_mask_gt.sum().item())
                unk_gt_boxes = gt_boxes_xyxy[unk_mask_gt]
                ck_gt_boxes = gt_boxes_xyxy[ck_mask_gt]

                pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes)

                for thr in THRS:
                    gate = unk_scores > thr
                    rec = scan[thr]
                    # unknown 检测: gate query, box=pred box, score=unk score
                    u_boxes = pred_boxes_xyxy[gate]
                    u_scores = unk_scores[gate]
                    if len(u_boxes) > 0 and len(unk_gt_boxes) > 0:
                        _, gt_cov = greedy_match(u_boxes, u_scores, unk_gt_boxes)
                        rec["u_recall_num"] += int(gt_cov.sum().item())
                    # known 检测: 非 gate query (sigmoid x sup 同官方口径略去 sup, 此处用纯 sigmoid)
                    kb = pred_boxes_xyxy[~gate]
                    kl = k_label_q[~gate]
                    ks = k_score_q[~gate]
                    rec["n_known_det"] += int((~gate).sum().item())
                    rec["ck_det"] += int(((kl >= CK_LO) & (kl < CK_HI)).sum().item())
                    if len(kb) > 0:
                        if len(unk_gt_boxes) > 0:
                            d2g, _ = greedy_match(kb, ks, unk_gt_boxes)
                            rec["aosa"] += int((d2g >= 0).sum().item())
                        ck_m = (kl >= CK_LO) & (kl < CK_HI)
                        if ck_m.any() and len(ck_gt_boxes) > 0:
                            d2gc, _ = greedy_match(kb[ck_m], ks[ck_m], ck_gt_boxes)
                            rec["ck_tp"] += int((d2gc >= 0).sum().item())
                    else:
                        pass

                # 无 gate 基线: 全部 query 按 eval_mode=False 的官方结果 (res)
                if len(unk_gt_boxes) > 0:
                    kmask_known = (k_labels < unk_idx) & (k_labels >= 0)
                    d2g0, _ = greedy_match(k_boxes[kmask_known], k_scores[kmask_known], unk_gt_boxes)
                    base["aosa"] += int((d2g0 >= 0).sum().item())
                ckm = (k_labels >= CK_LO) & (k_labels < CK_HI)
                base["ck_det"] += int(ckm.sum().item())
                if ckm.any() and len(ck_gt_boxes) > 0:
                    d2gc0, _ = greedy_match(k_boxes[ckm], k_scores[ckm], ck_gt_boxes)
                    base["ck_tp"] += int((d2gc0 >= 0).sum().item())

                n_img_done += 1
            except Exception as e:
                import traceback
                print("skip img", i, "err", repr(e), flush=True)
                traceback.print_exc()
                continue

    def stats(v):
        if not v:
            return {"n": 0}
        t = torch.tensor(v)
        hist = torch.histc(t, bins=10, min=0.0, max=1.0)
        return {
            "n": len(v),
            "mean": round(t.mean().item(), 4),
            "median": round(t.median().item(), 4),
            "p10": round(torch.quantile(t, 0.1).item(), 4),
            "p90": round(torch.quantile(t, 0.9).item(), 4),
            "frac>0.5": round((t > 0.5).float().mean().item(), 4),
            "frac>0.7": round((t > 0.7).float().mean().item(), 4),
            "hist_0.1bins": [int(x) for x in hist.tolist()],
        }

    out = {
        "ckpt_epoch": ck.get("epoch"),
        "n_images_done": n_img_done,
        "n_unk_gt": n_unk_gt_total,
        "n_ck_gt": n_ck_gt_total,
        "score_stats": {k: stats(v) for k, v in groups.items()},
        "unk_pos_with_iou50": stats(unk_pos_iou50),
        "no_gate_baseline": {
            "AOSA": base["aosa"], "CK_det": base["ck_det"], "CK_TP": base["ck_tp"],
            "CK_precision_proxy": round(base["ck_tp"] / max(base["ck_det"], 1), 4),
        },
        "threshold_scan": [],
    }
    for thr in THRS:
        r = scan[thr]
        out["threshold_scan"].append({
            "thr": thr,
            "U_R50": round(r["u_recall_num"] / max(n_unk_gt_total, 1), 4),
            "AOSA": r["aosa"],
            "n_known_det": r["n_known_det"],
            "CK_det": r["ck_det"],
            "CK_TP": r["ck_tp"],
            "CK_precision_proxy": round(r["ck_tp"] / max(r["ck_det"], 1), 4),
        })
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
