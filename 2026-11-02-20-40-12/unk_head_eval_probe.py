"""
独立 Unknown Head 评估探针 (PostProcess eval_mode=True 官方路径).

- B ckpt (unk_head 存在): unk 通道 = pred_unk.sigmoid(), 由 eval_threshold gate
- A ckpt (无 unk_head):   unk 通道 = legacy residual 1 - sum(known softmax)
同一段代码自动覆盖两种模型, 保证 A/B 同口径 (相同数据子集/阈值/Top-K=100).

每个 eval_threshold in {0.0(base), 0.05, 0.1, 0.2, 0.3, 0.5, 0.7} 记录:
  U_R50       unknown 检测(label 80)对 unknown GT 的贪心召回 (IoU>=0.5)
  AOSA        known 检测与 unknown GT IoU>=0.5 的检测数
  CK_det/CK_TP/CK_Pproxy  CK 侧检测数/命中/精度代理

用法: python unk_head_eval_probe.py <N_IMG> --ckpt <path>
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

DATA_ROOT = "/icislab/volume1/wyy/O1O-main/data/OWOD"
TEST_SET = "owod_all_task_test_nodup"
ANN_DIR = os.path.join(DATA_ROOT, "Annotations")
THRS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n_img", type=int, default=300)
    ap.add_argument("--ckpt", default="/dev/shm/unkhead_out/task2_train/checkpoint.pth")
    args_cli = ap.parse_args()

    ck = torch.load(args_cli.ckpt, map_location="cpu")
    args = ck["args"]
    has_unk_head = getattr(args, "unk_head_w", 0.0) > 0
    print("ckpt:", args_cli.ckpt, "epoch:", ck.get("epoch"),
          "unk_head_w:", getattr(args, "unk_head_w", None), flush=True)
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
    missing, unexpected = model.load_state_dict(ck["model"], strict=True)
    print("loaded strict. missing=%d unexpected=%d model_has_unk_head=%s"
          % (len(missing), len(unexpected), getattr(model, "unk_head", None) is not None), flush=True)

    unk_idx = args.num_classes - 1          # 80
    PREV = int(args.PREV_INTRODUCED_CLS)
    CUR = int(args.CUR_INTRODUCED_CLS)
    CK_LO, CK_HI = PREV, PREV + CUR         # [20, 40)

    ds = OWDetection(args, DATA_ROOT, image_set=TEST_SET, dataset=args.dataset,
                     transforms=make_coco_transforms(TEST_SET))

    # XML 快速预扫描: 类名不在前 PREV+CUR 个已知类 -> unknown, 与官方映射口径一致
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

    def greedy_match(det_boxes, det_scores, gt_boxes):
        """按 score 降序贪心; 返回各 det 的 GT 索引(-1 无)与 GT 覆盖 bool."""
        order = torch.argsort(det_scores, descending=True)
        dev = det_boxes.device
        gt_used = torch.zeros(len(gt_boxes), dtype=torch.bool, device=dev)
        det2gt = -torch.ones(len(det_boxes), dtype=torch.int64, device=dev)
        for d in order.tolist():
            if len(gt_boxes) == 0:
                break
            ious = tvops.box_iou(det_boxes[d:d + 1], gt_boxes)[0]   # [G]
            ious = torch.where(gt_used, torch.zeros_like(ious), ious)
            v, g = ious.max(0)
            if v.item() >= 0.5:
                det2gt[d] = g.item()
                gt_used[g] = True
        return det2gt, gt_used

    scan = {t: {"u_num": 0, "u_hit": 0, "aosa": 0, "ck_det": 0, "ck_tp": 0,
                "n_det80": 0, "n_det_known": 0}
            for t in THRS}
    n_unk_gt = 0
    n_ck_gt = 0
    n_img_done = 0

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
                size = target["orig_size"].unsqueeze(0).to(device)

                # GT: 归一化 cxcywh -> 绝对像素 xyxy (与 PostProcess 输出同尺度)
                oh, ow = target["orig_size"].tolist()
                scale = torch.tensor([ow, oh, ow, oh], device=device)
                gt_xyxy = box_cxcywh_to_xyxy(tgt["boxes"]) * scale
                unk_mask = (tgt["labels"] == unk_idx)
                ck_mask = (tgt["labels"] >= CK_LO) & (tgt["labels"] < CK_HI)
                n_unk_gt += int(unk_mask.sum().item())
                n_ck_gt += int(ck_mask.sum().item())
                unk_gt = gt_xyxy[unk_mask]
                ck_gt = gt_xyxy[ck_mask]

                for thr in THRS:
                    results = postprocessors["bbox"](out, size, eval_mode=True,
                                                     eval_threshold=thr)
                    res = results[0]
                    sc = res["scores"].to(device)
                    lb = res["labels"].to(device)
                    bx = res["boxes"].to(device)
                    r = scan[thr]
                    m_u = (lb == unk_idx)
                    m_k = (lb != unk_idx)
                    r["n_det80"] += int(m_u.sum().item())
                    r["n_det_known"] += int(m_k.sum().item())
                    if m_u.any() and len(unk_gt) > 0:
                        _, cov = greedy_match(bx[m_u], sc[m_u], unk_gt)
                        r["u_hit"] += int(cov.sum().item())
                    if m_k.any() and len(unk_gt) > 0:
                        d2g, _ = greedy_match(bx[m_k], sc[m_k], unk_gt)
                        r["aosa"] += int((d2g >= 0).sum().item())
                    m_ck = m_k & (lb >= CK_LO) & (lb < CK_HI)
                    r["ck_det"] += int(m_ck.sum().item())
                    if m_ck.any() and len(ck_gt) > 0:
                        d2g, _ = greedy_match(bx[m_ck], sc[m_ck], ck_gt)
                        r["ck_tp"] += int((d2g >= 0).sum().item())
                n_img_done += 1
                if n_img_done % 50 == 0:
                    print("images done:", n_img_done, flush=True)
            except Exception as e:
                import traceback
                print("skip img", i, "err", repr(e), flush=True)
                traceback.print_exc()
                continue

    out = {
        "ckpt": args_cli.ckpt,
        "ckpt_epoch": ck.get("epoch"),
        "model_has_unk_head": has_unk_head,
        "n_images_done": n_img_done,
        "n_unk_gt": n_unk_gt,
        "n_ck_gt": n_ck_gt,
        "eval_mode_true_scan": [],
    }
    for t in THRS:
        r = scan[t]
        out["eval_mode_true_scan"].append({
            "eval_threshold": t,
            "U_R50": round(r["u_hit"] / max(n_unk_gt, 1), 4),
            "AOSA": r["aosa"],
            "n_det80": r["n_det80"],
            "n_det_known": r["n_det_known"],
            "CK_det": r["ck_det"],
            "CK_TP": r["ck_tp"],
            "CK_R50_proxy": round(r["ck_tp"] / max(n_ck_gt, 1), 4),
            "CK_P_proxy": round(r["ck_tp"] / max(r["ck_det"], 1), 4),
        })
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
