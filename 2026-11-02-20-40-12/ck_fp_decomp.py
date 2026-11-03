"""
CK FP 三类分解探针 (GPU 子集, 方向性; 绕过 OWDetection 有 bug 的 parse_voc_xml).

背景: 本机 OWDetection.load_instances / parse_voc_xml 会静默丢弃非 VOC 类
(含 CK 20..39 与 unknown >=40) 的 GT 实例, 导致无法对 CK 预测做基于 GT 的
FP 分解, 也与 A-run log 中 CK_AP50=35.84 / AOSA=8157 所依赖的 GT 不一致.
因此本脚本直接解析 Annotations 下原始 XML 得到完整 GT(含所有类), 与官方
指标口径对齐; 模型预测走官方 evaluate 使用的 eval_mode=False (sigmoid) 路径.

用法: python ck_fp_decomp.py <N_IMG>
"""
import sys, os, json, argparse
sys.path.insert(0, "/dev/shm/unkhead/fmp")
sys.path.insert(0, "/dev/shm/unkhead/proj")

import torch
import torchvision.ops as tvops
import xml.etree.ElementTree as ET
from util.misc import nested_tensor_from_tensor_list
from models import build_model
from datasets.torchvision_datasets.open_world_o1o import (
    OWDetection, VOC_CLASS_NAMES_COCOFIED, BASE_VOC_CLASS_NAMES)
from datasets.coco import make_coco_transforms

CKPT = "/dev/shm/fmp_t2/task2_train/checkpoint.pth"
DATA_ROOT = "/icislab/volume1/wyy/O1O-main/data/OWOD"
TEST_SET = "owod_all_task_test_nodup"
ANN_DIR = os.path.join(DATA_ROOT, "Annotations")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n_img", type=int, default=2)
    args_cli = ap.parse_args()

    ck = torch.load(CKPT, map_location="cpu")
    args = ck["args"]
    # 强制 CUDA / 非分布式 / 残差 unk 路径 (ckpt 无 unk_head)
    # 多尺度 deformable attention 是 CUDA-only, 无法在 CPU 跑.
    args.device = "cuda"
    args.num_gpus = 0
    args.distributed = False
    args.unk_head_w = 0.0
    args.output_dir = "/dev/shm"
    args.data_root = DATA_ROOT
    args.test_set = TEST_SET
    args.eval = True
    args.eval_threshold = 0.0

    device = torch.device("cuda")

    model, criterion, postprocessors, _ = build_model(args, mode=args.model_type)
    model.to(device)
    model.eval()
    sd = ck["model"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("loaded. missing=%d unexpected=%d" % (len(missing), len(unexpected)), flush=True)

    ds = OWDetection(args, DATA_ROOT, image_set=TEST_SET, dataset=args.dataset,
                     transforms=make_coco_transforms(TEST_SET))

    CLASS_NAMES = ds.CLASS_NAMES
    UNK = len(CLASS_NAMES) - 1
    PREV = int(args.PREV_INTRODUCED_CLS)
    CUR = int(args.CUR_INTRODUCED_CLS)
    CK_LO, CK_HI = PREV, PREV + CUR  # [20, 40)

    def name_to_idx(nm):
        if nm in VOC_CLASS_NAMES_COCOFIED:
            nm = BASE_VOC_CLASS_NAMES[VOC_CLASS_NAMES_COCOFIED.index(nm)]
        try:
            return CLASS_NAMES.index(nm)
        except ValueError:
            return UNK  # 不在已知 81 类表中的名字 -> 视作 unknown

    def parse_gt(str_imgid):
        """直接解析 XML, 返回 (boxes_abs_xyxy [N,4], labels [N]) 绝对像素坐标."""
        path = os.path.join(ANN_DIR, str_imgid + ".xml")
        if not os.path.exists(path):
            return [], []
        root = ET.parse(path).getroot()
        boxes, labels = [], []
        for obj in root.findall("object"):
            nm = obj.find("name").text
            cid = name_to_idx(nm)
            if cid >= (PREV + CUR) or nm == "unknown":  # 非 seen 类 -> unknown
                cid = UNK
            bb = obj.find("bndbox")
            if bb is None:
                continue
            xmin = float(bb.find("xmin").text) - 1.0
            ymin = float(bb.find("ymin").text) - 1.0
            xmax = float(bb.find("xmax").text) - 1.0
            ymax = float(bb.find("ymax").text) - 1.0
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(cid)
        return boxes, labels

    cat_unknown = cat_bg = cat_miscls = cat_tp = 0
    aosa_total = aosa_ck = aosa_old = aosa_future = 0
    n_ck_pred = 0
    n_img_done = 0
    n_ck_gt = 0
    n_unk_gt = 0

    # 预扫描: 找出真正含有 CK(20..39) 或 unknown(>=40) GT 的图像索引.
    # 前若干图像是纯 VOC(0..19) 图, 不含 CK/unknown GT, 直接跑会得到全 background 假分解.
    valid_idx = []
    for i in range(len(ds.str_imgids)):
        boxes, labels = parse_gt(ds.str_imgids[i])
        if any((CK_LO <= c < CK_HI) or (c == UNK) for c in labels):
            valid_idx.append(i)
    print("images with CK/unk GT:", len(valid_idx), "of", len(ds), flush=True)
    torch.manual_seed(0)
    perm = torch.randperm(len(valid_idx)).tolist()
    sel = [valid_idx[j] for j in perm[:args_cli.n_img]]
    print("selected", len(sel), "images for decomposition", flush=True)

    for i in sel:
        try:
            img, target = ds[i]
            if not isinstance(img, torch.Tensor):
                continue
            str_imgid = ds.str_imgids[i]
            gt_boxes_abs, gt_labels_list = parse_gt(str_imgid)
            if len(gt_boxes_abs) > 0:
                gt_boxes = torch.tensor(gt_boxes_abs, dtype=torch.float32, device=device)
                gt_labels = torch.tensor(gt_labels_list, dtype=torch.int64, device=device)
            else:
                gt_boxes = torch.zeros((0, 4), dtype=torch.float32, device=device)
                gt_labels = torch.zeros((0,), dtype=torch.int64, device=device)
            n_ck_gt += int(((gt_labels >= CK_LO) & (gt_labels < CK_HI)).sum().item())
            n_unk_gt += int((gt_labels == UNK).sum().item())

            samples = nested_tensor_from_tensor_list([img.to(device)])
            with torch.no_grad():
                out, _ = model(samples, dn_args=args.num_patterns)
                size = target['orig_size'].unsqueeze(0).to(device)
                # 官方训练期 evaluate 默认 eval_mode=False (sigmoid) -> 与 log 中
                # CK_AP50 / AOSA 指标对应的预测口径一致.
                results = postprocessors['bbox'](out, size, eval_mode=False, eval_threshold=0.0)
            res = results[0]
            scores = res['scores']
            labels = res['labels']
            boxes = res['boxes'].to(device)

            for k in range(len(scores)):
                L = int(labels[k].item())
                if CK_LO <= L < CK_HI:
                    n_ck_pred += 1
                    pb = boxes[k:k + 1]
                    if len(gt_boxes) > 0:
                        iou = tvops.box_iou(pb, gt_boxes)[0]
                        best, bi = iou.max(0)
                    else:
                        best = torch.tensor([-1.0])
                        bi = None
                    if best.item() >= 0.5:
                        g = int(gt_labels[bi].item())
                        if g == UNK or g >= CK_HI:
                            cat_unknown += 1
                        elif CK_LO <= g < CK_HI:
                            if g == L:
                                cat_tp += 1
                            else:
                                cat_miscls += 1
                        else:  # 0..PREV 旧已知类误分类
                            cat_miscls += 1
                    else:
                        cat_bg += 1

            # AOSA 拆解 (对齐官方 voc_eval, 按 detection 计):
            # 对每个 known 类预测(0..79), 若其与任一 unknown GT IoU>=0.5 则计入,
            # 并按该预测所属类别归入 CK / 旧已知 / 未来类.
            unk_mask = (gt_labels == UNK)
            if unk_mask.any() and len(boxes) > 0:
                unk_gt = gt_boxes[unk_mask]
                # box_iou 直接返回 [P, U] 矩阵(非元组), 此前误加 [0] 取成第一行导致越界.
                iou_all = tvops.box_iou(boxes, unk_gt)  # [P, U]
                for p in range(len(boxes)):
                    pl = int(labels[p].item())
                    if 0 <= pl < UNK:  # 仅计 known 类预测
                        mi = iou_all[p].max().item()
                        if mi >= 0.5:
                            aosa_total += 1
                            if CK_LO <= pl < CK_HI:
                                aosa_ck += 1
                            elif 0 <= pl < CK_LO:
                                aosa_old += 1
                            else:
                                aosa_future += 1
            n_img_done += 1
        except Exception as e:
            import traceback
            print("skip img", i, "err", repr(e), flush=True)
            traceback.print_exc()
            continue

    total_fp = cat_unknown + cat_bg + cat_miscls
    out = {
        "n_images_done": n_img_done,
        "n_ck_predictions": n_ck_pred,
        "n_ck_gt_on_subset": n_ck_gt,
        "n_unk_gt_on_subset": n_unk_gt,
        "CK_FP_unknown_origin": cat_unknown,
        "CK_FP_background_origin": cat_bg,
        "CK_FP_known_misclass": cat_miscls,
        "CK_TP": cat_tp,
        "CK_FP_total": total_fp,
        "CK_FP_pct": {
            "unknown": round(cat_unknown / max(total_fp, 1), 4),
            "background": round(cat_bg / max(total_fp, 1), 4),
            "known_misclass": round(cat_miscls / max(total_fp, 1), 4),
        },
        "AOSA_total_on_subset": aosa_total,
        "AOSA_pred_as_CK": aosa_ck,
        "AOSA_pred_as_old_known": aosa_old,
        "AOSA_pred_as_future_known": aosa_future,
        "AOSA_ck_fraction": round(aosa_ck / max(aosa_total, 1), 4),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
