"""
DN query 错位验证探针 (在未修复的当前服务器代码上跑).

验证用户假设: loss_unk 的 matcher indices (normal query 坐标系) 直接写进
未裁剪 pred_unk (带 DN 前缀) 造成错位.

输出:
  1. unk_pos_total / unk_pos_dn / unk_pos_regular (用户点名的三个数)
     - unk_pos_dn: 正样本索引 < pad_size -> 错位落到 DN query 区
  2. pred_unk (未裁剪) 分数分布: DN 区 vs normal 区
     - 当前 ckpt 已被错位监督训 1 epoch, 预期 DN 区出现高分、normal 区全低
用法: python dn_misalign_probe.py
"""
import sys, os
sys.path.insert(0, "/dev/shm/unkhead/fmp")
sys.path.insert(0, "/dev/shm/unkhead/proj")

import torch
from util.misc import nested_tensor_from_tensor_list
from models import build_model
from datasets.torchvision_datasets.open_world_o1o import OWDetection
from datasets.coco import make_coco_transforms

CKPT = "/dev/shm/unkhead_out/task2_train/checkpoint.pth"
DATA_ROOT = "/icislab/volume1/wyy/O1O-main/data/OWOD"

device = torch.device("cuda")
ck = torch.load(CKPT, map_location="cpu")
args = ck["args"]
args.device = "cuda"
args.num_gpus = 0
args.distributed = False
args.output_dir = "/dev/shm"
args.data_root = DATA_ROOT
args.eval = False
print("epoch:", ck.get("epoch"), "unk_head_w:", getattr(args, "unk_head_w", None),
      "scalar:", args.scalar, "label_noise:", args.label_noise_scale,
      "box_noise:", args.box_noise_scale, flush=True)

model, criterion, postprocessors, _ = build_model(args, mode=args.model_type)
model.to(device)
criterion.to(device)
model.load_state_dict(ck["model"], strict=True)
model.train()  # DN 仅在 training 模式构造

matcher = criterion.matcher
unk_idx = args.num_classes - 1

ds = OWDetection(args, DATA_ROOT, image_set="owod_t2_train", dataset=args.dataset,
                 pseudo_path=getattr(args, "pseudo_path", ""),
                 transforms=make_coco_transforms("owod_t2_train"))
print("train set size:", len(ds), flush=True)

stats = {"unk_pos_total": 0, "unk_pos_dn": 0, "unk_pos_regular": 0,
         "known_pos_dn": 0, "known_pos_regular": 0}
dn_scores_all, reg_scores_all = [], []
pads, q_totals, q_norms = [], [], []

torch.manual_seed(0)
# 预扫描: 选 targets 含 unknown(label==80) 的训练图 (pseudo-unknown 注入后)
unk_img_idx = []
for j in range(min(4000, len(ds))):
    t = ds[j][1]
    if (t["labels"] == unk_idx).any():
        unk_img_idx.append(j)
    if len(unk_img_idx) >= 8:
        break
print("unk-bearing train images found:", len(unk_img_idx), flush=True)

with torch.no_grad():
    n_done = 0
    for bi in range(0, len(unk_img_idx) - 1, 2):  # bs=2
        samples_list, targets = [], []
        for j in unk_img_idx[bi:bi + 2]:
            img, tgt = ds[j]
            if not isinstance(img, torch.Tensor):
                continue
            samples_list.append(img)
            tgt_d = {}
            for k, v in tgt.items():
                tgt_d[k] = v.to(device) if isinstance(v, torch.Tensor) else v
            targets.append(tgt_d)
        if len(samples_list) < 2:
            continue
        samples = nested_tensor_from_tensor_list([s.to(device) for s in samples_list])
        dn_args = (targets, args.scalar, args.label_noise_scale,
                   args.box_noise_scale, args.num_patterns)
        outputs, mask_dict = model(samples, dn_args=dn_args)
        pad = int(mask_dict["pad_size"]) if mask_dict else 0
        p = outputs["pred_unk"].squeeze(-1).sigmoid()          # [B, Q_unk]
        fixed = (p.shape[1] == outputs["pred_logits"].shape[1])
        if fixed:
            # 修复后: pred_unk 已与 pred_logits 同长(DN 已裁剪), 正样本索引天然对齐
            print("FIXED-CODE detected: pred_unk Q=%d == pred_logits Q=%d, pad=%d"
                  % (p.shape[1], outputs["pred_logits"].shape[1], pad), flush=True)
            dn_scores_all += []
            reg_scores_all += p.flatten().tolist()
            pads.append(pad)
            q_totals.append(p.shape[1])
            q_norms.append(outputs["pred_logits"].shape[1])
        else:
            dn_scores_all += p[:, :pad].flatten().tolist()
            reg_scores_all += p[:, pad:].flatten().tolist()
            pads.append(pad)
            q_totals.append(p.shape[1])
            q_norms.append(outputs["pred_logits"].shape[1])

        # matcher 在 DN 裁剪后的 outputs 上 (与 criterion 训练路径一致)
        indices = matcher({"pred_logits": outputs["pred_logits"],
                           "pred_boxes": outputs["pred_boxes"]}, targets)
        for i, (pi, ti) in enumerate(indices):
            tl = targets[i]["labels"].to(device)
            ml = tl[ti.to(device)]
            pos = pi.to(device)[ml == unk_idx]
            knw = pi.to(device)[ml != unk_idx]
            stats["unk_pos_total"] += len(pos)
            stats["unk_pos_dn"] += int((pos < pad).sum().item())
            stats["unk_pos_regular"] += int((pos >= pad).sum().item())
            stats["known_pos_dn"] += int((knw < pad).sum().item())
            stats["known_pos_regular"] += int((knw >= pad).sum().item())
        n_done += 1

def st(v):
    if not v:
        return {"n": 0}
    t = torch.tensor(v)
    return {"n": len(v), "mean": round(t.mean().item(), 4),
            "median": round(t.median().item(), 4),
            "p99": round(torch.quantile(t, 0.99).item(), 4),
            "max": round(t.max().item(), 4),
            "frac>0.5": round((t > 0.5).float().mean().item(), 6)}

import json
out = {
    "n_batches": n_done,
    "pad_size_list": pads,
    "Q_total": q_totals[0] if q_totals else None,
    "Q_normal": q_norms[0] if q_norms else None,
    "pos_stats": stats,
    "pred_unk_scores_DN_region": st(dn_scores_all),
    "pred_unk_scores_normal_region": st(reg_scores_all),
}
print(json.dumps(out, indent=2))
