import io
path = r"C:/Users/xc/WorkBuddy/2026-11-02-20-40-12/.workbuddy/memory/2026-11-03.md"
block = r"""

### Baseline(O1O-main no-FMP) vs FMP(正在跑) 真实对比（2026-11-03 12:50）
- 用户纠正术语：O1O-main 才是 baseline（原始 OWOD，无 FMP）；之前说的"A run"=FMP方法（残差式unk）；B run=FMP+独立unk head。
- Baseline = O1O-main/exps/o1o_mowod_task2_train（ep39, ckpt0039, 492MB）；与 FMP 同 resume o1o_mowod_task1_train/checkpoint0029，同 exemplar replay（已确认 baseline args: exemplar_replay_selection=True, max_len=1743, prev=learned_owod_t1_ft.txt）。
- FMP = fmp_t2/task2_train（ep38）。两者 test_metrics 同口径（OWEvaluator）。

| 指标 | Baseline(ep39) | FMP(ep38) | Delta |
|------|------|------|------|
| U_R50 | 55.42 | 38.50 | -16.92 (FMP更差) |
| U_AP50 | 0.526 | 0.298 | -0.228 |
| U_P50 | 0.665 | 0.730 | +0.066 |
| K_AP50 | 19.87 | 39.94 | +20.08 |
| CK_AP50 | 39.74 | 35.84 | -3.89 |
| PK_AP50 | 0.00 | 44.04 | +44.04 |
| WI | 0.0169 | 0.0235 | +0.0066(更差) |
| AOSA | 109 | 8157 | +8048(差75x) |

- FMP 的得：旧类保留 PK 0→44（FMP 核心目的）；K_AP50 升因 baseline 的 K 含被忘的 task1=0。
- FMP 的失：unknown 发现崩塌（U_R50 55→38、AOSA 109→8157、WI 升）。
- 新类学习 CK_AP50：baseline 39.74 > FMP 35.84（baseline 塑性更高）。
- anomaly：baseline PK=0 尽管启了 exemplar replay——或是 task1 真灾难性遗忘（FMP 流形保护给 PK=44），或 eval 口径差异。最干净对比轴是 unknown 发现（baseline 明显胜），FMP 在此轴上全面落后于 baseline。
- B run 目标再确认：在保 FMP 的 PK≈44 同时，把 U_R50 从 38.5 拉回 baseline 的 ~55（独立 unk head 设计初衷）。
"""
with io.open(path, "a", encoding="utf-8") as f:
    f.write(block)
print("MEMORY_APPENDED")
