import io
path = r"C:/Users/xc/WorkBuddy/2026-11-02-20-40-12/.workbuddy/memory/2026-11-03.md"
block = r"""

### B run（独立 unk head FMP）正式启动（2026-11-03 12:57）
- 用户指令：kill GPU3/4 的 A run，跑独立 unk head 的 FMP task2_train + task2_ft，然后分析 CK_AP50 下降原因。
- 已 kill A run（run_fmp 1052419 + rank 1053679/1053680），A 的 log.txt 备份为 log.txt.A_ref（ep38 参照）。
- B run 启动：`run_fmp.py --stage 1 --gpus 3,4 --num-gpus 2 --lr-scale 0.25 --out-root /dev/shm/unkhead_out --chain`，PID 4040634/4040639，GPU3/4 DDP。
- 启动验证通过：
  - resume 正确：start_epoch→30，从 task1 ckpt0029 续。
  - [FOTG] old_model loaded=616 missing=2 unexpected=0，missing=['unk_head.weight','unk_head.bias']（新头正确随机初始化）。
  - loss_unk 已接入训练（11 次出现），unk_n_pos≈36-37（伪框真成 unk 正样本），loss_unk 1.85→1.61 下降。
  - 训练正常（loss 33.8→21.4，eta~1.5h/epoch，共10 epoch）。
- --chain 会串 task2_train(ep30-40) → task2_ft(stage2)，resume 路径经 out_root 重写正确衔接（/dev/shm/fmp_chain/ → /dev/shm/unkhead_out/）。
- B 环境 /dev/shm/unkhead/ 完整（proj 编辑版 detector 16 处 unk_head + main 有 --unk_head_w 与 optimizer remap + run_fmp.py=run_fmp_b）。
EOF
"""
with io.open(path, "a", encoding="utf-8") as f:
    f.write(block)
print("MEMORY_APPENDED")
