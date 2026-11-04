#!/bin/bash
# task2-4 全流程启动器：GPU3/4 双卡 DDP，从 task2_train 续训，--chain 自动衔接 ft/task3/task4
cd /dev/shm/unkhead
PYTHONNOUSERSITE=1 nohup /icislab/volume2/wyy/conda/envs/rfdetr/bin/python -u /dev/shm/unkhead/fmp/run_fmp.py \
  --stage 1 \
  --gpus 3,4 \
  --num-gpus 2 \
  --chain \
  --out-root /dev/shm/unkhead_out \
  --lr-scale 0.25 \
  --fmp-on 1 \
  --fmp-project-on 1 \
  --code-dir /dev/shm/unkhead/fmp \
  > /dev/shm/unkhead/task2to4_chain.log 2>&1 &
echo "launched PID=$!"
