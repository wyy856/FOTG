#!/bin/bash
export FG_HEAD_W=1.0
export FG_LR_MULT=100
export FG_FOCAL_GAMMA=2.0
export FG_FOCAL_ALPHA=0.9
export PYTHONNOUSERSITE=1
cd /dev/shm/unkhead/fmp || exit 1
nohup /icislab/volume2/wyy/conda/envs/rfdetr/bin/python -u /dev/shm/unkhead/fmp/run_fmp.py \
  --stage 1 --gpus 1,2,3,4 --num-gpus 4 --chain --batch-size 1 --out-root /dev/shm/unkhead_out_fg \
  --lr-scale 0.25 --fmp-on 1 --fmp-project-on 1 --code-dir /dev/shm/unkhead/fmp \
  > /dev/shm/unkhead/fg_chain.log 2>&1 &
echo "STARTED_PID=$!"
