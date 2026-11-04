#!/bin/bash
export PYTHONPATH=/dev/shm/unkhead/proj
export CUDA_VISIBLE_DEVICES=3,4
mapfile -t args < /dev/shm/unkhead/relaunch_cmd.txt
cd /dev/shm/unkhead/proj
exec "${args[@]}"
