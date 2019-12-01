#!/usr/bin/env bash
exp_id=003

model=config${exp_id}
config=config/${model}.py

export CUDA_VISIBLE_DEVICES="0,1,2,3"

python -m exps.exp${exp_id} ${config} --fold 0