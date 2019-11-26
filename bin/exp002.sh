#!/usr/bin/env bash
exp_id=002

model=config${exp_id}
config=config/${model}.py

python -m exps.exp${exp_id} ${config} --fold 0