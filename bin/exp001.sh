#!/usr/bin/env bash
exp_id=001

model=config${exp_id}
config=config/${model}.py

python -m exps.exp${exp_id} ${config}