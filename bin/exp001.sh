#!/usr/bin/env bash
exp_id=001

model=model${exp_id}
config=conf/${model}.py

python -m exps.exp${exp_id} ${config}