#!/bin/bash

python $PRJ_HOME/src/train.py \
    --data_dir $PRJ_HOME/data \
    --transformer_embedding bert-base-cased \
    --finetune \
    --lr 1e-5
    --max_epochs 10
