#!/bin/bash

python $PRJ_HOME/src/train.py \
    --train_dir $PRJ_HOME/outputs/flair_bilstm_crf \
    --data_dir $PRJ_HOME/data \
    --word_embedding glove \
    --use_rnn \
    --use_crf \
    --lr 0.1 \
    --max_epochs 100
