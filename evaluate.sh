#!/bin/bash

FOLDER='/media/disk0/mcescobar/bcv/Smart_Pooling_PLOSONE/' #Full path to where you have the github repository
OUTPUT='experiments/best_model/'
mkdir -p $OUTPUT
BEST_TEST=$FOLDER'best_model/all_train_model/GBM_model_python_1591466176897_1'
BEST_VAL=$FOLDER'best_model/train_model/GBM_grid__1_AutoML_20200606_175626_model_198'
PORT=54328
POOLSIZE=10

python -W ignore evaluate.py --savegraph --output-dir $OUTPUT  --eval --path-to-best $BEST_TEST --path-to-val $BEST_VAL --port $PORT --poolsize $POOLSIZE >> $OUTPUT'Info.log'
