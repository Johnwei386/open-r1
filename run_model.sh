#!/bin/sh

MODEL=/home/john/Datums/Researching/Qwen2-0.5B-Instruct
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,generation_parameters={max_new_tokens:32768}"
OUTPUT_DIR=data/evals/$MODEL

# AIME 2024
TASK=aime24
lighteval accelerate $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR
    large-scale