#!/usr/bin/env python
# -*- coding:utf-8 -*-

''' 使用GRPOTrainer进行训练 '''

from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer

# dataset = load_dataset("trl-lib/tldr", split="train")
dataset = load_dataset('/home/john/Datums/Researching/Datasets/trl-lib_tldr/data', split='train')

# Define the reward function, which rewards completions that are close to 20 characters
def reward_len(completions, **kwargs):
    return [-abs(20 - len(completion)) for completion in completions]

training_args = GRPOConfig(output_dir="Qwen2-0.5B-GRPO",
                           per_device_train_batch_size=4,  # 每个GPU上的批次大小
                           gradient_accumulation_steps=2,  # 此参数应与accelerate的配置文件中对应的参数相同,此处默认为1
                           fp16=True,                      # 启用混合精度训练
                           logging_steps=10,
                          )
trainer = GRPOTrainer(
    # model="Qwen/Qwen2-0.5B-Instruct",
    model="/home/john/Datums/Researching/Qwen2-0.5B-Instruct",
    reward_funcs=reward_len,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()

