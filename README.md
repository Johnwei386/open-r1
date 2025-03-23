# Open R1

*A fully open reproduction of DeepSeek-R1. This repo is a work in progress, let's build it together!*

**Table of Contents**  
- [Open R1](#open-r1)
  - [Overview](#overview)
    - [Plan of attack](#plan-of-attack)
  - [Installation](#installation)
  - [Training models](#training-models)
    - [SFT](#sft)
    - [GRPO](#grpo)
      - [👨‍💻 Training with a code interpreter](#-training-with-a-code-interpreter)
    - [Launching jobs on a Slurm cluster](#launching-jobs-on-a-slurm-cluster)
  - [Evaluating models](#evaluating-models)
  - [Reproducing Deepseek's evaluation results](#reproducing-deepseeks-evaluation-results)
    - [AIME 2024](#aime-2024)
    - [MATH-500](#math-500)
    - [GPQA Diamond](#gpqa-diamond)
    - [LiveCodeBench](#livecodebench)
  - [Data generation](#data-generation)
    - [Generate data from a smol distilled R1 model](#generate-data-from-a-smol-distilled-r1-model)
    - [Generate data from DeepSeek-R1](#generate-data-from-deepseek-r1)
  - [Contributing](#contributing)
    - [训练grpo](#训练grpo)

## Overview

The goal of this repo is to build the missing pieces of the R1 pipeline such that everybody can reproduce and build on top of it. The project is simple by design and mostly consists of:


- `src/open_r1`: contains the scripts to train and evaluate models as well as generate synthetic data:
    - `grpo.py`: trains a model with GRPO on a given dataset.
    - `sft.py`: performs a simple SFT of a model on a dataset.
    - `evaluate.py`: evaluates a model on the R1 benchmarks.
    - `generate.py`: generates synthetic data from a model using [Distilabel](https://github.com/argilla-io/distilabel).
- `Makefile`: contains easy-to-run commands for each step in the R1 pipeline leveraging the scripts above.

### Plan of attack

We will use the DeepSeek-R1 [tech report](https://github.com/deepseek-ai/DeepSeek-R1) as a guide, which can roughly be broken down into three main steps:

* Step 1: replicate the R1-Distill models by distilling a high-quality corpus from DeepSeek-R1.
* Step 2: replicate the pure RL pipeline that DeepSeek used to create R1-Zero. This will likely involve curating new, large-scale datasets for math, reasoning, and code.
* Step 3: show we can go from base model to RL-tuned via multi-stage training.

<center>
    <img src="assets/plan-of-attack.png" width="500">
</center>


## Installation

> [!CAUTION]
> Libraries rely on CUDA 12.4. If you see errors related to segmentation faults, double check the version your system is running with `nvcc --version`.

To run the code in this project, first, create a Python virtual environment using e.g. `uv`.
To install `uv`, follow the [UV Installation Guide](https://docs.astral.sh/uv/getting-started/installation/).


```shell
uv venv openr1 --python 3.11 && source openr1/bin/activate && uv pip install --upgrade pip
```

> [!TIP]
> For Hugging Face cluster users, add `export UV_LINK_MODE=copy` to your `.bashrc` to suppress cache warnings from `uv`

Next, install vLLM and FlashAttention:

```shell
uv pip install vllm==0.7.2
uv pip install setuptools && uv pip install flash-attn --no-build-isolation
```

This will also install PyTorch `v2.5.1` and it is **very important** to use this version since the vLLM binaries are compiled for it. You can then install the remaining dependencies for your specific use case via `pip install -e .[LIST OF MODES]`. For most contributors, we recommend:

```shell
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e ".[dev]"
```

```bash
# 以可编辑模式安装,库不会安装到site-package中,会引用到当前目录,同时会安装所有依赖文件
pip install -e .
```

Next, log into your Hugging Face and Weights and Biases accounts as follows:

```shell
huggingface-cli login
wandb login
```

Finally, check whether your system has Git LFS installed so that you can load and push models/datasets to the Hugging Face Hub:

```shell
git-lfs --version
```

If it isn't installed, run:

```shell
sudo apt-get install git-lfs
```

## Training models

We support training models with either DDP or DeepSpeed (ZeRO-2 and ZeRO-3). For example, to run SFT on a dataset distilled from DeepSeek-R1 with reasoning traces such as [open-r1/OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k), run:

```shell
# Train via command line
accelerate launch --config_file=recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --dataset_name open-r1/OpenR1-Math-220k \
    --learning_rate 1.0e-5 \
    --num_train_epochs 1 \
    --packing \
    --max_seq_length 16384 \
    --per_device_train_batch_size 16 \
    --gradient_checkpointing \
    --bf16 \
    --output_dir data/Qwen2.5-1.5B-Open-R1-Distill

# Train via YAML config
accelerate launch --config_file recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py \
    --config recipes/Qwen2.5-1.5B-Instruct/sft/config_demo.yaml
```

Currently, the following tasks are supported:

* Supervised Fine-Tuning `sft`
* Group Relative Policy Optimization `grpo`

> [!TIP]
> If you scale up/down the number of GPUs, we recommend also scaling up the per-device batch size or number of gradient accumulation steps to keep the global batch size constant.

By default, these scripts will push each model to your Hugging Face Hub username, i.e. `{username}/{model_name}-{task}`. You can override the parameters in each YAML config by appending them to the command as follows: 

```shell
# Change batch size, number of epochs etc
accelerate launch --config_file recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py \
    --config recipes/Qwen2.5-1.5B-Instruct/sft/config_demo.yaml
    --per_device_train_batch_size=1 --num_train_epochs=5
```

If you also wish to override the Weights and Biases default settings, you can do so as follows:

```shell
accelerate launch --config_file recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py \
    --config recipes/Qwen2.5-1.5B-Instruct/sft/config_demo.yaml
    --wandb_entity huggingface --wandb_project open-r1 --run_name Qwen2.5-1.5B-GRPO
```

> [!NOTE]
> The training commands below are configured for a node of 8 x H100s (80GB). For different hardware and topologies, you may need to tune the batch size and number of gradient accumulation steps.

### SFT

To run SFT on a dataset distilled from DeepSeek-R1 with reasoning traces such as [open-r1/OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k), run:

```shell
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero3.yaml \
    src/open_r1/sft.py \
    --config recipes/Qwen2.5-1.5B-Instruct/sft/config_demo.yaml
```

### GRPO

To train via the GRPO trainer, we use one GPU to run vLLM for faster generation and the remaining GPUs for training. For example, one a node with 8 GPUs, set `--num_processes` to override the default value in the `accelerate` configs:

```shell
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero2.yaml \
    --num_processes=7 src/open_r1/grpo.py \
    --config recipes/DeepSeek-R1-Distill-Qwen-1.5B/grpo/config_demo.yaml
```

> [!WARNING]
> The chat template used in the distilled DeepSeek models omits the contents of the reasoning block within the `<think>` and `</think>` tags. It also prefills the assistant response with `<think>` which interferes with the format reward function. To handle that, it is important to override the chat template as done in e.g.  [recipes/DeepSeek-R1-Distill-Qwen-1.5B/grpo/config_demo.yaml](./recipes/DeepSeek-R1-Distill-Qwen-1.5B/grpo/config_demo.yaml).


We provide a minimal reproducible experiment using GRPO for mathematical reasoning, referencing the approach from [SimpleRL-Reason](https://hkust-nlp.notion.site/simplerl-reason) which uses a 7B model trained on 8K examples. Running this on 8 H100 80G GPU takes about 3 hours:

```shell
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero2.yaml \
    --num_processes=7 src/open_r1/grpo.py \
    --config recipes/Qwen2.5-Math-7B/grpo/config_simple_rl.yaml
```

Our final [model](https://huggingface.co/Dongwei/Qwen-2.5-7B_Base_Math_smalllr), while using different learning rates, loss functions and reward structures, achieves 69.4% accuracy on MATH-500, demonstrating a 17%+ improvement over the base model.

#### 👨‍💻 Training with a code interpreter

We provide a `code` reward function for executing code generated by the policy during training. Currently, this reward function targets code contests like [Codeforces](https://codeforces.com), where solutions are executed against a set of test cases and the overall success rate is returned as the final reward. To ensure safe execution, we use [E2B](https://e2b.dev) sandboxes, which are fast and cheap to run. To use this reward function, first install the necessary dependencies:

```shell
uv pip install -e '.[code]'
```

Then create a `.env` file and place an API token from E2B within it:

```
E2B_API_KEY="e2b_xxx"
```

Then make sure your dataset contains a `verification_info` column with the following schema (adopted from PrimeIntellect's excellent [datasets](https://huggingface.co/collections/PrimeIntellect/synthetic-1-67a2c399cfdd6c9f7fae0c37) of verifiable problems):

```python
{
    "language": "python",
    "test_cases": [
        {
            "input": "4\n4\n0001\n1000\n0011\n0111\n3\n010\n101\n0\n2\n00000\n00001\n4\n01\n001\n0001\n00001\n",
            "output": "1\n3 \n-1\n0\n\n2\n1 2 \n",
            "type": "stdin_stdout",
        }
    ],
}
```

For example, to train a smol model on Python problems, run:

```shell
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero2.yaml \
    --num_processes=7 src/open_r1/grpo.py \
    --config recipes/Qwen2.5-1.5B-Instruct/grpo/config_demo_code.yaml
```

### Launching jobs on a Slurm cluster

If you have access to a Slurm cluster, we provide a `slurm/train.slurm` script that will automatically queue training jobs for you. Here's how you can use it:

```shell
sbatch --job-name=open_r1 --nodes=1 slurm/train.slurm {model_name} {task} {config_suffix} {accelerator}
```

Here `{model_name}` and `{task}` are defined as above, while `{config_suffix}` refers to the specific config and `{accelerator}` refers to the choice of 🤗 Accelerate config in `recipes/accelerate_configs`. If you wish to override the default config parameters, you can provide them by appending a space-separated string like `'--arg1=value1 --arg2=value2'`. Here's a concrete example to run SFT on 1 node of 8 GPUs:

```shell
# Launch on Slurm and override default hyperparameters
sbatch --job-name=open_r1 --nodes=1 slurm/train.slurm Qwen2.5-1.5B-Instruct sft demo zero3 '--per_device_train_batch_size=1 --num_train_epochs=5'
```

You can scale the number of nodes by increasing the `--nodes` flag.

> [!NOTE]
> The configuration in `slurm/train.slurm` is optimised for the Hugging Face Compute Cluster and may require tweaking to be adapted to your own compute nodes.

## Evaluating models

We use `lighteval` to evaluate models, with custom tasks defined in `src/open_r1/evaluate.py`. For models which fit on a single GPU, run:

```shell
MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
OUTPUT_DIR=data/evals/$MODEL

# AIME 2024
TASK=aime24
lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR

# MATH-500
TASK=math_500
lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR

# GPQA Diamond
TASK=gpqa:diamond
lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR

# LiveCodeBench
lighteval vllm $MODEL_ARGS "extended|lcb:codegeneration|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR 
```

> [!IMPORTANT]
> You must set `max_model_length=32768` in the `vllm` command to align with the `max_new_tokens` we define per eval. Without this, `lighteval` will throw an error.

To increase throughput across multiple GPUs, use _data parallel_ as follows:

```shell
NUM_GPUS=8
MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
TASK=aime24
OUTPUT_DIR=data/evals/$MODEL

lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR 
```

For large models which require sharding across GPUs, use _tensor parallel_ and run:

```shell
NUM_GPUS=8
MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,tensor_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
TASK=aime24
OUTPUT_DIR=data/evals/$MODEL

export VLLM_WORKER_MULTIPROC_METHOD=spawn
lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR 
```

You can also launch an evaluation with `make evaluate`, specifying the model, task, and optionally the parallelism technique and number of GPUs.

To evaluate on a single GPU:

```shell
make evaluate MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B TASK=aime24
```

To use Data Parallelism:

```shell
make evaluate MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B TASK=aime24 PARALLEL=data NUM_GPUS=8
```

To use Tensor Parallelism:

```shell
make evaluate MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B TASK=aime24 PARALLEL=tensor NUM_GPUS=8
```

## Reproducing Deepseek's evaluation results

> [!NOTE]
> The DeepSeek-R1 paper uses sampling with 64 responses per query to estimate `pass@1`. Below, we report the results from sampling 1 response per query, which likely explains the small 1-3σ discrepancies between our results and theirs.

### AIME 2024

We are able to reproduce Deepseek's reported results on the AIME 2024 benchmark within ~1-3 standard deviations:

| Model                         | AIME 2024 (🤗 LightEval) | AIME 2024 (DeepSeek Reported) |
|:------------------------------|:-----------------------:|:----------------------------:|
| DeepSeek-R1-Distill-Qwen-1.5B |          26.7           |             28.9             |
| DeepSeek-R1-Distill-Qwen-7B   |          56.6           |             55.5             |
| DeepSeek-R1-Distill-Qwen-14B  |          60.0           |             69.7             |
| DeepSeek-R1-Distill-Qwen-32B  |          73.2           |             72.6             |
| DeepSeek-R1-Distill-Llama-8B  |          43.3           |             50.4             |
| DeepSeek-R1-Distill-Llama-70B |          73.3           |             70.0             |

To reproduce these results use the following command:

```shell
NUM_GPUS=1 # Set to 8 for 32B and 70B models
MODEL=deepseek-ai/{model_name}
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,data_parallel_size=$NUM_GPUS,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
OUTPUT_DIR=data/evals/$MODEL

lighteval vllm $MODEL_ARGS "custom|aime24|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR
```

Alternatively, you can launch Slurm jobs as follows:

```shell
python scripts/run_benchmarks.py --model-id {model_id}  --benchmarks aime24
```

### MATH-500

We are able to reproduce Deepseek's reported results on the MATH-500 benchmark within ~1-3 standard deviations:

| Model                         | MATH-500 (🤗 LightEval) | MATH-500 (DeepSeek Reported) |
|:------------------------------|:-----------------------:|:----------------------------:|
| DeepSeek-R1-Distill-Qwen-1.5B |          84.6           |             83.9             |
| DeepSeek-R1-Distill-Qwen-7B   |          93.0           |             92.8             |
| DeepSeek-R1-Distill-Qwen-14B  |          95.0           |             93.9             |
| DeepSeek-R1-Distill-Qwen-32B  |          96.6           |             94.3             |
| DeepSeek-R1-Distill-Llama-8B  |          88.6           |             89.1             |
| DeepSeek-R1-Distill-Llama-70B |          96.4           |             94.5             |

To reproduce these results use the following command:

```shell
NUM_GPUS=1 # Set to 8 for 32B and 70B models
MODEL=deepseek-ai/{model_name}
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,data_parallel_size=$NUM_GPUS,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
OUTPUT_DIR=data/evals/$MODEL

lighteval vllm $MODEL_ARGS "custom|math_500|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR
```

Alternatively, you can launch Slurm jobs as follows:

```shell
python scripts/run_benchmarks.py --model-id {model_id}  --benchmarks math_500
```

### GPQA Diamond

We are able to reproduce Deepseek's reported results on the GPQA Diamond benchmark within ~1-3 standard deviations:

| Model                         | GPQA Diamond (🤗 LightEval) | GPQA Diamond (DeepSeek Reported) |
|:------------------------------|:---------------------------:|:--------------------------------:|
| DeepSeek-R1-Distill-Qwen-1.5B |            34.3             |               33.8               |
| DeepSeek-R1-Distill-Qwen-7B   |            50.5             |               49.1               |
| DeepSeek-R1-Distill-Qwen-14B  |            59.6             |               59.1               |
| DeepSeek-R1-Distill-Qwen-32B  |            63.6             |               62.1               |
| DeepSeek-R1-Distill-Llama-8B  |            52.0             |               49.0               |
| DeepSeek-R1-Distill-Llama-70B |            67.2             |               65.2               |

To reproduce these results use the following command:

```shell
NUM_GPUS=1 # Set to 8 for 32B and 70B models
MODEL=deepseek-ai/{model_name}
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,data_parallel_size=$NUM_GPUS,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
OUTPUT_DIR=data/evals/$MODEL

lighteval vllm $MODEL_ARGS "custom|gpqa:diamond|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR
```

```shell
python scripts/run_benchmarks.py --model-id {model_id}  --benchmarks gpqa
```

### LiveCodeBench

We are able to reproduce Deepseek's reported results on the LiveCodeBench code generation benchmark within ~1-3 standard deviations:

| Model                         | LiveCodeBench (🤗 LightEval) | GPQA Diamond (DeepSeek Reported) |
|:------------------------------|:----------------------------:|:--------------------------------:|
| DeepSeek-R1-Distill-Qwen-1.5B |             16.3             |               16.9               |
| DeepSeek-R1-Distill-Qwen-7B   |             36.6             |               37.6               |
| DeepSeek-R1-Distill-Qwen-14B  |             51.5             |               53.1               |
| DeepSeek-R1-Distill-Qwen-32B  |             56.6             |               57.2               |
| DeepSeek-R1-Distill-Llama-8B  |             37.0             |               39.6               |
| DeepSeek-R1-Distill-Llama-70B |             54.5             |               57.5               |

To reproduce these results use the following command:

```shell
NUM_GPUS=1 # Set to 8 for 32B and 70B models, or data_parallel_size=8 with the smaller models for speed
MODEL=deepseek-ai/{model_name}
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,data_parallel_size=$NUM_GPUS,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
OUTPUT_DIR=data/evals/$MODEL

lighteval vllm $MODEL_ARGS "extended|lcb:codegeneration|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR
```

```shell
python scripts/run_benchmarks.py --model-id {model_id}  --benchmarks lcb
```

## Data generation

### Generate data from a smol distilled R1 model

The following example can be run in 1xH100. 
First install the following dependencies:

```shell
uv pip install "distilabel[vllm]>=1.5.2"
```

Now save the following snippet into a file named `pipeline.py` and run it with `python pipeline.py`. It will generate 4 outputs for each of the 10 examples (change the username for the repository to your org/user name):

```python
from datasets import load_dataset
from distilabel.models import vLLM
from distilabel.pipeline import Pipeline
from distilabel.steps.tasks import TextGeneration


prompt_template = """\
You will be given a problem. Please reason step by step, and put your final answer within \boxed{}:
{{ instruction }}"""

dataset = load_dataset("AI-MO/NuminaMath-TIR", split="train").select(range(10))

model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"  # Exchange with another smol distilled r1

with Pipeline(
    name="distill-qwen-7b-r1",
    description="A pipeline to generate data from a distilled r1 model",
) as pipeline:

    llm = vLLM(
        model=model_id,
        tokenizer=model_id,
        extra_kwargs={
            "tensor_parallel_size": 1,
            "max_model_len": 8192,
        },
        generation_kwargs={
            "temperature": 0.6,
            "max_new_tokens": 8192,
        },
    )
    prompt_column = "problem"
    text_generation = TextGeneration(
        llm=llm, 
        template=prompt_template,
        num_generations=4,
        input_mappings={"instruction": prompt_column} if prompt_column is not None else {}
    )


if __name__ == "__main__":
    distiset = pipeline.run(dataset=dataset)
    distiset.push_to_hub(repo_id="username/numina-deepseek-r1-qwen-7b")
```

Take a look at the sample dataset at [HuggingFaceH4/numina-deepseek-r1-qwen-7b](https://huggingface.co/datasets/HuggingFaceH4/numina-deepseek-r1-qwen-7b).


### Generate data from DeepSeek-R1

To run the bigger DeepSeek-R1, we used 2 nodes, each with 8×H100 GPUs using the slurm file present in this repo at `slurm/generate.slurm`. First, install the dependencies:

(for now we need to install the vllm dev wheel that [fixes the R1 cuda graph capture](https://github.com/vllm-project/vllm/commits/221d388cc5a836fa189305785ed7e887cea8b510/csrc/moe/moe_align_sum_kernels.cu))
```shell
pip install https://wheels.vllm.ai/221d388cc5a836fa189305785ed7e887cea8b510/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl --extra-index-url https://download.pytorch.org/whl/cu121

uv pip install "distilabel[vllm,ray,openai]>=1.5.2"
```

And then run the following command:

```shell
sbatch slurm/generate.slurm \
    --hf-dataset AI-MO/NuminaMath-TIR \
    --temperature 0.6 \
    --prompt-column problem \
    --model deepseek-ai/DeepSeek-R1 \
    --hf-output-dataset username/r1-dataset
```

> [!NOTE]  
> While the job is running, you can setup an SSH tunnel through the cluster login node to access the Ray dashboard from your computer running `ssh -L 8265:ray_ip_head_node:8265 <login_node>`, then browsing `http://localhost:8265`

## Contributing

Contributions are welcome. Please refer to https://github.com/huggingface/open-r1/issues/23.

## 研究记录

### 1. 训练grpo

使用两张3060显卡训练qwen0.5b模型, 在recipes目录新建一个Qwen2-0.5B-Instruct的目录, 该目录下新建config_demo.yaml和zero3.yaml两个文件, 使用如下命令启动训练：

```bash
accelerate launch --config_file recipes/Qwen2-0.5B-Instruct/zero3.yaml src/open_r1/grpo.py  --config recipes/Qwen2-0.5B-Instruct/config_demo.yaml"
```

经测试可成功启动训练过程, 训练过程中遇到一些问题, 记录一下:

1. 提示train_batch_size计算后的值与实际给出的gradient_accumulation_steps等参数不相符问题：
  train_batch_size=per_device_train_batch_size * gradient_accumulation_steps * world_size, 正常train_batch_size应该等于上述公式计算出来的结果，训练过程中会调用`deepspeed.HfTrainerDeepSpeedConfig::trainer_config_process() `来自动根据参数计算train_batch_size的大小；

  检查per_device_train_batch_size、gradient_accumulation_steps和world_size参数的设置即可

2. 训练过程中一些重要参数解释：

  - num_processes：与word_size相同，但更加通用，用来表示分布式训练中的进程总数；
  - num_machines：参与训练的机器或节点数量；
  - gradient_accumulation_steps：是梯度累积的步数，用于在显存有限的情况下模拟更大的批量大小，不会增加显存占用，但会增加计算时间；
  - per_device_train_batch_size：每个设备（如 GPU）上的训练批量大小，批量大小越大，显存占用越多，但训练速度可能更快，批量大小越小，显存占用越少，但训练速度可能更慢；
  - num_generations：是生成任务中的一个参数，表示生成多少个候选结果，在文本生成任务中，如果 `num_generations=5`，则模型会生成 5 个不同的候选文本，生成数量越多，显存占用越多，计算时间也越长；
  - num_train_epochs：控制训练的总轮次，不会导致显存溢出，但会延长训练时间，它会直接影响总优化步数(**Total optimization steps**)；
  - max_prompt_length：生成任务中提示(prompt)序列的最大长度，通常用于控制输入序列的长度，它会直接影响显存占用，输入序列越长，模型需要处理的Token数量越多，显存占用也会增加；
  - max_completion_length：生成任务中生成序列的最大长度，其长度直接影响显存占用；
  - max_steps：训练过程中优化步骤(optimization steps)的最大数量，通常用于控制训练的总步数，本身不会直接影响显存占用，但在长时间训练中，显存泄露可能会导致显存使用逐渐增加；
  - 

3. 使用vllm一直失败，使用Ubuntu18，需要升级cuda到12.4，对应vllm版本需要是0.7以上，目前只能装到0.5，实际使用存在很多问题，然后，vllm只能使用一张显卡进行训练，放弃使用vllm进行训练，采用accelerate来训练之；

