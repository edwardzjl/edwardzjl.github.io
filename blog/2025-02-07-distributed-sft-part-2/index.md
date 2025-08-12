---
slug: distributed-sft-part-2
authors: [jlzhou]
tags: [LLM, distributed-training]
---

# Distributed SFT Part 2: Scaling Locally

## Introduction

[In the first part of this series](https://edwardzjl.github.io/distributed-sft-part-1), we covered the basics of setting up a local SFT experiment using `trl`. We learned how to format datasets for `trl`'s `SFTTrainer` and preprocess them to fit the required structure.

Now, it's time to take the next step. In this post, we'll focus on scaling the SFT setup to handle larger tasks. Specifically, we'll explore how to fine-tune an LLM in a single-node, multi-GPU environment. Along the way, we'll discuss optimization techniques to reduce memory usage, speed up training, and enable fine-tuning of even larger models. Let's get started!

<!-- truncate -->

## Prerequisites

To follow along with this tutorial, you'll need a machine equipped with multiple NVIDIA GPUs. Ensure that the GPUs are connected via high-speed interconnects to minimize communication overhead. For reference, I ran this experiment using 8 NVIDIA V100 SXM2 GPUs.

**Important Considerations:**

1. GPU Architecture: While I ran this experiment with V100 GPUs, newer architectures like Ampere or Hopper are strongly recommended. These GPUs offers advanced features, such as support for more efficient precision types and improved communication speeds. Additionally, techniques like [flash-attention](https://github.com/Dao-AILab/flash-attention) are [only compatible with Ampere or newer GPUs](https://github.com/Dao-AILab/flash-attention/issues/524).

2. Interconnect Quality: Verify GPU communication bandwidth using `nvidia-smi topo -m`. Poor interconnects can become a bottleneck during training.

Additionally, you'll need to install the following dependencies:

```txt
datasets
torch
transformers
trl
```

## Tuning Hyperparameters

[BAAI/Infinity-Instruct](https://huggingface.co/datasets/BAAI/Infinity-Instruct) provides several officially fine-tuned models, including [Llama3.1-70B](https://huggingface.co/BAAI/Infinity-Instruct-7M-Gen-Llama3_1-70B), [mistral-7B](https://huggingface.co/BAAI/Infinity-Instruct-7M-Gen-mistral-7B), [Qwen2-7B](https://huggingface.co/BAAI/Infinity-Instruct-3M-0625-Qwen2-7B) and [Yi-1.5-9B](https://huggingface.co/BAAI/Infinity-Instruct-3M-0625-Yi-1.5-9B). They also generously share the training details for these models.

For this tutorial, we'll use the [hyperparameters for Qwen2-7B](https://huggingface.co/BAAI/Infinity-Instruct-3M-0625-Qwen2-7B#training-details) as a reference. Here's how these hyperparameters map to training arguments in `trl`:

- epoch: `--num_train_epochs`
- lr: `--learning_rate`
- lr_warmup_steps: `--warmup_steps`
- lr_decay_style: `--lr_scheduler_type` (Set to `cosine_with_min_lr` along with `min_lr`. Available scheduler options can be found [here](https://huggingface.co/docs/transformers/main_classes/optimizer_schedules#transformers.SchedulerType).)
- min_lr: `--lr_scheduler_kwargs` (Set to `"{\"min_lr\": 0}"`. This argument isn't clearly documented; I discovered it through [this PR](https://github.com/huggingface/transformers/pull/29341) and [this test case](https://github.com/huggingface/transformers/blob/d3af76df58476830eb5b5981decc64af15e369f5/tests/trainer/test_trainer.py#L1065).)
- weight_decay: `--weight_decay`
- adam_beta1: `--adam_beta1`
- adam_beta2: `--adam_beta2`
- clip_grad: `--max_grad_norm`

One additional parameter worth mentioning is the `global_batch_size`, which isn't directly set in the training script. The global batch size is determined by the equation:

`global_batch_size = per_device_train_batch_size * gradient_accumulation_steps * numGPUs`.

For example, if our target global batch size is 528 and we're using 8 GPUs, the local batch size (per GPU) would be:

`528 / 8 = 66`.

If we can fit 2 samples per batch in each GPU, we can then set `per_device_train_batch_size` to 2 and `gradient_accumulation_steps` to 33.

Another important consideration is training precision. Modern GPUs (Ampere series or newer) supports `bf16` and `tf32`, while older GPUs only support `fp16` and `fp32`. When fine-tuning, make sure the precision matches that of the base model. Specifically, avoid using `fp16` if the base model was trained with `bf16`. For more details, refer to [this PR](https://github.com/huggingface/transformers/pull/10956).

You can find the data type of your base model by the `torch_dtype` field in the `config.json` file. So if you're fine-tuning a `bf16` model but don't have access to Ampere or newer GPUs (like me), it's best to stick with `fp32` for now.

Now that we've covered the essential hyperparameters and considerations, let's move on to some optimization techniques that will help improve training efficiency and resource usage.

### Gradient Accumulation

You may have noticed that I used `per_device_train_batch_size` and `gradient_accumulation_steps` to calculate the local batch size. Gradient accumulation allows you to accumulate gradients over multiple mini-batches before updating the model. This technique is particularly useful when the desired batch size exceeds your hardware's memory capacity.

As a general guideline:

- Use the largest `per_device_train_batch_size` that fits within your VRAM
- Adjust `gradient_accumulation_steps` to achieve your target batch size if necessary.

This way, you can effectively simulate a larger batch size without running into memory limitations

### Gradient Checkpointing

Gradient checkpointing is a memory optimization technique that reduces memory usage by trading off computation. During training, a large portion of memory is used to store intermediate activations for backpropagation. Gradient checkpointing reduces this memory usage by selectively saving a subset of activations and recomputing the rest during the backward pass.

Note: According to <https://pytorch.org/docs/stable/checkpoint.html>:

> There are currently two checkpointing implementations available, determined by the `use_reentrant` parameter. It is recommended that you use `use_reentrant=False`.

You can read that section for a deeper understanding of  the differences between the two implementations.

At the time of writing, the `transformers` library (v4.48.1) [uses the reentrant implementation by default](https://github.com/huggingface/transformers/blob/2e752ead46a8845e8a160d2043c1336447895690/src/transformers/modeling_utils.py#L2538). To use the non-reentrant version, you must explicitly pass the following argument:

```sh
--gradient_checkpointing_kwargs "{\"use_reentrant\": false}"
```

### ZeRO

There are several approaches to parallelizing training tasks, including **Data Parallelism (DP)**, **Tensor Parallelism (TP)**, **Pipeline Parallelism (PP)**, **Zero Redundancy Optimizer (ZeRO)**, **Sequence Parallelism** and **Expert Parallelism**. For a detailed overview of these methods, I recommend checking out [this excellent resource](https://github.com/stas00/ml-engineering/tree/master/training/model-parallelism#scalability-concepts).

In this tutorial, we'll focus on **ZeRO**, which provides greater efficiency than traditional DP without requiring modifications to the training code.

ZeRO (Zero Redundancy Optimizer) is a powerful technique for scaling training by reducing memory usage. If you're new to ZeRO, check out [the original paper](https://arxiv.org/abs/1910.02054) or [this detailed article](https://github.com/stas00/ml-engineering/tree/master/training/model-parallelism#zero-data-parallelism).

ZeRO has three stages, each targeting different aspects  of memory savings:

- Stage 1: Reduces optimizer state memory.
- Stage 2: Further reduces gradient memory.
- Stage 3: Fully partitions model states, achieving the highest memory savings at the cost of increased communication overhead.

Stage 3 provides the greatest memory efficiency but can significantly slow down training if inter-GPU communication is not fast enough. As a general guideline:

- Start with Stage 2.
- Try Stage 3 only if Stage 2 still leads to CUDA OOM.

That being said, it’s always worth testing both Stage 2 and Stage 3 on your setup to determine which one performs better on your hardware.

For this tutorial, we will use the official implementation of ZeRO -- [DeepSpeed](https://www.deepspeed.ai/). To use DeepSpeed, you'll need to install it first. DeepSpeed provides C++/CUDA ops that can be pre-installed or JIT-compiled. If you choose the pre-installation option, refer to [this documentation](https://www.deepspeed.ai/tutorials/advanced-install/#pre-install-deepspeed-ops). we’ll install DeepSpeed using the JIT method by running:

```sh
pip install deepspeed
```

The Hugging Face `transformers` library provides built-in support for DeepSpeed. You can enable it by specifying a DeepSpeed config file using the `--deepspeed` flag in your training script. For more information, refer to the [DeepSpeed documentation in transformers](https://huggingface.co/docs/transformers/deepspeed).

### Liger Kernel

[Liger Kernel](https://github.com/linkedin/Liger-Kernel) is a collection of Triton kernels developed by LinkedIn to reduce memory usage and increase training throughput. The best part is that it requires no complex configuration, making it an easy addition to your setup. To install it, run:

```sh
pip install liger-kernel
```

Once installed, add the `--use_liger` flag to your training script, and you'll automatically save VRAM without any extra setup or hassle. It's a straightforward way to optimize your training without sacrificing performance.

### Sample Packing

Large models are trained on GPUs to leverage their parallelism. However, in the context of language models, where we train on text sequences, the length of each sample varies.

The traditional approach to handle variable-length sequences is to pad each sample to match the longest one in a batch. While this ensures uniform input dimensions, it also leads to considerable memory waste due to the padding.

[Sample packing](https://arxiv.org/abs/2407.09105) addresses this issue by combining shorter samples into a single sequence. This technique allows for more efficient GPU memory usage, reducing waste and potentially speeding up training.

While the concept is straightforward, Implementing it correctly can be one of the most challenging tasks of this experiment for me.

At the first glance, [trl supports packing dataset by simply passing an argument](https://huggingface.co/docs/trl/sft_trainer#packing-dataset--constantlengthdataset-). However, uppon [further investigation](https://github.com/huggingface/trl/blob/f34b70a32ef2820d3fd5c5b1ff6d1fd1e7799f04/trl/trainer/sft_trainer.py#L459), I found that the implementation might not suit my needs. As pointed out in [this issue](https://github.com/huggingface/trl/issues/805), the attention mask is not handled properly, which can lead to potential cross contamination in attention between sequences. The following image illustrates this issue clearly. On the left is the result of using `--packing`, and on the right is the correct way to pack samples:

![image/png](https://cdn-uploads.huggingface.co/production/uploads/63eb008e5c837d9968f1eb71/lzpKqOADV5mdOdclPbQ9C.png)

After further digging, I found that, at least for now, ['the correct way of packing' is supported only with Flash Attention](https://github.com/huggingface/transformers/issues/27640#issuecomment-2619471784). If you don't have access to Ampere or newer GPUs, you may need to stick with the traditional padding approach.

However, if you're lucky enough to have those GPUs, you can follow [this blog post](https://huggingface.co/blog/packing-with-FA2) to enable sample packing during training. Note that I haven't personally validated this approach. Also, as of writing, there are some PRs related to this feature that aren't released yet (for example [this one](https://github.com/huggingface/trl/pull/2158)). To access this functionality, you may need to install `trl` and `transformers` from source:

```sh
pip install git+https://github.com/huggingface/trl
pip install git+https://github.com/huggingface/tranformers
```

## Distributed Training

With all the optimizations in place, we're now ready to scale our SFT experiment across multiple GPUs. To do so, we can use tools like [torchrun](https://pytorch.org/docs/stable/elastic/run.html), [deepspeed](https://www.deepspeed.ai/getting-started/) or [accelerate](https://huggingface.co/docs/accelerate/index). Personally I prefer `torchrun` for its simplicity and ease of use.

By running the following command, we can distribute the training job across multiple GPUs:

Oh, and don't forget to set up `wandb` for logging — we're doing proper fine-tuning now! 😉

<details><summary>sft2.sh</summary>

```sh
torchrun \
  --nproc_per_node 8 \
  sft.py \
  --model_name_or_path Qwen/Qwen2.5-3B \
  --dataset_name BAAI/Infinity-Instruct \
  --dataset_config 0625 \
  --do_train \
  --learning_rate 1e-5 \
  --lr_scheduler_type cosine_with_min_lr \
  --lr_scheduler_kwargs "{\"min_lr\": 0}" \
  --warmup_steps 40 \
  --weight_decay 0.0 \
  --max_grad_norm 1.0 \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --per_device_train_batch_size 11 \
  --gradient_accumulation_steps 6 \
  --gradient_checkpointing \
  --gradient_checkpointing_kwargs "{\"use_reentrant\": false}" \
  --num_train_epochs 3 \
  --use_liger \
  --deepspeed ./ds-config.json \
  --output_dir /tmp/Qwen2.5-3B-Infinity-Instruct-0625 \
  --report_to wandb \
  --run_name my-second-sft-exp
```

</details>

<details><summary>ds-config.json</summary>

```json
{
    "fp16": {
        "enabled": false
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto"
        }
    },
    "zero_optimization": {
        "stage": 2,
        "overlap_comm": false,
        "allgather_bucket_size": 5e8,
        "reduce_bucket_size": "auto",
        "allgather_partitions": true,
        "reduce_scatter": true,
        "contiguous_gradients": true,
        "round_robin_gradients": true
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false
}
```

</details>

Thanks to all the optimizations, I was able to fine-tune a 3B model instead of the 0.5B model used in the first part.

It did take a considerable amount of time (about 133 hours) to complete the training on V100s, so I highly recommend use modern GPUs and enabling Flash Attention and sample packing for better performance.

## Evaluating

Now that the training is complete, it’s important to evaluate whether everything was done correctly.

A quick way to check the model's performance is to interact with it. You can refer to [the quickstart section of the official SFT model](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct#quickstart) to try it out. Here's an example of interacting with the model I just fine-tuned:

```text
Me: Give me a short introduction to large language model.

AI: A large language model (LLM) is a type of artificial intelligence (AI) that is designed to understand and generate human language. These models are trained on vast amounts of text data, allowing them to learn patterns, structures, and nuances
```

(Note: that the response from the AI is truncated due to the `max_new_tokens` I set, but you can see that the model is responding appropriately.)

While direct interactions are useful for quick checks, formal evaluations are essential for more rigorous validation. Evaluating LLMs is quite a broad topic, and I'm only going to share a few tips here.

There are several frameworks available for evaluating LLMs, making it challenging to choose the best one, and comparing results from different frameworks can sometimes lead to unfair conclusions.

One well-known evaluation platform is the [Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard#/), which ranks LLMs based on their evaluation results. The Open LLM Leaderboard uses [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) as its backend. By using this same tool, you can ensure fair comparisons with models in the leaderboard. So for this tutorial, I'll to use `lm-evaluation-harness` to run the same evaluations used on the Open LLM Leaderboard to assess the model I just fine-tuned.

The `lm-evaluation-harness` [integrates all the tasks used in the Open LLM Leaderboard](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/leaderboard/README.md). To evaluate your model on these tasks, you can run the following command:

```sh
lm_eval \
  --model hf \
  --model_args pretrained=$MODEL_YOU_WANT_TO_EVAL \
  --tasks leaderboard
```

### MATH-hard Task Unavailable

However, As of writing, the `competition_math` dataset is [currently unavailable due to legal issues](https://huggingface.co/datasets/hendrycks/competition_math/discussions/5). As a result, we'll need to skip the `MATH-hard` task that relies on this dataset. You can modify your script to include all other tasks except `leaderboard_math_hard`:

```sh
lm_eval \
  --model hf \
  --model_args pretrained=$MODEL_YOU_WANT_TO_EVAL \
  --tasks leaderboard_bbh,leaderboard_gpqa,leaderboard_ifeval,leaderboard_mmlu_pro,leaderboard_musr
```

### Evaluating Code Generation

In addition to the leaderboard evaluations, if you're interested in evaluating your model on code generation tasks (such as [humaneval](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/humaneval)), keep in mind that the generated code usually needs to be executed to evaluate its correctness. Since executing LLM generated code can be risky, most frameworks will default to abort on such tasks. To allow code execution during evaluation, you need to set `HF_ALLOW_CODE_EVAL` to `1` and include the `--confirm_run_unsafe_code` argument in your evaluation command:

```sh
lm_eval \
  --model hf \
  --model_args pretrained=$MODEL_YOU_WANT_TO_EVAL \
  --tasks leaderboard_bbh,leaderboard_gpqa,leaderboard_ifeval,leaderboard_mmlu_pro,leaderboard_musr \
  --confirm_run_unsafe_code  # Add this line
```

## Conclusion

In this post, we’ve covered everything from the basic setup to advanced techniques for scaling large language models in a single-node, multi-GPU environment. By utilizing DeepSpeed and trl, we can efficiently fine-tune models like Qwen2-3B and beyond, even on hardware that would otherwise be unable to support such models. I've also uploaded the fine-tuned model to the Hugging Face model hub, so you can try it out for yourself: <https://huggingface.co/jlzhou/Qwen2.5-3B-Infinity-Instruct-0625>.

In the next part of this series, we’ll explore distributed training across multiple nodes, tackling more complex setups with multiple GPUs across different machines. Stay tuned!
