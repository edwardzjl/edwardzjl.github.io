---
slug: distributed-sft-part-1
title: Distributed SFT Part 1: Starting Locally
authors: [jlzhou]
tags: [LLM, distributed-training]
---

# Distributed SFT Part 1: Starting Locally

## Introduction

Welcome to this series of articles documenting the lessons I learned during my first attempt at running distributed supervised fine-tuning (SFT) tasks using [trl](https://github.com/huggingface/trl) and [DeepSpeed](https://github.com/microsoft/DeepSpeed).

This series will walk you through my journey, starting with a simple local experiment and progressively scaling up to a distributed environment. The three parts of this series are:

- **Part 1: The Local Experiment** -- I will show you how I ran my very first local SFT experiment, following the official [trl documentation](https://huggingface.co/docs/trl/sft_trainer).

- **Part 2: Multi GPU** -- We will leverage **single-machine, multi-GPU** parallel training to complete a full SFT task in our local environment.

- **Part 3: Multi Machine** -- We'll take things a step further by submitting the same training task to a Kubernetes cluster, utilizing **multi-machine, multi-GPU** training with [Kubeflow's Training Operator](https://github.com/kubeflow/training-operator).

A quick note about myself: I'm a software development engineer who is fairly new to the field of deep learning. If these articles seem too basic for you, I appreciate your patience as I navigate this learning journey.

<!-- truncate -->

## Prerequisites

To follow this tutorial, you'll need a machine with at least one NVIDIA GPU. I ran the experiment on a V100 without encountering any memory issues. If your GPU has less than 32GB of VRAM, you may need to reduce the `per_device_train_batch_size` or consider using truncation (although this is not recommended) to prevent CUDA out-of-memory (OOM) errors.

You'll also need the following dependencies:

```txt
datasets
transformers
trl
torch
```

## Training

The `trl` library offers some excellent example training scripts, and we'll start with this one: <https://github.com/huggingface/trl/blob/main/trl/scripts/sft.py>

Copy the script to your development machine (or notebook), select a base model, and pick an SFT dataset to run the experiment. For this experiment, I chose [Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) as the base model for its compact size, and [BAAI/Infinity-Instruct](https://huggingface.co/datasets/BAAI/Infinity-Instruct) as the SFT dataset (somehow randomly 😌). You can explore other interesting datasets here: <https://github.com/mlabonne/llm-datasets>.

### Command-line Arguments

The training script (`sft.py`) exposes a variety of useful command-line arguments that allow you to customize the fine-tuning process. These arguments are mapped to specific properties in the following classes:

- [ScriptArguments](https://huggingface.co/docs/trl/v0.13.0/en/script_utils#trl.ScriptArguments)
- [ModelConfig](https://github.com/huggingface/trl/blob/v0.13.0/trl/trainer/model_config.py#L20)
- [SFTConfig](https://huggingface.co/docs/trl/v0.13.0/en/sft_trainer#trl.SFTConfig), which extends [TrainingArguments](https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments)

You can pass any of these arguments directly from the command line by prepending them with `--`. For instance, passing `--dataset_name` will set the `dataset_name` field in the `trl.ScriptArguments` class.

Let's take a look at the arguments used for this tutorial:

- `--model_name_or_path`: Specifies the base model to fine-tune.
- `--dataset_name`: Defines the dataset to use for fine-tuning.
- `--dataset_config`: Some datasets come with multiple configurations (versions). This argument lets you choose the version you want to use.
- `--do_train`: Tells the script to start the training process.
- `--per_device_train_batch_size`: Defines the batch size for each GPU.
- `--output_dir`: Specifies the directory where the model will be saved.
- `--max_steps`: Sets the maximum number of training steps.
- `--logging_steps`: Sets how often logs are recorded during training.

For convenience, I prefer to save the full command in a shell script for easy execution. Here's the script I used for this tutorial:

```sh
python sft.py \
  --model_name_or_path Qwen/Qwen2.5-0.5B \
  --dataset_name BAAI/Infinity-Instruct \
  --dataset_config 0625 \
  --do_train \
  --per_device_train_batch_size 4 \
  --output_dir /tmp/my-first-sft-exp \
  --max_steps 10 \
  --logging_steps 1
```

Notes:

- I selected the smallest version of the dataset and limited the experiment to just 10 steps for a quicker run.
- Since the training is only 10 steps, I set `--logging_steps` to 1 to see logs more frequently.
- The `--per_device_train_batch_size` is set to 4, as the goal here isn't model quality but simply to run the experiment without CUDA OOM. Any number that can fit in your VRAM should work.

> *Updated 2025-02-18:*
>
> `trl` provides a convenient helper function to parse training args from a YAML file, you can find more details [here](https://huggingface.co/docs/trl/main/script_utils#trl.TrlParser.parse_args_and_config).
>
> With this feature, you can save the above training arguments in a YAML file (e.g., `recipe.yaml`) as follows:
>
> ```yaml
> model_name_or_path: Qwen/Qwen2.5-0.5B
> dataset_name: BAAI/Infinity-Instruct
> dataset_config: '0625'
> do_train: true
> per_device_train_batch_size: 4
> output_dir: /tmp/my-first-sft-exp
> max_steps: 10
> logging_steps: 1
> ```
>
> And launch the training with:
> ```sh
> python sft.py --config recipe.yaml
> ```

### The Oops

Now if you use the same dataset and execute the same script, you'll likely encounter a (not so helpful) error message:

```console
$ ./quickstart.sh 
Resolving data files: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 35/35 [00:00<00:00, 50.35it/s]
Map:   0%|                                                                                                                                                                         | 0/659808 [00:00<?, ? examples/s]
Traceback (most recent call last):
  File "/home/jovyan/sft-walkthrough/sft.py", line 126, in <module>
    main(script_args, training_args, model_args)
  File "/home/jovyan/sft-walkthrough/sft.py", line 97, in main
    trainer = SFTTrainer(
  ...
  File "/home/jovyan/.local/lib/python3.10/site-packages/trl/trainer/sft_trainer.py", line 416, in tokenize
    element[dataset_text_field] if formatting_func is None else formatting_func(element),
  File "/home/jovyan/.local/lib/python3.10/site-packages/datasets/formatting/formatting.py", line 277, in __getitem__
    value = self.data[key]
KeyError: 'text'
```

### The Fix

> *Updated 2025-02-18:*
>
> - Starting from trl 0.15.0 (in [this PR](https://github.com/huggingface/trl/pull/2405)) the 'conversations' column is no longer supported. We need to rename it to 'messages'.
> - In [this PR](https://github.com/huggingface/trl/pull/2862) (not yet released as of writing), support for the 'conversations' column is back and the whole preprocessing is simplified, we no longer need to map the dict keys('from' -> 'role', 'value' -> 'content') ourselves.
>
> *Updated 2025-02-19:*
>
> The above PR is released in trl 0.15.1.

This error message is a bit confusing--it states that the `SFTTrainer` requires the dataset to have a 'text' field. However, according to the [dataset format and types](https://huggingface.co/docs/trl/dataset_formats#overview-of-the-dataset-formats-and-types), 'text' is used for standard dataset, while 'messages' should be used for conversational datasets. After a lot of googling, I came across [this tracking issue](https://github.com/huggingface/trl/issues/2071), [this line of code](https://github.com/huggingface/trl/blob/v0.13.0/trl/trainer/sft_trainer.py#L250) and [this function](https://github.com/huggingface/trl/blob/v0.13.0/trl/extras/dataset_formatting.py#L78). It seems that for the current implementation (`trl == 0.13.0`) we have two options:

1. Format the dataset ourselves (apply a chat template) and place the formatted data into the 'text' field.
2. Convert our dataset in a way that allows `trl` to handle the transformation for us.

For the second option to work, the dataset must:

- Contain a 'messages' or 'conversations' field.
- Have each element in the 'messages' (or 'conversations') field include both a 'content' field and a 'role' field.

Examining the dataset I used revealed a mismatch: while it has a 'conversations' field, the elements inside use 'from' and 'value' as keys instead of 'role' and 'content'. As a lazy coder, I opted for the second approach and updated the training script accordingly. Additionally, I also remove all other fields in the dataset, as they are unused for the SFT task. Removing them will slightly reduce memory footprint and speed up processing.

```python
...
def main(script_args, training_args, model_args):
    ...
    ################
    # Dataset
    ################
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    def convert_fields(message: dict) -> dict:
        _message = {
          "role": message["from"],
          "content": message["value"],
        }
        # Qwen2.5 tokenizer only supports "system", "user", "assistant" and "tool"
        # See <https://huggingface.co/Qwen/Qwen2.5-3B/blob/main/tokenizer_config.json#L198>
        if _message["role"] == "human":
            _message["role"] = "user"
        elif _message["role"] == "gpt":
            _message["role"] = "assistant"
        elif _message["role"] == "system":
            # nothing to be done.
            ...
        else:
            # In case there are any other roles, print them so we can improve in next iteration.
            print(_message["role"])
        return _message

    def convert_messages(example):
        example["conversations"] = [convert_fields(message) for message in example["conversations"]]
        return example

    # remove unused fields
    dataset = dataset.remove_columns(["id", "label", "langdetect", "source"]).map(convert_messages)
    ...
```

With that update, the script ran without any issues! You should be able to see the training log like:

```console
$ ./quickstart.sh 
Resolving data files: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 35/35 [00:02<00:00, 17.26it/s]
Map: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 659808/659808 [01:19<00:00, 8280.44 examples/s]
Map: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 659808/659808 [08:33<00:00, 1284.45 examples/s]
{'loss': 1.8859, 'grad_norm': 14.986075401306152, 'learning_rate': 1.8e-05, 'epoch': 0.0}                                                                                                                                     
{'loss': 1.4527, 'grad_norm': 13.9092378616333, 'learning_rate': 1.6000000000000003e-05, 'epoch': 0.0}                                                                                                                        
{'loss': 1.467, 'grad_norm': 7.388503074645996, 'learning_rate': 1.4e-05, 'epoch': 0.0}                                                                                                                                       
{'loss': 1.7757, 'grad_norm': 9.457520484924316, 'learning_rate': 1.2e-05, 'epoch': 0.0}                                                                                                                                      
{'loss': 1.9043, 'grad_norm': 10.256357192993164, 'learning_rate': 1e-05, 'epoch': 0.0}                                                                                                                                       
{'loss': 1.6163, 'grad_norm': 10.774249076843262, 'learning_rate': 8.000000000000001e-06, 'epoch': 0.0}                                                                                                                       
{'loss': 1.1774, 'grad_norm': 5.897563457489014, 'learning_rate': 6e-06, 'epoch': 0.0}                                                                                                                                        
{'loss': 1.8093, 'grad_norm': 8.3130464553833, 'learning_rate': 4.000000000000001e-06, 'epoch': 0.0}                                                                                                                          
{'loss': 1.8387, 'grad_norm': 7.102719306945801, 'learning_rate': 2.0000000000000003e-06, 'epoch': 0.0}                                                                                                                       
{'loss': 1.4251, 'grad_norm': 9.853829383850098, 'learning_rate': 0.0, 'epoch': 0.0}                                                                                                                                          
{'train_runtime': 38.8598, 'train_samples_per_second': 1.029, 'train_steps_per_second': 0.257, 'train_loss': 1.635251808166504, 'epoch': 0.0}                                                                                 
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:38<00:00,  3.89s/it]
```

## Conclusion

In this first part, we've walked through setting up a local SFT experiment using `trl`. This library provides a user-friendly interface for fine-tuning LLMs with custom datasets. We also covered the correct dataset format required for `trl`'s `SFTTrainer` and how to preprocess datasets to meet these requirements.

In the next part, we'll delve into scaling this setup locally using a single-node, multi-GPU configuration to tackle a complete SFT task. Additionally, we'll explore various optimization techniques to fit a bigger model into your GPU and accelerate the training process. Stay tuned!
