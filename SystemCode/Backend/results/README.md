---
license: apache-2.0
base_model: t5-large
tags:
- generated_from_trainer
model-index:
- name: T5-large-10K-summarization
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# T5-large-10K-summarization

This model is a fine-tuned version of [t5-large](https://huggingface.co/t5-large) on an unknown dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 8
- eval_batch_size: 8
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 3.0

### Framework versions

- Transformers 4.39.3
- Pytorch 2.2.2+cu121
- Tokenizers 0.15.2
