import csv
import os

import pandas as pd
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset, Dataset

from eval import bleu

#加载模型
max_seq_length = 2048
dtype = None
load_in_4bit = True
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    token = "https://hf-mirror.com"
)

#准备训练数据
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{}
### Input:
{}
### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token # 必须添加 EOS_TOKEN
def formatting_prompts_func(examples):
    inputs       = examples["code"]
    outputs      = examples["doc"]
    texts = []
    for input, output in zip(inputs, outputs):
        instruction = "Please summarize the following code."
        # 必须添加EOS_TOKEN，否则无限生成
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

def decode_preds(preds):
    preds, labels = preds
    decoded_preds = tokenizer.decode_batch(preds)
    decoded_labels = tokenizer.decode_batch(labels)
    return decoded_labels, decoded_preds

# compute metrics
def compute_valid_metrics(eval_preds):
    decoded_labels, decoded_preds = decode_preds(eval_preds)
    refs = [ref.strip().split() for ref in decoded_labels]
    cans = [can.strip().split() for can in decoded_preds]
    result = {}
    result.update(bleu(references=refs, candidates=cans))
    return result
#hugging face数据集路径
# train_dataset = load_dataset("code-search-net/code_search_net", split=f"train[:100000]", trust_remote_code=True)
dataset = load_dataset('json', data_files={'train': 'tl_data/train.json', 'test': 'tl_data/test.json'})
train_dataset = dataset["train"]
train_dataset = train_dataset.map(formatting_prompts_func, batched = True,)

test_dataset = dataset["test"]
test_dataset = test_dataset.map(formatting_prompts_func, batched = True)
test_dataset = test_dataset.train_test_split(test_size=0.1)

#设置训练参数
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = True,
    random_state = 3407,
    max_seq_length = max_seq_length,
    use_rslora = False,
    loftq_config = None,
)

trainer = SFTTrainer(
    model = model,
    train_dataset = train_dataset,
    eval_dataset = test_dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    tokenizer = tokenizer,
    compute_metrics=compute_valid_metrics,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 10,
        eval_strategy = "steps",
        eval_steps = 10,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        output_dir = "outputs",
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
    ),
)
#开始训练
trainer.train()

#保存微调模型
model.save_pretrained("lora_model")

#合并模型，保存为16位hf
model.save_pretrained_merged("outputs", tokenizer, save_method = "merged_16bit",)

#合并模型，并量化成4位gguf
#model.save_pretrained_gguf("model", tokenizer, quantization_method = "q4_k_m")
