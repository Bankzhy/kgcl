import csv
import json
import os

import pandas as pd
from tqdm import tqdm
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset, Dataset
from transformers.trainer_utils import get_last_checkpoint
from eval import bleu

#加载模型
max_seq_length = 2048
dtype = None
load_in_4bit = True
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/codellama-7b-bnb-4bit",
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
code1: {}\n
code2: {}
### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token # 必须添加 EOS_TOKEN
def formatting_prompts_func(examples):
    # inputs       = examples["code"]
    codes1       = examples["code1"]
    codes2       = examples["code2"]
    outputs      = examples["label"]
    texts = []
    for code1, code2, output in zip(codes1, codes2, outputs):
        instruction = """Please check if the following two code fragments are similar or identical fragments? Answer with "Yes" or "No"""
        # 必须添加EOS_TOKEN，否则无限生成
        text = alpaca_prompt.format(instruction, code1, code2, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

#hugging face数据集路径
# train_dataset = load_dataset("code-search-net/code_search_net", split=f"train[:100000]", trust_remote_code=True)
# dataset = load_dataset('json', data_files={'train': 'tl_data/train.json', 'test': 'tl_data/test.json'})
# train_dataset = dataset["train"]
# train_dataset = train_dataset.map(formatting_prompts_func, batched = True,)

train_dataset_l = []
json_data = {}
dataset_dir = "bc_data/dataset"
json_file = os.path.join(dataset_dir, "data.jsonl")
with open(json_file, encoding='ISO-8859-1') as jf:
    lines = jf.readlines()
    print("loading dataset:")
    for line in tqdm(lines):
        # print(line)
        data = json.loads(line.strip())
        # st, nl = self.parse_kg(data["kg"])
        json_data[data["idx"]] = {
            "code": data["func"],
            # "st": st,
            # "nl": nl,
        }

with open(os.path.join(dataset_dir, "train.txt"), encoding='ISO-8859-1') as f:
    lines = f.readlines()
    for line in tqdm(lines):
        try:
            ll = line.split("\t")
            if ll[0] not in json_data.keys() or ll[1] not in json_data.keys():
                continue
            code1 = json_data[ll[0]]["code"]
            code2 = json_data[ll[1]]["code"]

            if ll[2].strip() == "1":
                label = "Yes"
            else:
                label = "No"

            train_dataset_l.append(
                {
                    "code1": code1,
                    "code2": code2,
                    "label": label,
                }
            )

        except Exception as e:
            print(e)
            continue
# Convert list of dicts to a Dataset
train_dataset = Dataset.from_list(train_dataset_l)
train_dataset = train_dataset.map(formatting_prompts_func, batched = True,)


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
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    tokenizer = tokenizer,
    # compute_metrics=compute_valid_metrics,
    args = TrainingArguments(
        per_device_train_batch_size = 8,
        gradient_accumulation_steps = 4,
        warmup_steps = 10,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        output_dir = "outputs_clone",
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        save_total_limit=3
    ),
)
#开始训练
# trainer.train()
last_checkpoint = get_last_checkpoint("outputs")
train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
#保存微调模型
model.save_pretrained("lora_model_cl")

#合并模型，保存为16位hf
# model.save_pretrained_merged("outputs", tokenizer, save_method = "merged_16bit",)

#合并模型，并量化成4位gguf
# model.save_pretrained_gguf("model", tokenizer, quantization_method = "q4_k_m")
