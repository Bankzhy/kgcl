import json
import os
import re

from datasets import load_dataset, Dataset
from unsloth import FastLanguageModel
import tqdm
from eval import bleu

#加载模型
max_seq_length = 2048
dtype = None
load_in_4bit = True
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "outputs_clone/checkpoint-1500",
    # model_name = "outputs/checkpoint-26139",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    token = "https://hf-mirror.com"
)
FastLanguageModel.for_inference(model) # Enable native 2x faster inference
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

# if False:
#     from unsloth import FastLanguageModel
#     model, tokenizer = FastLanguageModel.from_pretrained(
#         model_name = "lora_model", # YOUR MODEL YOU USED FOR TRAINING
#         max_seq_length = max_seq_length,
#         dtype = dtype,
#         load_in_4bit = load_in_4bit,
#     )
#     FastLanguageModel.for_inference(model) # Enable native 2x faster inference

# alpaca_prompt = You MUST copy from above!

# from transformers import TextStreamer
# text_streamer = TextStreamer(tokenizer)
def predict(code1, code2):
    input = f"""
            code1: {code1}\n
            code2: {code2}
            """
    inputs = tokenizer(
        [
            alpaca_prompt.format(
                """Please check if the following two code fragments are similar or identical fragments? Answer with "Yes" or "No""",  # instruction
                input,  # input
                "",  # output - leave this blank for generation!
            )
        ], return_tensors="pt").to("cuda")
    output = model.generate(**inputs,max_new_tokens = 128)
    output = tokenizer.decode(output[0])
    match = re.search(r'### Response:\s*(.*?)(?:\n### Response:|\Z)', output, re.DOTALL)
    first_response = match.group(1).strip() if match else None
    first_response = first_response.replace('<|end_of_text|>', "")
    first_response = first_response.split("\n")[0]
    if first_response == "Yes":
        return 1
    else:
        return 0
    return first_response


def compute_valid_metrics(predictions, labels):
    # decoded_preds, decoded_labels = eval_preds

    from sklearn.metrics import recall_score
    recall = recall_score(labels, predictions)
    from sklearn.metrics import precision_score
    precision = precision_score(labels, predictions)
    from sklearn.metrics import f1_score
    f1 = f1_score(labels, predictions)
    result = {
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_f1": float(f1),
    }
    print(result)
    return result


# dataset = load_dataset('json', data_files={'train': 'tl_data/train.json', 'test': 'tl_data/test.json'})
# test_dataset = dataset["test"]
# test_dataset = test_dataset.map(formatting_prompts_func, batched = True)
test_dataset_l = []
json_data = {}
dataset_dir = "bc_data/dataset"
json_file = os.path.join(dataset_dir, "data.jsonl")
with open(json_file, encoding='ISO-8859-1') as jf:
    lines = jf.readlines()
    print("loading dataset:")
    for line in tqdm.tqdm(lines):
        # print(line)
        data = json.loads(line.strip())
        # st, nl = self.parse_kg(data["kg"])
        json_data[data["idx"]] = {
            "code": data["func"],
            # "st": st,
            # "nl": nl,
        }

with open(os.path.join(dataset_dir, "test.txt"), encoding='ISO-8859-1') as f:
    lines = f.readlines()
    for line in tqdm.tqdm(lines):
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

            test_dataset_l.append(
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
test_dataset_l = test_dataset_l[:42000]
test_dataset = Dataset.from_list(test_dataset_l)
test_dataset = test_dataset.map(formatting_prompts_func, batched = True,)

preds = []
labels = []

for data in tqdm.tqdm(test_dataset):
    pred = predict(data["code1"], data["code2"])
    preds.append(pred)
    if data["label"] == "Yes":
        labels.append(1)
    else:
        labels.append(0)

compute_valid_metrics(preds, labels)
