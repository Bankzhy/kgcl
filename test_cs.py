import re

from datasets import load_dataset
from unsloth import FastLanguageModel
import tqdm
from eval import bleu

#加载模型
max_seq_length = 2048
dtype = None
load_in_4bit = True
model, tokenizer = FastLanguageModel.from_pretrained(
    # model_name = "unsloth/codellama-7b-bnb-4bit",
    model_name = "outputs/checkpoint-26139",
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
    inputs       = examples["code"]
    outputs      = examples["doc"]
    texts = []
    for input, output in zip(inputs, outputs):
        instruction = "Please summarize the following code."
        # 必须添加EOS_TOKEN，否则无限生成
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
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
def predict(code):
    inputs = tokenizer(
        [
            alpaca_prompt.format(
                "Please summarize the following code.",  # instruction
                code,  # input
                "",  # output - leave this blank for generation!
            )
        ], return_tensors="pt").to("cuda")
    output = model.generate(**inputs,max_new_tokens = 128)
    output = tokenizer.decode(output[0])
    match = re.search(r'### Response:\s*(.*?)(?:\n### Response:|\Z)', output, re.DOTALL)
    first_response = match.group(1).strip() if match else None
    first_response = first_response.replace('<|end_of_text|>', "")
    return first_response


def compute_valid_metrics(decoded_preds, decoded_labels):
    # decoded_labels, decoded_preds = decode_preds(eval_preds)
    refs = [ref.strip().split() for ref in decoded_labels]
    cans = [can.strip().split() for can in decoded_preds]
    result = {}
    result.update(bleu(references=refs, candidates=cans))
    print(result)
    return result


dataset = load_dataset('json', data_files={'train': 'tl_data/train.json', 'test': 'tl_data/test.json'})
test_dataset = dataset["test"]
test_dataset = test_dataset.map(formatting_prompts_func, batched = True)

preds = []
labels = []

for data in tqdm(test_dataset):
    pred = predict(data["code"])
    preds.append(pred)
    labels.append(data["doc"])

compute_valid_metrics(preds, labels)
