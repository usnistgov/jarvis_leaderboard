# conda activate chemdata
import argparse
import os
import torch
import numpy as np
import pandas as pd
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoModelForCausalLM,
)
import time
from tqdm import tqdm
from jarvis.db.jsonutils import loadjson

d = loadjson("mmlu_test.json")
device = "cpu"
if torch.cuda.is_available():

    device = torch.device("cuda")
#model_name = "mistralai/Mistral-7B-v0.1"
odel_name = "itsliupeng/llama2_70b_mmlu"
model_name = "meta-llama/Llama-2-7b"
model_name = "meta-llama/Llama-2-7b-hf"
model_name = "meta-llama/Llama-2-13b-hf"
model_name = "meta-llama/Llama-2-7b-chat-hf"
model_name = "itsliupeng/llama2_7b_zh"
if "t5" in model_name:
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)


if "t5" not in model_name:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
# model.to(device)
# model.to(devices[0])
# if num_gpus > 1:
#    model = torch.nn.DataParallel(model)  # Use multiple GPUs
#    #model = torch.nn.DataParallel(model, device_ids=devices)  # Use multiple GPUs

f = open("AI-TextClass-quiz-mmlu_test-test-acc_meta-llama_Llama-2-7b-chat-hf.csv", "w")
#f = open("AI-TextClass-quiz-mmlu_test-test-acc.csv", "w")
f.write("id,target,prediction\n")
# target_labels=[]
# pred_labels=[]
for ii, i in enumerate(tqdm(d)):
   #if ii>10805:
    prompt = i["prompt"]
    label = i["answer"]
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(
        model.device
    )  # .cuda()
    # decoder_input_ids = tokenizer("", return_tensors="pt").input_ids #.cuda()
    # decoder_input_ids = model._shift_right(decoder_input_ids)
    # logits = model(
    #     input_ids=input_ids, decoder_input_ids=decoder_input_ids
    # ).logits.flatten()
    # input_ids.to(device)
    # logits = model(input_ids=input_ids).logits.flatten()
    # logits = model(input_ids=input_ids.to(device)).logits.flatten()
    logits = model(input_ids=input_ids).logits[0, -1]
    probs = (
        torch.nn.functional.softmax(
            torch.tensor(
                [
                    logits[tokenizer("A").input_ids[-1]],
                    logits[tokenizer("B").input_ids[-1]],
                    logits[tokenizer("C").input_ids[-1]],
                    logits[tokenizer("D").input_ids[-1]],
                ]
            ),
            dim=0,
        )
        .detach()
        .cpu()
        .numpy()
    )
    pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]
    # print("prompt",prompt)
    # print("label",label)
    # print("pred",pred)
    # print()
    # target_labels.append(label)
    # pred_labels.append(pred)
    line = i["id"] + "," + label + "," + pred + "\n"
    # print(line)
    f.write(line)
    del input_ids
    del logits
    del probs
f.close()
#!zip AI-TextClass-quiz-mmlu_test-test-acc.csv.zip AI-TextClass-quiz-mmlu_test-test-acc.csv
