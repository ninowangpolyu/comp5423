import os
from transformers import BartForConditionalGeneration, BertTokenizer, BartConfig
import torch
import json
from typing import List, Tuple, Optional
import time

model_path = './model/model.pt'
output_dir = './model'
print("Loading model...")
print("Model path:", model_path)
model = torch.load(model_path, map_location=torch.device('cpu'))
print("Model loaded.")
tokenizer = BertTokenizer.from_pretrained(output_dir, do_lower_case=False)
print("Tokenizer loaded.")

def get_resp(encoded_input):
    with torch.no_grad():
        output = model.bart.generate(
            input_ids=encoded_input['input_ids'].to('cpu'),
            attention_mask=encoded_input['attention_mask'].to('cpu'),
            num_beams=1,
            max_length=128,
            # repetition_penalty=2.5,
            # length_penalty=1.5,
            early_stopping=True,
        )
        for g in output:
            resp = tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            return resp

def preprocess_trained_model(chatbot: List[Tuple[str, str]]):
    text_prompt = ""
    for i in range(len(chatbot)):
        text_prompt += str(chatbot[i][0]) + ' '
        if i != len(chatbot)-1:
            text_prompt += str(chatbot[i][1]) + ' '
    print(text_prompt)
    encoded_input = tokenizer(text_prompt, padding=True, truncation=True, return_tensors="pt")
    res_trained = get_resp(encoded_input)
    res_trained = res_trained.replace(" ", "")

    chatbot[-1][1] = ""
    for chunk in res_trained:
        for i in range(0, len(chunk), 10):
            section = chunk[i:i + 10]
            chatbot[-1][1] += section
            time.sleep(0.01)
            yield chatbot

    # if __name__ == '__main__':
    # model_path = './model/model.pt'
    # output_dir = './model'
    # input_file = './data/test.txt'
    # tokenizer = BertTokenizer.from_pretrained(output_dir, do_lower_case=False)
    #
    # input_all = []
    # while True:
    #     # 读取用户输入
    #     user_input = input("请输入内容: ")
    #     input_all.append(user_input)
    #     conversation_string = "".join(input_all)
    #     encoded_input = tokenizer(conversation_string, padding=True, truncation=True, return_tensors="pt")
    #     get_resp(encoded_input)
