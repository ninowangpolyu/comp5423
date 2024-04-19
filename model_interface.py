import os
from transformers import BartForConditionalGeneration, BertTokenizer, BartConfig
import torch
import json
import msvcrt  # 用于在Windows上检测按键

def get_resp(encoded_input,model):
    with torch.no_grad():
        output = model.bart.generate(
            input_ids=encoded_input['input_ids'].to('cpu'),
            attention_mask=encoded_input['attention_mask'].to('cpu'),
            num_beams=1,
            max_length=32,
            # repetition_penalty=2.5,
            # length_penalty=1.5,
            early_stopping=True,
        )
        for g in output:
            resp = tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            return resp

if __name__ == '__main__':
    model_path = './model/model.pt'
    output_dir = './model'
    input_file = './data/test.txt'
    model = torch.load(model_path,map_location=torch.device('cpu'))
    tokenizer = BertTokenizer.from_pretrained(output_dir, do_lower_case=False)

    input_all = []
    while True:
        if msvcrt.kbhit():  # 检查是否有按键按下
            key = msvcrt.getch()  # 获取按下的键
            print("对话已退出")
            break  # 退出循环
        # 读取用户输入
        user_input = input("请输入内容: ")
        input_all.append(user_input)
        conversation_string = "".join(input_all)
        encoded_input = tokenizer(conversation_string, padding=True, truncation=True, return_tensors="pt")
        get_resp(encoded_input)
