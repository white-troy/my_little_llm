'''将txt or json格式的数据集文件制作为数据集文件'''
import os
import tiktoken
import numpy as np
import json
from tqdm import tqdm

# openai tokenizer的库，tiktoken
# 使用方法 https://zhuanlan.zhihu.com/p/629776230

def split_txt(input_file:str):
    if not input_file.endswith('txt'):
        raise ValueError('not txt file')
    input_file_path = os.path.join(os.path.dirname(__file__), input_file)
    with open(input_file_path, 'r', encoding='utf-8') as f:
        data = f.read()
    n = len(data)
    train_data = data[:int(n * 0.9)]
    val_data = data[int(n * 0.9):]

    # encode with tiktoken gpt3 bpe
    enc = tiktoken.get_encoding("cl100k_base")

    train_ids = enc.encode_ordinary(train_data)
    val_ids = enc.encode_ordinary(val_data)
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")

    # export to bin files
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
    val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

def process_wiki_json(input_file):
    # encode with tiktoken gpt2 bpe
    enc = tiktoken.get_encoding("cl100k_base")
    file = os.path.join(os.path.dirname(__file__), input_file)
    with open(file,'r',encoding='utf-8') as f:
        data=json.load(f)

    # 划分数据集
    n = len(data)
    train_data = data[:int(n*0.9)]
    val_data = data[int(n*0.9):]
    train_doc_ids = []
    for line in tqdm(train_data):
        text=line['completion']
        # chat glm的tokenizer，这里使用tiktoken
        # text_id=tokenizer.encode(text,add_special_tokens=False)
        # text_id.append(tokenizer.special_tokens['<eos>'])
        text_id = enc.encode_ordinary(text)
        end_of_text_token = '<endOfText>'
        end_of_text_id = enc.encode_ordinary(end_of_text_token)
        text_id.extend(end_of_text_id)
        if len(text_id)>5:
            train_doc_ids+=text_id
    val_doc_ids = []
    for line in tqdm(val_data):
        text=line['completion']
        # chat glm的tokenizer，这里使用tiktoken
        # text_id=tokenizer.encode(text,add_special_tokens=False)
        # text_id.append(tokenizer.special_tokens['<eos>'])
        text_id = enc.encode_ordinary(text)
        end_of_text_token = '<endOfText>'
        end_of_text_id = enc.encode_ordinary(end_of_text_token)
        text_id.extend(end_of_text_id)
        if len(text_id)>5:
            val_doc_ids+=text_id
    train_ids = np.array(train_doc_ids,dtype=np.uint16)
    val_ids = np.array(val_doc_ids,dtype=np.uint16)
    # Save to bin files
    train_ids.tofile(os.path.join(os.path.dirname(__file__), './wiki/train.bin'))
    val_ids.tofile(os.path.join(os.path.dirname(__file__), './wiki/val.bin'))

if __name__ == "__main__":
    # txt_file = 'sherlock/sherlock.txt'
    # split_txt(txt_file)
    wiki_json = 'wiki/wikipedia-cn-20230720-filtered.json'
    process_wiki_json(wiki_json)
