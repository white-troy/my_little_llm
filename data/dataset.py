import tiktoken
import json
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class TextDataset(Dataset):
    def __init__(self,data_path,block_size,max_lines,tokenizer_name="gpt2"):
        self.block_size = block_size
        self.max_lines = max_lines
        # 初始化tokenizer
        self.enc = tiktoken.get_encoding(tokenizer_name)
        self.eos_token = self.enc.encode(
            "<|endoftext|>",
            allowed_special={"<|endoftext|>"}
        )[0]
        # 预处理数据
        self.encoded_data = []
        self.load_and_process_data(data_path)

    def load_and_process_data(self,path):
        raw_data = []
        if not (path.endswith('.json') or path.endswith('.jsonl')):
            raise ValueError(f"文件 {path} 不是JSON格式文件(扩展名需为.json或.jsonl)")
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                if i >= self.max_lines:
                    break
                try:
                    # 先验证是否为有效JSON行
                    json.loads(line.strip())  # 仅验证不保存结果
                    json_line = json.loads(line.strip())
                    
                    # 检查是否存在必需的'text'字段
                    if 'text' not in json_line:
                        raise KeyError(f"第{i+1}行缺少必需的'text'字段")
                        
                    text = json_line['text']
                    raw_data.append(text)
                except json.JSONDecodeError as e:
                    print(f"警告：跳过第{i+1}行- {str(e)}")
                    continue

        
        full_encoded = []
        for text in tqdm(raw_data,desc='编码数据'):
            encoded_text = self.enc.encode(text)
            full_encoded.extend(encoded_text + [self.eos_token])

        for i in range(0, len(full_encoded),self.block_size):
            chunk = full_encoded[i:i+self.block_size + 1] # +1的原因：第0至blocksize为input，第1至blocksize+1为label
            # 填充不足长度的块
            if len(chunk) < self.block_size + 1:
                chunk = chunk + [self.eos_token] * (self.block_size + 1 - len(chunk))
            
            self.encoded_data.append(chunk)

    def __len__(self):
        return len(self.encoded_data)
    
    def __getitem__(self, idx):
        chunk = self.encoded_data[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y
    
    def encode(self, text):
        """将文本编码为token IDs"""
        return self.enc.encode(text)
    
    def decode(self, ids):
        """将token IDs解码为文本"""
        return self.enc.decode(ids)
    
class SFTDataset(Dataset):
    def __init__(self, data_path, block_size, max_lines, tokenizer_name="gpt2"):
        self.block_size = block_size
        self.max_lines = max_lines
        self.enc = tiktoken.get_encoding(tokenizer_name)
        self.eos_token = self.enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]
        self.encoded_data = []
        self.loss_masks = []
        self.load_and_process_data(data_path)

    def format_conversations(self, conversations):
        prompt = ""
        for turn in conversations:
            role = turn["role"]
            if role == "user":
                prompt += "<|user|>" + turn["content"]
            elif role == "assistant":
                prompt += "<|assistant|>" + turn["content"]
        return prompt + "<|endoftext|>"

    def load_and_process_data(self, path):
        raw_data = []
        if not (path.endswith('.json') or path.endswith('.jsonl')):
            raise ValueError("数据格式必须为 .json 或 .jsonl")

        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= self.max_lines:
                    break
                try:
                    data = json.loads(line.strip())
                    if 'conversations' not in data:
                        continue
                    raw_data.append(data['conversations'])
                except json.JSONDecodeError:
                    print(f"警告：跳过第{i+1}行- {str(e)}")
                    continue

        for conv in tqdm(raw_data, desc="编码对话"):
            full_text = self.format_conversations(conv)
            encoded = self.enc.encode(full_text, allowed_special={"<|user|>", "<|assistant|>", "<|endoftext|>"})

            for i in range(0, len(encoded), self.block_size + 1):
                chunk = encoded[i:i+self.block_size+1]
                if len(chunk) < self.block_size + 1:
                    chunk += [self.eos_token] * (self.block_size + 1 - len(chunk))
                self.encoded_data.append(chunk)

                # 创建loss_mask：仅mask assistant部分
                mask = [0] * len(chunk)
                j = 0
                while j < len(chunk):
                    # 找到 <|assistant|>
                    if chunk[j:j+3] == self.enc.encode("<|assistant|>", allowed_special={"<|assistant|>"}):
                        start = j + 3
                        end = start
                        while end < len(chunk):
                            # 找到下一个特殊token或结尾
                            if chunk[end:end+3] == self.enc.encode("<|user|>", allowed_special={"<|user|>"}) or \
                               chunk[end:end+1] == [self.eos_token]:
                                break
                            end += 1
                        for k in range(start, end):
                            mask[k] = 1
                        j = end
                    else:
                        j += 1

                self.loss_masks.append(mask)

    def __len__(self):
        return len(self.encoded_data)

    def __getitem__(self, idx):
        chunk = self.encoded_data[idx]
        loss_mask = self.loss_masks[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        lm = torch.tensor(loss_mask[1:], dtype=torch.long)
        return x, y, lm

    def encode(self, text):
        return self.enc.encode(text)

    def decode(self, ids):
        return self.enc.decode(ids)