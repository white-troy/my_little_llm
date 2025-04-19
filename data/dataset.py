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