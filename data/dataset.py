import tiktoken
import json
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

def get_chatml_tokenizer(name="gpt2"):
    base_enc = tiktoken.get_encoding(name)
    base_vocab_size = 50257 
    special_tokens = {
        "<|im_start|>": base_vocab_size,
        "<|im_end|>": base_vocab_size + 1,
    }

    # 构建新编码器
    enc = tiktoken.Encoding(
        name=base_enc.name,
        pat_str=base_enc._pat_str,
        mergeable_ranks=base_enc._mergeable_ranks,
        special_tokens={**base_enc._special_tokens, **special_tokens},
    )
    return enc

class TextDataset(Dataset):
    def __init__(self,data_path,block_size,max_lines,tokenizer_name="gpt2"):
        self.block_size = block_size
        self.max_lines = max_lines
        # 初始化tokenizer
        self.enc = get_chatml_tokenizer(tokenizer_name)
        self.eos_token = self.enc.encode("<|endoftext|>",allowed_special={"<|endoftext|>"})[0]
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
                    # json.loads(line.strip())  # 仅验证不保存结果
                    json_line = json.loads(line.strip())
                    
                    # 检查是否存在必需的'text'字段
                    if 'text' not in json_line:
                        print(f"第{i+1}行缺少必需的'text'字段")
                        continue
                    raw_data.append(json_line['text'])
                except json.JSONDecodeError as e:
                    print(f"警告：跳过第{i+1}行- {str(e)}")
                    continue

        
        full_encoded = []
        for text in tqdm(raw_data,desc='编码数据'):
            encoded_text = self.enc.encode(text, allowed_special={"<|im_start|>", "<|im_end|>"})
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
        self.enc = get_chatml_tokenizer(tokenizer_name)

        # 特殊token
        self.im_start = "<|im_start|>"
        self.im_end = "<|im_end|>"
        self.im_start_ids = self.enc.encode(self.im_start, allowed_special={self.im_start})
        self.im_end_ids = self.enc.encode(self.im_end, allowed_special={self.im_end})

        # 打印debug
        print(f"[Init] <|im_start|> tokens: {self.im_start_ids}")
        print(f"[Init] <|im_end|> tokens: {self.im_end_ids}")

        self.encoded_data = []
        self.loss_masks = []
        self.load_and_process_data(data_path)

    def format_conversations(self, conversations):
        parts = []
        for turn in conversations:
            role = turn["role"]
            content = turn["content"]
            parts.append(f"{self.im_start}{role}\n{content}{self.im_end}\n")
        return ''.join(parts)

    def load_and_process_data(self, path):
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
                    if not any(turn["role"] == "assistant" for turn in data['conversations']):
                        continue
                    full_text = self.format_conversations(data['conversations'])
                    encoded = self.enc.encode(full_text, allowed_special={self.im_start, self.im_end})
                    mask = self._generate_loss_mask(encoded)
                    self.encoded_data.append(encoded)
                    self.loss_masks.append(mask)
                except Exception as e:
                    print(f"跳过第{i+1}行（出错）: {str(e)}")
                    continue

    def _generate_loss_mask(self, input_ids):
        mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            # 查找 <|im_start|>
            if input_ids[i:i+len(self.im_start_ids)] == self.im_start_ids:
                role_start = i + len(self.im_start_ids)

                # 从 role_start 开始找 role 的字符串（例如 'assistant'）
                remaining = input_ids[role_start:]
                for role in ['assistant', 'user']:
                    role_ids = self.enc.encode(role + "\n")
                    if remaining[:len(role_ids)] == role_ids:
                        if role == 'assistant':
                            content_start = role_start + len(role_ids)
                            j = content_start
                            while j < len(input_ids):
                                if input_ids[j:j+len(self.im_end_ids)] == self.im_end_ids:
                                    break
                                j += 1
                            for k in range(content_start, j):
                                mask[k] = 1
                            i = j + len(self.im_end_ids)
                        else:
                            # 跳过 user 的部分
                            end_idx = role_start + len(role_ids)
                            while end_idx < len(input_ids):
                                if input_ids[end_idx:end_idx+len(self.im_end_ids)] == self.im_end_ids:
                                    break
                                end_idx += 1
                            i = end_idx + len(self.im_end_ids)
                        break
                else:
                    # 没有匹配到对应，跳过
                    i += 1
            else:
                i += 1
        return mask


    def __len__(self):
        return len(self.encoded_data)

    def __getitem__(self, idx):
        chunk = self.encoded_data[idx]
        mask = self.loss_masks[idx]

        pad_token = self.enc.eot_token  # 对应 <|endoftext|>
        pad_len = (self.block_size + 1) - len(chunk)
        if pad_len > 0:
            chunk += [pad_token] * pad_len
            mask += [0] * pad_len

        chunk = chunk[:self.block_size + 1]
        mask = mask[:self.block_size + 1]

        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        lm = torch.tensor(mask[1:], dtype=torch.long)

        return x, y, lm

    def encode(self, text):
        return self.enc.encode(text)

    def decode(self, ids):
        return self.enc.decode(ids)