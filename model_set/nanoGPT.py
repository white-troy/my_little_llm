import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken

class SingelHeadAttention(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.query = nn.Linear(model_config.n_embed,model_config.head_size)
        self.key = nn.Linear(model_config.n_embed,model_config.head_size)
        self.value = nn.Linear(model_config.n_embed,model_config.head_size)
        self.head_size = model_config.head_size

        # register_buffer注册的Tensor会被标记为requires_grad=False
        # 因此不会参与梯度反向传播，从而节省一半的内存
        self.register_buffer(
            'attention_mask',
            torch.tril(
                torch.ones(model_config.block_size,model_config.block_size)
            )
        )

        self.dropout = nn.Dropout(model_config.drop_out)

    def forward(self,x):
        bz,sq_len,hidden_size = x.size()
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        weight = q @ k.transpose(-2,-1) # 转置矩阵
        weight = weight.masked_fill(
            self.attention_mask[:sq_len,:sq_len] == 0,float('-inf') # 将掩膜中等于0的位置的权重设置为负无穷
        )/math.sqrt(self.head_size) # 开根号
        weight = F.softmax(weight,dim=-1)
        weight = self.dropout(weight)
        out = weight @ v
        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self,model_config):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                SingelHeadAttention(model_config)
                for _ in range(model_config.n_head)
            ]
        )
        self.proj = nn.Linear(model_config.n_embed,model_config.n_embed) # 投影
        self.dropout = nn.Dropout(model_config.drop_out)

    def forward(self,x):
        output = torch.cat([h(x) for h in self.heads],dim= -1)
        output = self.proj(output)
        output = self.dropout(output)
        return output
    
class FeedForward(nn.Module):
    def __init__(self,model_config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(model_config.n_embed,4*model_config.n_embed),
            nn.GELU(),
            nn.Linear(4*model_config.n_embed,model_config.n_embed),
            nn.Dropout(model_config.drop_out)
        )
    
    def forward(self,x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self,model_config):
        super().__init__()
        head_size = model_config.n_embed//model_config.n_head
        self.attn = MultiHeadAttention(model_config)
        self.ffn = FeedForward(model_config)
        self.ln1 = nn.LayerNorm(model_config.n_embed)
        self.ln2 = nn.LayerNorm(model_config.n_embed)

    def forward(self,x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

# 构建模型
class NANOGPT(nn.Module):
    def __init__(self,model_config):
        super().__init__()
        # print(model_config.__dict__.keys())
        self.token_embedding_table = nn.Embedding(model_config.vocab_size,model_config.n_embed)
        self.position_embedding_tabel = nn.Embedding(model_config.block_size,model_config.n_embed)
        self.blocks = nn.Sequential(
            *[Block(model_config) for _ in range(model_config.n_layer)]
        )
        self.ln_final = nn.LayerNorm(model_config.n_embed)
        self.lm_head = nn.Linear(model_config.n_embed, model_config.vocab_size, bias=False)
        self.apply(self.init_weights)

        # 通过tiktoken指出停止符
        enc = tiktoken.get_encoding("gpt2")
        self.eos_token = enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]

    # 初始化权重，使用正态分布
    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self,idx,targets=None):
        batch,sq_len = idx.size()
        token_emb = self.token_embedding_table(idx)

        pos_emb = self.position_embedding_tabel(
            torch.arange(sq_len,device=idx.device)
        )

        x = token_emb + pos_emb # 位置编码与输入的编码相加
        x = self.blocks(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)   # shape is (batch, seq_len, vocab_size)
        
        if targets is None:
            loss = None
        else:
            batch, sq_len, vocab_size = logits.size()
            logits = logits.view(batch * sq_len, vocab_size)
            targets = targets.view(batch * sq_len)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens,temperature=1.0,stop_eos=True):
        # idx 形状为(B, T)的张量，表示当前上下文（B为批大小，T为序列长度）。
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            logits,_ = self(idx_cond) # 模型前向传播，输出形状(B, T, vocab_size)
            logits = logits[:,-1,:] # 取最后一个时间步的logits (B, vocab_size)，因为自回归生成是逐步进行的

            if temperature != 1.0:
                logits = logits / temperature
            
            probs = F.softmax(logits,dim=-1) # 通过softmax转换为概率分布

            idx_next = torch.multinomial(probs, num_samples=1)  # 从分布中采样 (B, 1)

            if stop_eos and idx_next.item() == self.eos_token:
                break

            idx = torch.cat((idx, idx_next), dim=1)             # 拼接新token (B, T+1)
        return idx
    
