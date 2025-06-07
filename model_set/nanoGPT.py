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

def build_rope_cache(seq_len, head_dim, device, base=10000):
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    position = torch.arange(seq_len, device=device).float()
    freqs = torch.einsum("i,j->ij", position, theta)  # (T, D//2)
    
    emb = torch.cat([freqs, freqs], dim=-1)  # (T, D)
    cos = torch.cos(emb)[None, None, :, :]   # (1, 1, T, D)
    sin = torch.sin(emb)[None, None, :, :]
    return cos, sin

def apply_rotary_pos_emb(q, k, cos, sin):
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., :x.shape[-1] // 2]), dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

    
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

class Attention_ROPE(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.n_heads = model_config.n_head
        self.head_dim = model_config.n_embed // model_config.n_head

        self.q_proj = nn.Linear(model_config.n_embed,model_config.n_embed)
        self.k_proj = nn.Linear(model_config.n_embed,model_config.n_embed)
        self.v_proj = nn.Linear(model_config.n_embed,model_config.n_embed)
        self.out_proj = nn.Linear(model_config.n_embed, model_config.n_embed)
        self.attn_dropout = nn.Dropout(model_config.drop_out)
        self.resid_dropout = nn.Dropout(model_config.drop_out)
    
    def forward(self, x, rope_cache, past_kv=None, attn_mask=None, use_cache=False):
        B, T, C = x.shape #batch seq_len
        H, D = self.n_heads, self.head_dim # head head_dim

        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        q = q.view(B, T, H, D).transpose(1, 2) # B H T D
        k = k.view(B, T, H, D).transpose(1, 2)
        v = v.view(B, T, H, D).transpose(1, 2)

        # if past_kv is not None:
            # print(f"k的属性是：{past_kv[0].shape}")
            # print(f"v的属性是：{past_kv[1].shape}")
        cos, sin = rope_cache
        # 根据past_kv决定旋转位置偏移
        # seq_start = past_kv[0].size(2) if past_kv is not None else 0
        # print(f"seq_start: {seq_start}, T: {T}")
        seq_start = 0
        cos = cos[:, :, seq_start:seq_start + T, :]
        sin = sin[:, :, seq_start:seq_start + T, :]
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_kv is not None:
            pk, pv = past_kv
            k = torch.cat([pk, k], dim=2) # 在T拼
            v = torch.cat([pv, v], dim=2)

        if use_cache:
            present_kv = (k, v)
        else:
            present_kv = None
        # print(f"注意力中present_kv的k形状: {present_kv[0].shape if present_kv else None}")
        S = k.size(2)

        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(D)
        # Causal mask
        causal_mask = torch.tril(torch.ones((T, S), device=x.device)).view(1, 1, T, S)
        attn_scores = attn_scores.masked_fill(causal_mask == 0, float("-inf"))

        # Pad attention mask
        if attn_mask is not None:
            extended_mask = attn_mask.view(B, 1, 1, -1).bool()
            attn_scores = attn_scores.masked_fill(~extended_mask, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Final output
        out = (attn_weights @ v).transpose(1, 2).reshape(B, T, H * D)
        out = self.out_proj(out)
        out = self.resid_dropout(out)

        return out, present_kv

class Block(nn.Module):
    def __init__(self,model_config):
        super().__init__()
        head_size = model_config.n_embed//model_config.n_head
        #self.attn = MultiHeadAttention(model_config)
        self.attn = Attention_ROPE(model_config)
        self.ffn = FeedForward(model_config)
        self.ln1 = nn.LayerNorm(model_config.n_embed)
        self.ln2 = nn.LayerNorm(model_config.n_embed)

    # def forward(self,x):
    #     x = x + self.attn(self.ln1(x))
    #     x = x + self.ffn(self.ln2(x))
    #     return x
    def forward(self, x, rope_cache, past_kv=None, attn_mask=None, use_cache=False):
        attn_out, present_kv = self.attn(self.ln1(x), rope_cache, past_kv, attn_mask, use_cache)
        # print(f"block中present_kv的k形状: {present_kv[0].shape if present_kv else None}")
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, present_kv

# 构建模型
class NANOGPT(nn.Module):
    def __init__(self,model_config):
        super().__init__()
        # print(model_config.__dict__.keys())
        self.token_embedding_table = nn.Embedding(model_config.vocab_size,model_config.n_embed)
        #self.position_embedding_tabel = nn.Embedding(model_config.block_size,model_config.n_embed) #不使用绝对位置编码

        self.blocks = nn.Sequential(
            *[Block(model_config) for _ in range(model_config.n_layer)]
        )
        self.ln_final = nn.LayerNorm(model_config.n_embed)
        self.lm_head = nn.Linear(model_config.n_embed, model_config.vocab_size, bias=False)
        self.config = model_config

        # 通过tiktoken指出停止符
        enc = tiktoken.get_encoding("gpt2")
        self.eos_token = enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]

        max_seq = model_config.block_size * 2  # support some lookahead
        rope_cos, rope_sin = build_rope_cache(max_seq, model_config.n_embed // model_config.n_head, device='cpu')
        self.register_buffer("rope_cos", rope_cos, persistent=False)
        self.register_buffer("rope_sin", rope_sin, persistent=False)

        self.apply(self.init_weights)

    # 初始化权重，使用正态分布
    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, past_kvs=None, attn_mask=None, use_cache=False):
        B, T = idx.size()
        x = self.token_embedding_table(idx)
        # print(f"经过embedding的idx的形状为{x.shape}")
        seq_offset = past_kvs[0][0].size(2) if past_kvs else 0
        rope_cache = (self.rope_cos[:, :, seq_offset:seq_offset + T, :],
                      self.rope_sin[:, :, seq_offset:seq_offset + T, :])
        # rope_cache = (self.rope_cos, self.rope_sin)


        presents = []
        for i, block in enumerate(self.blocks):
            past_kv = past_kvs[i] if past_kvs else None
            x, present_kv = block(x, rope_cache, past_kv, attn_mask, use_cache)
            if use_cache:
                presents.append(present_kv)

        x = self.ln_final(x)
        logits = self.lm_head(x)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        # print(f"模型中最后得到的present_kv的k形状: {presents[0][0].shape}")
        if use_cache:
            return logits, None, presents
        else:
            return logits, None, None

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, stop_eos=True, use_cache=True):
        past_kvs = None
        generated = idx  # 保留完整生成序列，idx只用于初始 prompt

        for step in range(max_new_tokens):
            # print(f"Step {step + 1}/{max_new_tokens}, 当前 generated 的形状: {generated.shape}")

            idx_cond = generated if past_kvs is None else generated[:, -1:]
            # print(f"idx_cond 的形状: {idx_cond.shape}")

            attn_mask = torch.ones_like(idx_cond, dtype=torch.bool, device=generated.device)

            logits, _, present_kvs = self.forward(
                idx_cond, past_kvs=past_kvs,
                attn_mask=attn_mask, use_cache=use_cache
            )

            # print(f"logits 的形状: {logits.shape}, 得到的 present_kvs 的层数: {len(present_kvs)}")

            if use_cache:
                past_kvs = present_kvs
                # if past_kvs is None:
                #     past_kvs = present_kvs
                # else:
                #     # 按层拼接 (k, v)，dim=2 为 seq_len 维度
                #     past_kvs = tuple(
                #         (
                #             torch.cat([past_k[0], curr_k[0]], dim=2),
                #             torch.cat([past_k[1], curr_k[1]], dim=2)
                #         )
                #         for past_k, curr_k in zip(past_kvs, present_kvs)
                #     )

                print(f"Step {step}, past_kv len: {[kv[0].shape[2] for kv in past_kvs]}")

            # 采样
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            if stop_eos and idx_next.item() == self.eos_token:
                print("生成终止：遇到 <|endoftext|> token")
                break

            generated = torch.cat([generated, idx_next], dim=1)
        # print(generated)
        return generated