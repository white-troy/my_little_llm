import os
import numpy as np
import torch
import torch.nn as nn
import math
from model_set.base_model import Model_args,GPT
from tqdm import tqdm

# ------------------------------------
# 模型参数设置
block_size = 1024 # 窗口大小GPT2为1024 即上下文长度（输入的文本块的大小）
n_layer = 12 # 模型层数
n_head = 12 # 注意力头
n_embed = 768 # 嵌入向量的维度，即embedding层
bias = False # 是否使用偏置项
dropout = 0.0

# ------------------------------------
# 数据集设置
gradient_accumulation_steps = 20 # used to simulate larger batch sizes
batch_size = 1 # if gradient_accumulation_steps > 1, this is the micro-batch size

dataset_path = './data/wiki'
init_from = 'resume' # 'scratch' or 'resume' # 从头训练还是继续
checkpoint_save_dir = './chat/checkpoints'
if not os.path.exists(checkpoint_save_dir):
    os.makedirs(checkpoint_save_dir,exist_ok=True)

# ------------------------------------
# 训练相关参数
eval_iters = 200 # 评估间隔
eval_interval = 200 # 每n步eval和保存checkpoint一次
learning_rate = 6e-3 # 初始学习率
warmup_iters = 10 # warmup的迭代次数
lr_decay_iters = 60000 # 学习率降低的步数，最好接近max_iter
min_lr = 6e-6
# 优化器参数
max_iters = 60000 # 模型的最大的参数更新次数
weight_decay = 1e-1
betas = (0.9,0.95)
grad_clip = 1.0 # 梯度裁剪

# system
device = 'cuda'
device_type = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
# 检查cuda是否支持bfloat16数据类型

ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)
# torch.amp.autocast混合精度

# 获取模型参数
model_args = dict(n_layer=n_layer, n_head=n_head, n_embed=n_embed, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout)

def get_batch(split,data_dir):
    # nanogpt作者说，memmap每个batch都要用一次，这样才不会内存泄漏
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    
    ix = torch.randint(len(data)-block_size,(batch_size,)) # 
    # torch.randint(a, b, (size,))即在（a,b）范围内生成size个随机数
    x = torch.stack([torch.from_numpy((data[i:i+block_size].astype(np.int64))) for i in ix]) # 根据ix从data里面取x,y
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size].astype(np.int64))) for i in ix])
    # torch.stack(inputs, dim=0),dim为拼接的新的维度

    x,y = x.pin_memory().to(device,non_blocking=True),y.pin_memory().to(device,non_blocking=True)
    # pin_memory()将张量锁定在内存中，non_blocking=True数据传输是非阻塞的，不会阻塞当前线程
    return x,y


def estimate_loss(data_dir,model):
    model.eval() # eval不计算梯度
    out = {}
    for split in ['train','val']:
        # 这里是训练集和验证集都算一下loss
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y = get_batch(split,data_dir)
            with ctx:
                _,loss = model(X,Y) # x,targets
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # 退出时回到train的模式
    return out


# nanogpt使用cos做learning rate的下降
def get_lr(now_iter):
    if(now_iter<warmup_iters):#(1)warmup阶段，线性上升
        return learning_rate*now_iter/warmup_iters
    elif(now_iter>lr_decay_iters):#(2)超过decay，到min了
        return min_lr
    else:# (3)在warmup和decay之间，用cos做lr衰减
        rate = (now_iter-warmup_iters)/(lr_decay_iters-warmup_iters)
        # 计算所占比例(0,1)
        return min_lr + 0.5*(1.0+math.cos(math.pi*rate)) * (learning_rate-min_lr)


def train(split,dataset_path,model_args):
    # dataloader
    data_dir = os.path.join(dataset_path)
    X, Y = get_batch('train',data_dir)

    best_val_loss = 1e9

    tqdm_info = None

    # TODO：lora代码处这里添加from pretrain
    assert init_from == 'scratch' or init_from == 'resume'
    if init_from == 'scratch':
        print("从头训练模型")
        model_args['vocab_size'] = 100256  # clk_100base
        gpt_args = Model_args(**model_args)
        model = GPT(gpt_args)  # 创建模型
        tqdm_info = tqdm(range(max_iters))

    elif init_from == 'resume':  # 继续训练
        print("继续训练模型")
        ckpt_path = os.path.join(checkpoint_save_dir, 'checkpoint.pt')  # 读取checkpoint路径
        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint['model_args']  # 从checkpoint里面读取模型参数
        for k in ['n_layer', 'n_head', 'n_embed', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = checkpoint_model_args[k]
        gpt_args = Model_args(**model_args)
        model = GPT(gpt_args)
        state_dict = checkpoint['model']  # 模型权重
        model.load_state_dict(state_dict)

        iter_num = checkpoint['iter_num']  # 迭代器步数
        best_val_loss = checkpoint['best_val_loss']
        tqdm_info = tqdm(range(iter_num, max_iters), initial=iter_num)
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
    # 优化：混合精度训练，大部分使用float16，少部分用float32

    model.to(device)
    optimizer = model.configure_optimizers(weight_decay, learning_rate, betas, device_type)
    if init_from == 'resume':
        optimizer.load_state_dict(checkpoint['optimizer'])
    checkpoint = None

    for iter in tqdm_info:
        iter_num = iter
        lr = get_lr(iter_num)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr # 设置学习率

        if iter_num > 0 and iter_num % eval_interval == 0:
            # eval
            loss_dict = estimate_loss(data_dir,model)
            print(f"当前进行{iter_num}个iter,train_loss: {loss_dict['train']},val_loss: {loss_dict['val']}")
            best_val_loss = min(loss_dict['val'],best_val_loss)
            # save checkpoint
            checkpoint = {
                'model':model.state_dict(),
                'optimizer':optimizer.state_dict(),
                'model_args': model_args,
                'iter_num':iter_num,
                'best_val_loss':best_val_loss
            }
            torch.save(checkpoint,os.path.join(checkpoint_save_dir,'checkpoint.pt'),_use_new_zipfile_serialization=False)

            print(f"\U0001F917,checkpoint保存在{checkpoint_save_dir}/checkpoint.pt")

        # with ctx:
        #     logits,loss = model(X,Y)
        #     # print(f"iter:{iter_num},loss:{loss.item()}")
        #     tqdm_info.set_description(f'iter [{iter_num + 1}/{max_iters}]')
        #     tqdm_info.set_postfix(lr=optimizer.param_groups[0]['lr'], loss=loss.item())
        #     scaler.scale(loss).backward()
        #     # 用scaler，scale loss(FP16)，backward得到scaled的梯度(FP16)
        for micro_step in range(gradient_accumulation_steps):
            with ctx:
                logits, loss = model(X, Y)
                loss = loss / gradient_accumulation_steps  # scale the loss to account for gradient accumulation
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = get_batch('train')
            # backward pass, with gradient scaling if training in fp16
            tqdm_info.set_description(f'iter [{iter_num + 1}/{max_iters}]')
            tqdm_info.set_postfix(lr=optimizer.param_groups[0]['lr'], loss=loss.item())
            scaler.scale(loss).backward()
        if grad_clip > 0.0:
            scaler.unscale_(optimizer) # unscale梯度回fp32
            nn.utils.clip_grad_norm_(model.parameters(),grad_clip)
            # 梯度进行裁剪，以防止梯度爆炸
        scaler.step(optimizer) # 用scaler执行optimizer.step()功能
        scaler.update() # scaler factor更新

        optimizer.zero_grad(set_to_none=True) # 释放内存

if __name__ == "__main__":
    print('执行训练')
    train('train',dataset_path=dataset_path,model_args=model_args)