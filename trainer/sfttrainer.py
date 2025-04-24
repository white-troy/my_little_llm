import torch
from data.dataset import SFTDataset
from config import default_config
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn

torch.manual_seed(114514)

def get_dataset(data_path,ratio=0.1):
    block_size = default_config.block_size
    MyDataset = SFTDataset(
        data_path=data_path,
        block_size=block_size,
        max_lines=10000
    )
    train_dataset, val_dataset = torch.utils.data.random_split(MyDataset, [1-ratio, ratio])

    train_loader = DataLoader(train_dataset,default_config.batch_size,shuffle=True)
    val_loader = DataLoader(val_dataset,default_config.batch_size,shuffle=True)

    return train_loader, val_loader

def train_func(model, train_loader, optimizer, lr, device, epoch):
    print("开始监督微调")
    model.train()
    total_loss = 0.0
    total_tokens = 0
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    train_info = tqdm(enumerate(train_loader), 
                total=len(train_loader),
                desc=f'Epoch [{epoch + 1}/{default_config.epochs}]',
                leave=True)
    for batch_idx,(x,y,loss_mask) in train_info:
        x,y,loss_mask = x.to(device), y.to(device), loss_mask.to(device)
        logits,_ = model(x,targets=y) # 不使用预训练时的loss计算方式
        # reshape计算损失
        logits = logits.view(-1, logits.size(-1))
        y = y.view(-1)
        loss_mask = loss_mask.view(-1)
        # 计算token级别的初损失
        loss = loss_fn(logits,y)
        # mask掉前面的token损失
        loss = (loss * loss_mask).sum()/loss_mask.sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr.step()

        total_loss += loss.item()
        total_tokens += loss_mask.sum().item()

    train_info.set_postfix({
            'lr': f"{optimizer.param_groups[0]['lr']:.2e}",
            'batch_loss': f"{loss.item():.4f}",
            'avg_loss': f"{total_loss / (batch_idx + 1):.4f}"
        })

    return total_loss / len(train_loader)

def eval_func(model,val_loader,device,epoch):
    model.eval()
    val_loss = 0.0
    total_tokens = 0
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    with torch.no_grad():
        val_info = tqdm(val_loader, 
                desc=f'Validating Epoch [{epoch + 1}/{default_config.epochs}]', 
                leave=False)
        for x, y, loss_mask in val_info:
            x, y, loss_mask = x.to(device), y.to(device), loss_mask.to(device)

            logits, _ = model(x)
            logits = logits.view(-1, logits.size(-1))
            y = y.view(-1)
            loss_mask = loss_mask.view(-1)

            loss = loss_fn(logits, y)
            loss = (loss * loss_mask).sum() / loss_mask.sum()

            val_loss += loss.item()
            val_info.set_postfix(val_loss=f"{val_loss / (len(val_info) + 1e-7):.4f}")

    return val_loss / len(val_loader)
