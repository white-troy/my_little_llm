import torch
from data.dataset import TextDataset
from config import default_config
from torch.utils.data import DataLoader
from tqdm import tqdm

torch.manual_seed(114514)

def get_dataset(data_path,ratio=0.1):
    block_size = default_config.block_size
    MyDataset = TextDataset(
        data_path=data_path,
        block_size=block_size,
        max_lines=380000
    )
    # 划分数据集
    # val_size = int(ratio * len(MyDataset))
    # train_size = len(MyDataset) - val_size
    # train_dataset, val_dataset = torch.utils.data.random_split(
    #     MyDataset, 
    #     [train_size, val_size]
    # )
    train_dataset, val_dataset = torch.utils.data.random_split(MyDataset, [1-ratio, ratio])

    train_loader = DataLoader(train_dataset,default_config.batch_size,shuffle=True)
    val_loader = DataLoader(val_dataset,default_config.batch_size,shuffle=True)

    return train_loader, val_loader

def train_func(model,train_loader,optimizer,lr,device,epoch):
    print("开始训练")
    model.train()
    total_loss = 0.0
    train_info = tqdm(enumerate(train_loader), 
                    total=len(train_loader),
                    desc=f'Epoch [{epoch + 1}/{default_config.epochs}]',
                    leave=True)
    for batch_idx,(x,y) in train_info:
        x,y = x.to(device),y.to(device)
        logits,loss = model(x,targets=y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr.step()
        total_loss += loss.item()
        train_info.set_postfix({
            'lr': f"{optimizer.param_groups[0]['lr']:.2e}",
            'batch_loss': f"{loss.item():.4f}",
            'avg_loss': f"{total_loss/(batch_idx+1):.4f}"
        })
    return total_loss / len(train_loader)

def eval_func(model,val_loader,device,epoch):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        val_info = tqdm(val_loader, 
                desc=f'Validating Epoch [{epoch + 1}/{default_config.epochs}]', 
                leave=False)
        for x, y in val_info:
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, targets=y)
            val_loss += loss.item()
            val_info.set_postfix(val_loss=f"{val_loss/(len(val_info)+1e-7):.4f}")
    return val_loss  / len(val_loader)

