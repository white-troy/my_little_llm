import torch
from config import default_config
from model_set.nanoGPT import NANOGPT
from trainer import get_dataset,train_func,eval_func

def calculate_params(model,config,device):
    model = NANOGPT(config)
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6} M")


def train(model,config,train_func,eval_func):
    epochs = config.epochs
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    optimizer_config = config.optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],  # 使用顶层的lr配置
        betas=(optimizer_config['beta1'], optimizer_config['beta2']),
        weight_decay=optimizer_config['weight_decay']
    )
    lr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
    train_loader,val_loader = get_dataset(config.dataset_path)

    best_loss = 1e9
    for epoch in range(epochs):
        train_loss = train_func(model,train_loader,optimizer,lr,device,epoch)
        val_loss = eval_func(model,val_loader,device,epoch)

        # 保存模型
        avg_val_loss = val_loss / len(val_loader)
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_state_dict': lr.state_dict(),
                'val_loss': avg_val_loss,
            }
            torch.save(checkpoint, f'checkpoints/best.pt')
            print('保存权重')


def main():
    model = NANOGPT(default_config)
    train(model,default_config,train_func,eval_func)

if __name__ == "__main__":
    main()