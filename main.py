import torch
from config import default_config
from model_set.nanoGPT import NANOGPT
from trainer import pre_data,pre_train_func,pre_eval_func,sft_data,sft_train_func,sft_eval_func
import os
from logger import logger
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

device = "cuda" if torch.cuda.is_available() else "cpu"
current_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_path)

def calculate_params(model,config,device):
    model = NANOGPT(config)
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6} M")


def pre_train(model,config,train_func,eval_func):
    epochs = config.epochs

    model = model.to(device)

    logger.info(f"Training started on device: {device}")
    logger.info(f"Total epochs: {config.epochs}")

    optimizer_config = config.optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.lr),
        betas=(float(optimizer_config['beta1']), float(optimizer_config['beta2'])),
        weight_decay=float(optimizer_config['weight_decay'])
    )
    lr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
    train_loader,val_loader = pre_data(config.pre_dataset_path)

    best_loss = 1e9
    print("开始训练")
    for epoch in range(epochs):
        train_loss = train_func(model,train_loader,optimizer,lr,device,epoch)
        val_loss = eval_func(model,val_loader,device,epoch)

        # 保存模型
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        logger.info(f"Epoch {epoch+1}/{config.epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")


        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            save_path = os.path.join(current_dir,'checkpoints')
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_state_dict': lr.state_dict(),
                'val_loss': avg_val_loss,
            }
            torch.save(checkpoint, f'{save_path}/best_pre.pt')
            logger.info(f"Saved best model with val loss: {avg_val_loss:.4f}")

# 加载模型权重
def load_model(weight_path):
    if not weight_path.endswith('.pt'):
        raise ValueError("请检查模型权重格式（应为 .pt）")
    checkpoint = torch.load(weight_path, map_location=device)

    # 加载模型配置和状态
    model = NANOGPT(default_config)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model

def sft_train(weight_path,config,train_func,eval_func):
    epochs = config.epochs

    model = load_model(weight_path).to(device)
    logger.info(f"SFT Training started on device: {device}")
    logger.info(f"Total epochs: {config.epochs}")

    optimizer_config = config.optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.lr),
        betas=(float(optimizer_config['beta1']), float(optimizer_config['beta2'])),
        weight_decay=float(optimizer_config['weight_decay'])
    )

    lr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)

    train_loader,val_loader = sft_data(config.sft_dataset_path)
    best_loss = 1e9
    print("开始监督微调")
    for epoch in range(epochs):
        train_loss = train_func(model,train_loader,optimizer,lr,device,epoch)
        val_loss = eval_func(model,val_loader,device,epoch)

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        logger.info(f"[SFT] Epoch {epoch+1}/{config.epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            save_path = os.path.join(current_dir, 'checkpoints')
            os.makedirs(save_path, exist_ok=True)

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_state_dict': lr.state_dict(),
                'val_loss': avg_val_loss,
            }
            torch.save(checkpoint, f'{save_path}/best_sft.pt')
            logger.info(f"[SFT] Saved best model with val loss: {avg_val_loss:.4f}")

def main(mode='pre'):
    model = NANOGPT(default_config)
    logger.info("Model initialized")
    if mode == 'pre':
        pre_train(model, default_config, pre_train_func, pre_eval_func)
    elif mode == 'sft':
        weight_path = r"checkpoints\best_pre.pt"
        sft_train(weight_path, default_config, sft_train_func, sft_eval_func)
    else:
        raise ValueError("请输入选择的训练方式：'pre' 或 'sft'")

if __name__ == "__main__":
    main()
    save_path = os.path.join(current_dir,'checkpoints')
    if not os.path.exists(save_path):   
        os.mkdir(save_path)
    print(f'{save_path}/best.pt')