import yaml
from typing import Dict, Any

def load_config(path,name) -> Dict[str,Any]:
    "从yml文件中加载模型的配置"
    with open(path,'r',encoding='utf-8') as f:
        model_config = yaml.safe_load(f)

    if model_config is None:
        raise ValueError(f"{name}配置文件为空或格式不正确")
    return model_config

def load_train_config(model_cfg, train_cfg):
    model_config = load_config(model_cfg,'model')
    train_config = load_config(train_cfg,'train')
    return model_config,train_config

if __name__ == "__main__":
    model_cfg = 'model_config.yml'
    train_cfg = 'train_config.yml'
    m,t = load_train_config(model_cfg,train_cfg)
    print(f"模型配置：{m}")
    print(f"训练配置：{t}")