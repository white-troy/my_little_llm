import yaml
from typing import Dict, Any

class ModelConfig:
    def __init__(self, model_cfg: str = None, train_cfg: str = None):
        self._mconfig, self._tconfig = self.load_configs(model_cfg, train_cfg)
        self.setup_dot_access()

    def load_configs(self, model_cfg: str, train_cfg: str) -> tuple:
        def load_single(path: str, name: str) -> Dict[str, Any]:
            with open(path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            if not config:
                raise ValueError(f"{name}配置文件为空或格式不正确")
            return config

        return (
            load_single(model_cfg, 'model'),
            load_single(train_cfg, 'train')
        )

    def setup_dot_access(self):
        for config_dict in [self._mconfig,self._tconfig]:
            for k,v in config_dict.items():
                setattr(self,k,v)

    @property
    def get_mcfg(self) -> Dict[str, Any]:
        return self._mconfig

    @property
    def get_tcfg(self) -> Dict[str, Any]:
        return self._tconfig


mcfg = 'config/model_config.yml'
tcfg = 'config/train_config.yml'

default_config = ModelConfig(mcfg, tcfg)


if __name__ == "__main__":
    mcfg = 'model_config.yml'
    tcfg = 'train_config.yml'

    default_config1 = ModelConfig(mcfg, tcfg)
    print(default_config1.get_mcfg)
    print(default_config1.get_tcfg)
    print(default_config1.model_name)