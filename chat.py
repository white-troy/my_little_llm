import tiktoken
import torch
from typing import List
from model_set.nanoGPT import NANOGPT
from config import default_config

class ChatBot:
    def __init__(
            self,
            checkpoint_dir:str = None,
            device:str='cuda',
            dtype:str='float16',
            tokenizer_name='gpt2',
            eos_token:str="<|endoftext|>",
            max_history:int=5
        ):
        self.device = device
        self.dtype=dtype
        # 初始化tokenizer
        self.enc = tiktoken.get_encoding(tokenizer_name)
        self.eos_token = self.enc.encode(
            eos_token,
            allowed_special={eos_token}
        )[0],
        self.model = self.load_model(checkpoint_dir)
        self.model.eval()
        self.model.to(device)
        self.history:List[str] = []

    def load_model(self, checkpoint_path: str):
        if not checkpoint_path.endswith('.pt'):
            raise ValueError("请检查模型权重格式（应为 .pt）")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # 加载模型配置和状态
        model = NANOGPT(default_config)
        model.load_state_dict(checkpoint['model_state_dict'])

        return model
    
    def reset_history(self):
        self.history=[]

    def append_to_history(self, role: str, text: str):
        self.history.append(f"{role}: {text}")
        if len(self.history) > self.max_history * 2:  # user + bot 每轮两句话
            self.history = self.history[-self.max_history * 2:]

    def get_context_prompt(self, new_prompt: str):
        dialogue = "\n".join(self.history + [f"User: {new_prompt}", "Bot:"])
        return dialogue
        
    @torch.no_grad()
    def chat(self, prompt: str, max_new_tokens: int = 50, temperature: float = 1.0, stop_eos: bool = True) -> str:
        self.model.eval()

        # 编码输入
        idx = self.enc.encode(prompt)
        idx = torch.tensor([idx], dtype=torch.long, device=self.device)

        # 生成新tokens
        output = self.model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature, stop_eos=stop_eos)

        # 解码输出
        generated = output[0].tolist()
        new_tokens = generated[len(idx[0]):]

        response = self.enc.decode(new_tokens)
        return response
        
def chat(ckpt_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bot = ChatBot(checkpoint_dir=ckpt_path,device=device)
    while True:
        print('按q退出\n')
        prompt = input("User:")
        if prompt.lower == "q":
            break
        response = bot.chat(prompt, max_new_tokens=256,temperature=0.7)
        print("Bot:", response)

if __name__ == "__main__":
    pt_path = "D:/python/pythonpj/LLM/my_little_llm/checkpoints/best.pt"
    chat(pt_path)