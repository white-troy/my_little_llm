import tiktoken
import torch
from typing import List,Dict
from model_set.nanoGPT import NANOGPT
from config import default_config
from data.dataset import get_chatml_tokenizer

class ChatBot:
    def __init__(
            self,
            checkpoint_dir:str = None,
            device:str='cuda',
            dtype:str='float16',
            tokenizer_name='gpt2',
            max_history:int=5
        ):
        self.device = device
        self.dtype=dtype
        # 初始化tokenizer
        self.enc = get_chatml_tokenizer(tokenizer_name)
        self.allowed = {"<|im_start|>", "<|im_end|>", "<|endoftext|>"}
        self.eos_token = self.enc.encode("<|im_end|>", allowed_special=self.allowed)[0]
        self.model = self.load_model(checkpoint_dir)
        self.model.eval()
        self.model.to(device)
        self.history: List[Dict[str, str]] = []
        self.max_history = max_history

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
        self.history.append({"role": role, "content": text})
        if len(self.history) > self.max_history * 2:
            self.history = self.history[-self.max_history * 2:]

    def build_prompt(self, new_input: str) -> str:
        prompt = ""
        for turn in self.history:
            role = turn["role"]
            content = turn["content"]
            prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        prompt += f"<|im_start|>user\n{new_input}<|im_end|>\n"
        prompt += f"<|im_start|>assistant\n"  # 生成从这里开始
        return prompt

    @torch.no_grad()
    def chat_pre(self, prompt: str, max_new_tokens: int = 128, temperature: float = 1.0, stop_eos: bool = True) -> str:
        self.model.eval()

        # 编码输入
        input = f"<|im_start|>{prompt}"
        idx = self.enc.encode(input, allowed_special=self.allowed)
        input_ids = torch.tensor([idx], dtype=torch.long, device=self.device)

        # 只传入首次prompt，后续不重复
        output = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            stop_eos=stop_eos,
            use_cache=True
        )

        # 解码输出
        new_tokens = output[0, input_ids.shape[1]:].tolist()
        response = self.enc.decode(new_tokens)
        return response
    
    @torch.no_grad()
    def chat_sft(self, prompt: str, max_new_tokens: int = 128, temperature: float = 1.0, stop_eos: bool = True) -> str:
        self.model.eval()
        prompt_str = self.build_prompt(prompt)
        input_ids = self.enc.encode(prompt_str, allowed_special={"<|im_start|>", "<|im_end|>","<|endoftext|>"})
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        output = self.model.generate(
            input_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            stop_eos=stop_eos,
            use_cache=False
        )

        # 取新生成部分
        generated = output[0].tolist()
        new_tokens = generated[len(input_ids):]

        # 截断到 <|im_end|>
        if self.eos_token in new_tokens:
            new_tokens = new_tokens[:new_tokens.index(self.eos_token)]

        response = self.enc.decode(new_tokens)
        self.append_to_history("user", prompt)
        self.append_to_history("assistant", response)
        return response
        
def chat(ckpt_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bot = ChatBot(checkpoint_dir=ckpt_path, device=device)
    use_sft = False
    if ckpt_path.endswith("sft.pt"):
        use_sft = True
    while True:
        print('按 q 退出\n')
        prompt = input("User: ")
        if prompt.strip().lower() == "q":
            break
        if use_sft:
            response = bot.chat_sft(prompt, max_new_tokens=512, temperature=0.8)
        else:
            response = bot.chat_pre(prompt, max_new_tokens=512, temperature=0.8)
        print("Bot:", response)

def chat_test(ckpt_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bot = ChatBot(checkpoint_dir=ckpt_path, device=device)
    test_prompt = "你好！"
    # response = bot.chat_pre(test_prompt, max_new_tokens=512, temperature=1)
    response = bot.chat_sft(test_prompt, max_new_tokens=512, temperature=1)
    print("Test input:", test_prompt)
    print("Test Response:", response)
    return response

if __name__ == "__main__":
    pt_path = "D:/python/pythonpj/LLM/my_little_llm/checkpoints/best_pre.pt"
    # chat(pt_path)
    chat_test(pt_path)

