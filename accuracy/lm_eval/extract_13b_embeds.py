# extract_13b_embeds.py
import torch
from transformers import OPTForCausalLM

print("Loading OPT-13B (CPU)...")
model_13b = OPTForCausalLM.from_pretrained("/home/liushenyu/opt-13b/", torch_dtype=torch.float16, device_map="cpu")

# 提取词嵌入矩阵 (50272 x 5120)
embed_weights = model_13b.model.decoder.embed_tokens.weight.detach().clone()
torch.save(embed_weights, "/home/liushenyu/test/opt_13b_embed_weights.pth")
print("Saved 13B embeddings successfully!")