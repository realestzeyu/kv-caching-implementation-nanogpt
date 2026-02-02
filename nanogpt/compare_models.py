"""
Simple script to compare inference speeds between standard GPT and GPT with KV caching.
DISCLAIMER: I used Claude AI to help me with how to write the comparison script.
"""

import os
import pickle
import time
import torch
from model import GPT as GPT_standard, GPTConfig
from model_kv import GPT as GPT_kv

# Configuration variables
init_from = "gpt2"  # 'gpt2' (for standard GPT-2)
start_prompt = "I play basketball"  # starting prompt
max_new_tokens = 100  # number of tokens to generate
temperature = 0.8
top_k = 200
seed = 1337
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)
compile = False

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = "cuda" if "cuda" in device else "cpu"
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]
ctx = (
    torch.no_grad()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)

# Load model config and meta
# Use GPT-2 config and tiktoken
gptconf = GPTConfig()  # default GPT-2 config
import tiktoken

enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s)
decode = lambda l: enc.decode(l)


# Prepare input
start_ids = encode(start_prompt)
x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]


# Function to load model
def load_model(ModelClass):
    model = ModelClass.from_pretrained(init_from, dict(dropout=0.0))
    model.eval()
    model.to(device)
    if compile:
        model = torch.compile(model)
    return model


# Load both models
print("Loading standard GPT model...")
model_standard = load_model(GPT_standard)
print("Loading GPT with KV caching...")
model_kv = load_model(GPT_kv)


# Function to time generation
def time_generation(model, x, max_new_tokens, temperature, top_k, model_name):
    start_time = time.time()
    with ctx:
        y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
    end_time = time.time()
    elapsed = end_time - start_time
    generated_tokens = y[0].tolist()
    generated_text = decode(generated_tokens)
    print(
        f"{model_name} generated {len(generated_tokens) - len(x[0])} tokens in {elapsed:.4f} seconds"
    )
    print(f"Tokens per second: {(len(generated_tokens) - len(x[0])) / elapsed:.2f}")
    return elapsed, generated_tokens, generated_text


# Compare
print("\nComparing inference speeds...\n")

time_standard, tokens_standard, text_standard = time_generation(
    model_standard, x, max_new_tokens, temperature, top_k, "Standard GPT"
)
time_kv, tokens_kv, text_kv = time_generation(
    model_kv, x, max_new_tokens, temperature, top_k, "GPT with KV caching"
)

speedup = time_standard / time_kv if time_kv > 0 else float("inf")
print(f"\n" + "=" * 50)
print(f"SPEEDUP WITH KV CACHING: {speedup:.2f}x faster")
print(f"Standard GPT time: {time_standard:.4f}s")
print(f"KV GPT time: {time_kv:.4f}s")
print("=" * 50)

# Print generated tokens
print("\nGenerated tokens (Standard GPT):")
print(tokens_standard)
print("\nGenerated tokens (KV GPT):")
print(tokens_kv)

# Print generated texts
print("\nGenerated text (Standard GPT):")
print(repr(text_standard))
print("\nGenerated text (KV GPT):")
print(repr(text_kv))
