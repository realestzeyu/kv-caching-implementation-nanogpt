"""
Simple benchmark: Original vs KV-cached NanoGPT
DISCLAIMER: I used Claude AI to help me with how to write the benchmark script.
"""

import torch
import time
import matplotlib.pyplot as plt

import model as orig
import model_kv as kv

device = "cpu"  # use all cpu so that anyone can run it, because cuda is not available to everyone LOL

# comparison betwen with and withou kv caching, also use gpt2 weights for convenience
model_orig = orig.GPT.from_pretrained("gpt2").to(device).eval()
model_kv = kv.GPT.from_pretrained("gpt2").to(device).eval()

# Test configs
prompts = [i for i in range(10, 201, 20)]  # no. of tokens in the prompt
gens = [i for i in range(10, 201, 20)]  # no. of tokens to generate

results = {
    "orig": [],
    "kv": [],
    "labels": [],
    "prompts": [],
    "gens": [],
}


print(f"\n{'Config':<15} {'Original':<12} {'KV-Cache':<12} {'Speedup':<10}")
print("=" * 70)

# Fixed generation length for prompt length test
fixed_gen = 50
for p in prompts:
    idx = torch.randint(0, 50257, (1, p), device=device)

    # Original
    t0 = time.time()
    with torch.no_grad():
        _ = model_orig.generate(idx.clone(), fixed_gen)
    time_orig = time.time() - t0

    # KV-cached
    t0 = time.time()
    with torch.no_grad():
        _ = model_kv.generate(idx.clone(), fixed_gen)
    time_kv = time.time() - t0

    speedup = time_orig / time_kv
    label = f"P{p}G{fixed_gen}"

    results["orig"].append(time_orig)
    results["kv"].append(time_kv)
    results["labels"].append(label)
    results["prompts"].append(p)
    results["gens"].append(fixed_gen)

    print(f"{label:<15} {time_orig:<12.3f} {time_kv:<12.3f} {speedup:<10.2f}")

# Fixed prompt length for generation length test
fixed_prompt = 50
for g in gens:
    if g == fixed_gen:
        continue
    idx = torch.randint(0, 50257, (1, fixed_prompt), device=device)

    # Original
    t0 = time.time()
    with torch.no_grad():
        _ = model_orig.generate(idx.clone(), g)
    time_orig = time.time() - t0

    # KV-cached
    t0 = time.time()
    with torch.no_grad():
        _ = model_kv.generate(idx.clone(), g)
    time_kv = time.time() - t0

    speedup = time_orig / time_kv
    label = f"P{fixed_prompt}G{g}"

    results["orig"].append(time_orig)
    results["kv"].append(time_kv)
    results["labels"].append(label)
    results["prompts"].append(fixed_prompt)
    results["gens"].append(g)

    print(f"{label:<15} {time_orig:<12.3f} {time_kv:<12.3f} {speedup:<10.2f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# -------------------------
# Plot 1: Time vs Prompt Length (fixed gen=50)
# -------------------------
indices = [i for i, gen in enumerate(results["gens"]) if gen == fixed_gen]
p_values = [results["prompts"][i] for i in indices]
orig_times = [results["orig"][i] for i in indices]
kv_times = [results["kv"][i] for i in indices]

ax1.plot(p_values, orig_times, marker="o", linestyle="-", label="Original")
ax1.plot(p_values, kv_times, marker="o", linestyle="--", label="KV-cache")

ax1.set_xlabel("Prompt Length (tokens)")
ax1.set_ylabel("Time (s)")
ax1.set_title(f"Time vs Prompt Length\n(fixed generation = {fixed_gen} tokens)")
ax1.legend()
ax1.grid(alpha=0.3)


# -------------------------
# Plot 2: Time vs Generation Length (fixed prompt=50)
# -------------------------
indices = [
    i
    for i, (prompt, gen) in enumerate(zip(results["prompts"], results["gens"]))
    if prompt == fixed_prompt
]
g_values = [results["gens"][i] for i in indices]
orig_times = [results["orig"][i] for i in indices]
kv_times = [results["kv"][i] for i in indices]

# Sort by generation length
sorted_idx = sorted(range(len(g_values)), key=lambda i: g_values[i])
g_values = [g_values[i] for i in sorted_idx]
orig_times = [orig_times[i] for i in sorted_idx]
kv_times = [kv_times[i] for i in sorted_idx]

ax2.plot(g_values, orig_times, marker="s", linestyle="-", label="Original")
ax2.plot(g_values, kv_times, marker="s", linestyle="--", label="KV-cache")

ax2.set_xlabel("Generation Length (tokens)")
ax2.set_ylabel("Time (s)")
ax2.set_title(f"Time vs Generation Length\n(fixed prompt = {fixed_prompt} tokens)")
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("newbenchmark_clean.png", dpi=150)
print("\n Saved benchmark_clean.png")
plt.show()
