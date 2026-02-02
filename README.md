# NanoGPT KV Caching Implementation

Implementation of Key-Value (KV) caching in NanoGPT(by Andrej Karpathy) to improve inference efficiency during autoregressive text generation.

## Preamble
Transformer architectures have become the standard in AI, particularly for large language models (LLMs), largely replacing RNNs for many NLP tasks. This is because language processing is inherently contextual: the meaning of a word can change drastically depending on previous words.

Transformers use self-attention to overcome the sequential processing limitation of RNNs. Each token is associated with a query (Q), key (K), and value (V), which interact with all other tokens to determine the next most probable token. However, this computation scales quadratically with sequence length, making it increasingly expensive as the number of tokens grows. 

This is the motivation for KV caching, as it addresses this by storing previously computed key-value pairs to avoid redundant calculations.

## Project Overview
NanoGPT is a simplified and lightweight GPT implementation that can be run locally. The original NanoGPT does not include KV caching, which means that token generation recomputes all attention values at each step. In this project, we implement KV caching to significantly reduce the time required for token processing and generation.
To start off:
1. Understand the mathematical computation of self-attention using Q, K, V.
2. Understand how NanoGPT utilises PyTorch's batch multiplication to do parallel processing of attention computation
3. Implement KV caching into NanoGPT, referencing GPT-2 (very tough ngl)
    - modifying the `forward()` methods in `model.py` of NanoGPT
4. Benchmark token processing and generation speed before and after KV caching.
    - Python script to run and visualise the improvement

### TLDR
> This project modifies the original NanoGPT implementation to include KV caching, based on my understanding of GPT which stores previously computed key-value pairs during generation. 

## Important Files

- [Documentation](https://github.com/realestzeyu/kv-caching-implementation-nanogpt/blob/main/KV%20Caching%20in%20NanoGPT%20Documentation.pdf) - Detailed explanation and walkthrough of the project
- Presentation Slides [1](https://github.com/realestzeyu/kv-caching-implementation-nanogpt/blob/main/KV%20Caching%20in%20NanoGPT%20Presentation%20Part%201.pdf) & [2](https://github.com/realestzeyu/kv-caching-implementation-nanogpt/blob/main/KV%20Caching%20in%20NanoGPT%20Presentation%20Part%202.pdf)- used to explain briefly & intuitively of the project
- `model.py` - Original NanoGPT model
- `model_kv.py` - **Modified** model with KV caching implementation
- `benchmark.py` - Script to benchmark autoregressive performance with and without KV caching
- `benchmark_clean.png` - Performance comparison graphs with and without KV caching (old)
- `newbenchmark_clean.png` - Performance comparison graphs with and without KV caching (new, look at note below)
- `compare_models.py` - Script to compare english text generation outputs

> Note: Extra line of code for attention computation in `model_kv.py` under `CausalSelfAttention.forward()` is added. (line 104 in `model_kv.py`) Accidentally left out in the previous version. Thus, the documentation and presentation slides misses this line. The explanation does not change, was just a careless mistake.

## Author

Wang Zeyu

## References

- [NanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy
