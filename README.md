# NanoGPT KV Caching Implementation

Implementation of Key-Value (KV) caching in NanoGPT to improve inference efficiency during autoregressive text generation.

## Project Overview

This project modifies the original NanoGPT implementation to include KV caching, which stores previously computed key-value pairs during generation. This optimization reduces time complexity from O(n²) to O(n) for generating n tokens.

# FOR TA/ PROF: these are the changes
## Files

- `model.py` - Original NanoGPT model
- `model_kv.py` - Modified model with KV caching implementation
- `benchmark.py` - Script to benchmark performance with and without KV caching
- `benchmark_clean.png` - Performance comparison graphs


### This Readme is still being updated...


## Author

Wang Zeyu

## References

- [NanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy
