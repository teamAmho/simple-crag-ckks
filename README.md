# CRAG:  CKKS-friendly retrieval-augmented generation framework

> **Paper:** *(link will be added upon publication)*

This repository contains the implementation and benchmark scripts accompanying our paper on GPU-accelerated CKKS homomorphic encryption applied to privacy-preserving Retrieval-Augmented Generation (RAG).

## Overview

We implemented a RAG pipeline where document embeddings are stored as CKKS ciphertexts, and similarity search (dot product) is performed entirely under encryption on the GPU. This repository includes:

- **`gpu_ckks_friendly_RAG.ipynb`** — end-to-end pipeline: embedding, encryption, encrypted retrieval
- **`scripts/benchmark_dot_product.py`** — benchmark for the encrypted dot product stage (query vs. encrypted DB)
- **`scripts/benchmark_matrix_mul.py`** — benchmark for the encrypted matrix multiplication stage (selector-based Top-k retrieval)
- **`data/cat-facts.txt`** — sample dataset used to build the encrypted DB in the notebook

> **Note on benchmarks:** The two scripts measure the dot product and MM stages independently, not as a single end-to-end pipeline. To reduce GPU memory pressure, only the first and last ciphertext results are decrypted for sanity checking — full intermediate results are not retained. See the paper for a detailed discussion of this design choice.

## Requirements

- NVIDIA GPU with CUDA support
- Python 3.10+
- [`desilofhe`](https://github.com/desilo-ai/desilofhe) — GPU-accelerated FHE library
- `chromadb`, `sentence-transformers`, `numpy`

Install dependencies:

```bash
pip install desilofhe chromadb sentence-transformers numpy
```

## Setup

Keys and the encrypted database are **not included** in this repository (see `.gitignore`). You must generate them by running the notebook, then manually place the files in the structure below.

### 1. Generate keys and encrypted DB

Open and run `gpu_ckks_friendly_RAG.ipynb` from top to bottom. The notebook will:

1. Load `data/cat-facts.txt` as the document corpus
2. Embed each document using `nomic-ai/nomic-embed-text-v1.5`
3. Generate CKKS keys (`secret.key`, `public.key`, `relin.key`, `rotation.key`)
4. Encrypt and store the transposed document embeddings as a SQLite DB

### 2. Arrange the generated files

The benchmark scripts expect the following directory structure.
Create the folders and copy the generated files accordingly:

```
data/
└── {dim}/               # e.g. 256 or 512 depending on embedding dimension
    ├── keys/
    │   ├── secret.key
    │   ├── public.key
    │   ├── relin.key
    │   └── rotation.key
    └── db/
        └── t_row_encrypted.db   # transposed encrypted DB
```

> The notebook saves files with different names/paths by default — rename and move them to match the structure above before running the benchmarks.

### 3. Run the benchmarks

After the notebook has run successfully:

```bash
# Benchmark encrypted dot product (dim=256, top_k=10, 5 runs with 1 warm-up)
python scripts/benchmark_dot_product.py --dim 256 --topk 10 --repeat 5

# Benchmark encrypted dot product with a custom query
python scripts/benchmark_dot_product.py --dim 256 --topk 10 --query "What do cats eat?"

# Benchmark encrypted matrix multiplication (dim=256, top_k=10)
python scripts/benchmark_matrix_mul.py --dim 256 --topk 10

# Run with 512-dimensional embeddings
python scripts/benchmark_dot_product.py --dim 512 --topk 10
python scripts/benchmark_matrix_mul.py --dim 512 --topk 10
```

Each script prints a Markdown-formatted results table at the end, matching the format used in the paper.

## Repository Structure

```
.
├── data/
│   ├── cat-facts.txt                # Sample document corpus
│   └── {256,512}/                   # Per-dimension dirs (keys/ and db/) — set up manually
├── gpu_ckks_friendly_RAG.ipynb      # Full pipeline notebook
└── scripts/
    ├── benchmark_dot_product.py     # Dot product benchmark
    └── benchmark_matrix_mul.py      # Matrix multiplication benchmark
```

## Acknowledgements

The sample corpus `cat-facts.txt` is from [ngxson/demo_simple_rag_py](https://huggingface.co/ngxson/demo_simple_rag_py) (MIT License).

## Citation

*(BibTeX will be added upon publication)*
