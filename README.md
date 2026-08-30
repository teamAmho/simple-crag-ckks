# CRAG: CKKS-Friendly Encrypted Retrieval for RAG

> **Paper:** *(link will be added after the review process)*

This repository contains the GPU-accelerated CKKS implementation and
evaluation scripts for CRAG. Document and query embeddings remain encrypted
during cloud-side similarity computation and selector-based document recovery.

## Repository Contents

- `gpu_ckks_friendly_RAG.ipynb`: reference notebook for the original pipeline
- `benchmarks/gpu_ckks_rag.py`: complete encrypted retrieval implementation
- `benchmarks/gpu_ckks_rag_streaming.py`: memory-bounded complete retrieval
- `benchmarks/rag.py`: plaintext retrieval baseline
- `benchmarks/compare.py`: document-aligned plaintext/encrypted comparison
- `benchmarks/multi_query_accuracy.py`: multi-query ranking evaluation
- `scripts/benchmark_dot_product.py`: encrypted similarity kernel benchmark
- `scripts/benchmark_matrix_mul.py`: naive encrypted recovery benchmark
- `benchmarks/results/`: CSV, JSON, and LaTeX outputs reported in the paper

The streaming implementation decrypts each completed recovery output before
computing the next one instead of retaining every output ciphertext on the
GPU. This changes output lifetime and peak memory use, not selector
construction, encrypted arithmetic, or ranking logic.

## Requirements

- NVIDIA GPU with CUDA support
- Python 3.10+
- [`desilofhe`](https://github.com/desilo-ai/desilofhe)
- `torch`, `numpy`, `chromadb`, and `sentence-transformers`
- `langchain-community` and `langchain-text-splitters`

Install the Python dependencies in a virtual environment:

```bash
pip install desilofhe torch numpy chromadb sentence-transformers \
  langchain-community langchain-text-splitters
```

## Build the Databases

Keys, ChromaDB files, and encrypted SQLite databases are generated locally and
are intentionally excluded from version control. From the repository root,
build the 256- and 512-dimensional databases as follows:

```bash
python benchmarks/gpu_ckks_rag_streaming.py \
  --data-dir data/256 --dim 256 build --doc cat-facts.txt

python benchmarks/gpu_ckks_rag_streaming.py \
  --data-dir data/512 --dim 512 build --doc cat-facts.txt
```

Each build creates the following layout:

```text
data/{dim}/
├── keys/
├── chroma/
└── db/
    ├── row_encrypted.db
    └── col_encrypted.db
```

The row-oriented database contains one packed document embedding per
ciphertext and is used for encrypted similarity computation. The
column-oriented database contains one encrypted embedding coordinate per
ciphertext and is used for selector-based document recovery.

## Complete Retrieval Comparison

The following example evaluates the original query at dimension 256 and
Top-4. Other evaluated cutoffs are 8, 16, and 32.

```bash
python benchmarks/gpu_ckks_rag_streaming.py \
  --data-dir data/256 --dim 256 retrieve \
  --query "What is bezoar?" --topk 4 \
  --save-vectors benchmarks/enc_256_k4_streaming.npy

python benchmarks/rag.py \
  --chroma-dir data/256/chroma --dim 256 --topk 4 \
  --query "What is bezoar?" \
  --save-vectors benchmarks/pt_256_k4.npy

python benchmarks/compare.py \
  benchmarks/pt_256_k4.npy benchmarks/enc_256_k4_streaming.npy
```

`compare.py` reports Top-k set overlap, exact-rank agreement, similarity-score
error, vector MAE, and maximum vector-coordinate error. Score and vector
differences are aligned by document index, so documents at different ranks are
never compared to one another.

## Multi-Query Evaluation

The query list is stored in `benchmarks/multi_query_queries.txt`. One encrypted
score vector is computed for each query and reused for all requested cutoffs.

```bash
python benchmarks/multi_query_accuracy.py \
  --dim 256 --data-dir data/256 --topks 4 8 16 32

python benchmarks/multi_query_accuracy.py \
  --dim 512 --data-dir data/512 --topks 4 8 16 32

python benchmarks/multi_query_report.py
```

## Kernel Benchmarks

The similarity and document-recovery kernels are measured independently, not
as end-to-end latency. The dot-product command below performs one warm-up and
four measured runs; the matrix-multiplication command performs one measured
run.

```bash
python scripts/benchmark_dot_product.py --dim 256 --topk 4 --repeat 5
python scripts/benchmark_matrix_mul.py --dim 256 --topk 4 --repeat 1
```

Repeat the commands for dimensions 256 and 512 and Top-k values 4, 8, 16, and
32. Top-k does not change the similarity kernel workload because all document
scores are computed before selection.

## Repository Structure

```text
.
├── benchmarks/
│   ├── results/
│   ├── compare.py
│   ├── gpu_ckks_rag.py
│   ├── gpu_ckks_rag_streaming.py
│   ├── multi_query_accuracy.py
│   ├── multi_query_queries.txt
│   ├── multi_query_report.py
│   └── rag.py
├── scripts/
│   ├── benchmark_dot_product.py
│   └── benchmark_matrix_mul.py
├── cat-facts.txt
└── gpu_ckks_friendly_RAG.ipynb
```

## Data Attribution

The sample corpus `cat-facts.txt` is derived from
[`ngxson/demo_simple_rag_py`](https://huggingface.co/ngxson/demo_simple_rag_py)
(MIT License).

## Citation

*(BibTeX will be added after publication.)*
