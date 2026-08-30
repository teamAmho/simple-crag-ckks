"""
Compare plaintext vs. encrypted top-k retrieval results.

Usage:
    # rag.py (plaintext selector-matmul) vs gpu_ckks_rag.py (encrypted selector-matmul)
    python compare.py pt_vectors.npy enc_matmul.npy

    # rag.py vs benchmark_dot_product.py (direct ciphertext decrypt)
    python compare.py pt_vectors.npy enc_direct.npy

Each .npy file is a dict saved with allow_pickle=True:
    "vectors" : np.ndarray  shape (k, dim)   — required
    "indices" : np.ndarray  shape (k,)        — required
    "scores"  : np.ndarray  shape (k,)        — optional (skipped if missing in either file)
"""
from pathlib import Path
import argparse
import numpy as np


def load(path: Path) -> dict:
    data = np.load(str(path), allow_pickle=True).item()
    for key in ("vectors", "indices"):
        assert key in data, f"{path} is missing required key '{key}'"
    return data


def print_index_overlap(pt_indices: np.ndarray, enc_indices: np.ndarray):
    pt_set = set(pt_indices.tolist())
    enc_set = set(enc_indices.tolist())
    overlap = pt_set & enc_set
    k = len(pt_indices)
    print(f"Top-{k} index overlap : {len(overlap)}/{k}")

    exact_rank_match = int(np.sum(pt_indices == enc_indices))
    print(f"Exact rank match      : {exact_rank_match}/{k}")

    if len(overlap) < k:
        only_pt = sorted(pt_set - enc_set)
        only_enc = sorted(enc_set - pt_set)
        print(f"  Only in plaintext   : {only_pt}")
        print(f"  Only in encrypted   : {only_enc}")


def align_by_document(pt_indices: np.ndarray, enc_indices: np.ndarray):
    """Return common document indices and their positions in each ranked list."""
    if len(set(pt_indices.tolist())) != len(pt_indices):
        raise ValueError("Plaintext indices contain duplicates")
    if len(set(enc_indices.tolist())) != len(enc_indices):
        raise ValueError("Encrypted indices contain duplicates")

    enc_positions = {int(doc_idx): pos for pos, doc_idx in enumerate(enc_indices)}
    common_docs = [int(doc_idx) for doc_idx in pt_indices if int(doc_idx) in enc_positions]
    pt_positions = np.array(
        [int(np.flatnonzero(pt_indices == doc_idx)[0]) for doc_idx in common_docs],
        dtype=np.int64,
    )
    enc_aligned_positions = np.array(
        [enc_positions[doc_idx] for doc_idx in common_docs],
        dtype=np.int64,
    )
    return np.array(common_docs, dtype=np.int64), pt_positions, enc_aligned_positions


def print_score_diff(
    pt_scores: np.ndarray,
    enc_scores: np.ndarray,
    doc_indices: np.ndarray,
    pt_positions: np.ndarray,
    enc_positions: np.ndarray,
):
    diff = np.abs(pt_scores - enc_scores)
    print("\nScore diff (|pt - enc|), aligned by document index:")
    print(
        f"  {'doc':>4}  {'pt_rank':>7}  {'enc_rank':>8}  "
        f"{'pt_score':>10}  {'enc_score':>10}  {'|diff|':>10}"
    )
    for doc_idx, pt_pos, enc_pos, ps, es, d in zip(
        doc_indices, pt_positions, enc_positions, pt_scores, enc_scores, diff
    ):
        print(
            f"  {doc_idx:4d}  {pt_pos + 1:7d}  {enc_pos + 1:8d}  "
            f"{ps:10.6f}  {es:10.6f}  {d:10.3e}"
        )
    print(f"  mean |score diff| : {diff.mean():.3e}")
    print(f"  max  |score diff| : {diff.max():.3e}")


def print_vector_diff(
    pt_vecs: np.ndarray,
    enc_vecs: np.ndarray,
    doc_indices: np.ndarray,
    pt_positions: np.ndarray,
    enc_positions: np.ndarray,
):
    # Per-document MAE across all embedding dimensions.
    per_doc_mae = np.abs(pt_vecs - enc_vecs).mean(axis=1)
    overall_mae  = np.abs(pt_vecs - enc_vecs).mean()
    overall_max  = np.abs(pt_vecs - enc_vecs).max()

    print(
        "\nVector diff (|pt_vec - enc_vec|), aligned by document index "
        f"[shape={pt_vecs.shape}]:"
    )
    print(
        f"  {'doc':>4}  {'pt_rank':>7}  {'enc_rank':>8}  "
        f"{'MAE':>12}  {'max|diff|':>12}"
    )
    for row, (doc_idx, pt_pos, enc_pos, mae) in enumerate(
        zip(doc_indices, pt_positions, enc_positions, per_doc_mae)
    ):
        max_d = np.abs(pt_vecs[row] - enc_vecs[row]).max()
        print(
            f"  {doc_idx:4d}  {pt_pos + 1:7d}  {enc_pos + 1:8d}  "
            f"{mae:12.3e}  {max_d:12.3e}"
        )

    print(f"\n  Overall MAE across aligned vector elements : {overall_mae:.3e}")
    print(f"  Overall max |diff|                        : {overall_max:.3e}")


def run(pt_path: Path, enc_path: Path):
    pt  = load(pt_path)
    enc = load(enc_path)

    pt_vecs  = pt["vectors"].astype(np.float64)
    enc_vecs = enc["vectors"].astype(np.float64)
    pt_idx   = pt["indices"]
    enc_idx  = enc["indices"]

    k_pt, dim_pt   = pt_vecs.shape
    k_enc, dim_enc = enc_vecs.shape

    print(f"Plaintext  : {pt_path}   shape={pt_vecs.shape}")
    print(f"Encrypted  : {enc_path}   shape={enc_vecs.shape}")

    if k_pt != k_enc or dim_pt != dim_enc:
        print(f"\n[ERROR] Shape mismatch: plaintext {pt_vecs.shape} vs encrypted {enc_vecs.shape}")
        return

    print()
    print_index_overlap(pt_idx, enc_idx)

    common_docs, pt_positions, enc_positions = align_by_document(pt_idx, enc_idx)
    if len(common_docs) == 0:
        print("\n[diff] skipped - the Top-k result sets have no common documents")
        return

    pt_aligned_vecs = pt_vecs[pt_positions]
    enc_aligned_vecs = enc_vecs[enc_positions]

    # Score diff — optional (skip if either file lacks scores)
    if "scores" in pt and "scores" in enc:
        pt_scores = pt["scores"].astype(np.float64)[pt_positions]
        enc_scores = enc["scores"].astype(np.float64)[enc_positions]
        print_score_diff(
            pt_scores,
            enc_scores,
            common_docs,
            pt_positions,
            enc_positions,
        )
    else:
        missing = []
        if "scores" not in pt:
            missing.append(str(pt_path))
        if "scores" not in enc:
            missing.append(str(enc_path))
        print(f"\n[scores] skipped — missing in: {', '.join(missing)}")

    print_vector_diff(
        pt_aligned_vecs,
        enc_aligned_vecs,
        common_docs,
        pt_positions,
        enc_positions,
    )


def main():
    parser = argparse.ArgumentParser(description="Compare plaintext vs encrypted top-k retrieval vectors.")
    parser.add_argument("pt_vectors",  type=Path,
                        help=".npy file from rag.py --save-vectors (plaintext)")
    parser.add_argument("enc_vectors", type=Path,
                        help=".npy file from gpu_ckks_rag.py or benchmark_dot_product.py --save-vectors")
    args = parser.parse_args()
    run(args.pt_vectors, args.enc_vectors)


if __name__ == "__main__":
    main()
