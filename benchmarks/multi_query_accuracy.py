"""Evaluate plaintext/CKKS ranking agreement for multiple queries.

This benchmark intentionally skips selector-matrix document recovery. The
single-query grid evaluates that stage separately; here, each encrypted score
vector is reused to evaluate every requested Top-k cutoff.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import chromadb
import numpy as np
import torch
from desilofhe import Engine

from gpu_ckks_rag_streaming import (
    COLLECTION_NAME,
    encrypted_inner_product,
    get_emb_fn,
    load_ciphertexts_sqlite,
    load_or_create_keys,
)


DEFAULT_TOPKS = (4, 8, 16, 32)


def load_queries(path: Path) -> list[str]:
    queries = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    queries = [query for query in queries if query and not query.startswith("#")]
    if not queries:
        raise ValueError(f"No queries found in {path}")
    if len(set(queries)) != len(queries):
        raise ValueError(f"Duplicate queries found in {path}")
    return queries


def load_plaintext_embeddings(chroma_dir: Path, emb_fn) -> np.ndarray:
    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=emb_fn,
    )
    records = collection.get(include=["embeddings"])
    order = sorted(range(len(records["ids"])), key=lambda i: int(records["ids"][i]))
    return np.asarray([records["embeddings"][i] for i in order], dtype=np.float32)


def rank(scores: np.ndarray) -> np.ndarray:
    # Stable sorting makes tie handling deterministic across both paths.
    return np.argsort(-scores, kind="stable")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def aggregate(detail_rows: list[dict], dim: int, topks: list[int]) -> list[dict]:
    summary = []
    for top_k in topks:
        rows = [row for row in detail_rows if row["top_k"] == top_k]
        query_count = len(rows)
        total_positions = query_count * top_k
        summary.append(
            {
                "dim": dim,
                "top_k": top_k,
                "queries": query_count,
                "mean_topk_agreement": sum(row["overlap"] for row in rows)
                / total_positions,
                "rank_agreement": sum(row["rank_matches"] for row in rows)
                / total_positions,
                "full_rank_match_rate": np.mean(
                    [row["full_rank_match"] for row in rows]
                ),
                "mean_score_diff": np.mean(
                    [row["mean_score_diff"] for row in rows]
                ),
                "max_score_diff": max(row["max_score_diff"] for row in rows),
                "min_boundary_gap": min(row["boundary_gap"] for row in rows),
                "min_gap_to_2maxerr_ratio": min(
                    row["gap_to_2maxerr_ratio"] for row in rows
                ),
                "all_boundaries_stable": all(
                    row["boundary_stable"] for row in rows
                ),
            }
        )
    return summary


def print_summary(rows: list[dict]) -> None:
    print("\n=== Multi-query summary ===")
    print(
        " dim    k  queries  top-k agree  rank agree  full match  "
        "mean score diff  max score diff  min gap  min gap/(2e)"
    )
    for row in rows:
        print(
            f"{row['dim']:4d}  {row['top_k']:3d}  {row['queries']:7d}  "
            f"{row['mean_topk_agreement']:11.6f}  "
            f"{row['rank_agreement']:10.6f}  "
            f"{row['full_rank_match_rate']:10.6f}  "
            f"{row['mean_score_diff']:15.3e}  "
            f"{row['max_score_diff']:14.3e}  "
            f"{row['min_boundary_gap']:7.3e}  "
            f"{row['min_gap_to_2maxerr_ratio']:12.3e}"
        )


def run(args: argparse.Namespace) -> None:
    dim = args.dim
    data_dir = args.data_dir or Path("data") / str(dim)
    key_dir = data_dir / "keys"
    row_db_path = data_dir / "db" / "row_encrypted.db"
    chroma_dir = data_dir / "chroma"
    queries = load_queries(args.queries_file)
    topks = sorted(set(args.topks))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("This benchmark requires CUDA for the DESILO GPU engine")

    print(f"[setup] dim={dim} device={device} queries={len(queries)} topks={topks}")
    engine = Engine(slot_count=dim, mode="gpu")
    secret_key, public_key, rel_key, rot_key = load_or_create_keys(key_dir, engine)
    row_db_cts = load_ciphertexts_sqlite(row_db_path, engine)
    n_docs = len(row_db_cts)
    if max(topks) >= n_docs:
        raise ValueError(f"max top_k={max(topks)} requires more than {n_docs} documents")

    emb_fn = get_emb_fn(dim, device)
    embeddings = load_plaintext_embeddings(chroma_dir, emb_fn)
    if len(embeddings) != n_docs:
        raise ValueError(
            f"Plaintext/encrypted DB size mismatch: {len(embeddings)} vs {n_docs}"
        )

    detail_rows: list[dict] = []
    score_rows: list[dict] = []
    started = time.perf_counter()

    for query_id, query in enumerate(queries, start=1):
        query_started = time.perf_counter()
        query_emb = np.asarray(emb_fn([query])[0], dtype=np.float64)

        db_tensor = torch.as_tensor(embeddings, dtype=torch.float32, device=device)
        query_tensor = torch.as_tensor(query_emb, dtype=torch.float32, device=device)
        plaintext_scores = (db_tensor @ query_tensor).cpu().numpy().astype(np.float64)

        encrypted_query = engine.encrypt(query_emb.tolist(), public_key)
        engine.to_cuda(encrypted_query)
        encrypted_scores = np.empty(n_docs, dtype=np.float64)
        for doc_idx, encrypted_doc in enumerate(row_db_cts):
            result = encrypted_inner_product(
                encrypted_query,
                encrypted_doc,
                dim,
                engine,
                rel_key,
                rot_key,
            )
            encrypted_scores[doc_idx] = float(engine.decrypt(result, secret_key)[0])

        score_diff = np.abs(plaintext_scores - encrypted_scores)
        plaintext_order = rank(plaintext_scores)
        encrypted_order = rank(encrypted_scores)

        plaintext_rank = np.empty(n_docs, dtype=np.int64)
        encrypted_rank = np.empty(n_docs, dtype=np.int64)
        plaintext_rank[plaintext_order] = np.arange(1, n_docs + 1)
        encrypted_rank[encrypted_order] = np.arange(1, n_docs + 1)
        for doc_idx in range(n_docs):
            score_rows.append(
                {
                    "query_id": query_id,
                    "query": query,
                    "dim": dim,
                    "doc_idx": doc_idx,
                    "plaintext_score": plaintext_scores[doc_idx],
                    "encrypted_score": encrypted_scores[doc_idx],
                    "abs_score_diff": score_diff[doc_idx],
                    "plaintext_rank": int(plaintext_rank[doc_idx]),
                    "encrypted_rank": int(encrypted_rank[doc_idx]),
                }
            )

        max_all_score_diff = float(score_diff.max())
        for top_k in topks:
            plaintext_top = plaintext_order[:top_k]
            encrypted_top = encrypted_order[:top_k]
            overlap = len(set(plaintext_top.tolist()) & set(encrypted_top.tolist()))
            rank_matches = int(np.sum(plaintext_top == encrypted_top))
            boundary_gap = float(
                plaintext_scores[plaintext_order[top_k - 1]]
                - plaintext_scores[plaintext_order[top_k]]
            )
            denominator = 2.0 * max_all_score_diff
            ratio = boundary_gap / denominator if denominator else float("inf")
            detail_rows.append(
                {
                    "query_id": query_id,
                    "query": query,
                    "dim": dim,
                    "top_k": top_k,
                    "overlap": overlap,
                    "rank_matches": rank_matches,
                    "full_rank_match": int(rank_matches == top_k),
                    "mean_score_diff": float(score_diff[plaintext_top].mean()),
                    "max_score_diff": float(score_diff[plaintext_top].max()),
                    "max_all_score_diff": max_all_score_diff,
                    "boundary_gap": boundary_gap,
                    "gap_to_2maxerr_ratio": ratio,
                    "boundary_stable": int(
                        encrypted_scores[plaintext_order[top_k - 1]]
                        > encrypted_scores[plaintext_order[top_k]]
                    ),
                }
            )

        elapsed = time.perf_counter() - query_started
        print(
            f"[query {query_id}/{len(queries)}] {query!r} "
            f"max_score_diff={max_all_score_diff:.3e} elapsed={elapsed:.1f}s",
            flush=True,
        )

    summary_rows = aggregate(detail_rows, dim, topks)
    output_dir = args.output_dir
    detail_path = output_dir / f"multi_query_dim{dim}_detail.csv"
    scores_path = output_dir / f"multi_query_dim{dim}_scores.csv"
    summary_path = output_dir / f"multi_query_dim{dim}_summary.csv"
    metadata_path = output_dir / f"multi_query_dim{dim}_metadata.json"

    write_csv(detail_path, detail_rows, list(detail_rows[0].keys()))
    write_csv(scores_path, score_rows, list(score_rows[0].keys()))
    write_csv(summary_path, summary_rows, list(summary_rows[0].keys()))
    metadata_path.write_text(
        json.dumps(
            {
                "dim": dim,
                "topks": topks,
                "queries": queries,
                "document_count": n_docs,
                "elapsed_seconds": time.perf_counter() - started,
                "score_difference_alignment": "document index",
                "boundary_ratio": "plaintext gap / (2 * max all-document score error)",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print_summary(summary_rows)
    print(f"\n[save] {detail_path}")
    print(f"[save] {scores_path}")
    print(f"[save] {summary_path}")
    print(f"[save] {metadata_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", type=int, choices=(256, 512), required=True)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument(
        "--queries-file",
        type=Path,
        default=Path(__file__).with_name("multi_query_queries.txt"),
    )
    parser.add_argument("--topks", type=int, nargs="+", default=list(DEFAULT_TOPKS))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("results"),
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
