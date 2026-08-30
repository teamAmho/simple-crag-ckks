from pathlib import Path
import sqlite3
import time
import argparse
import statistics
import numpy as np
import chromadb.utils.embedding_functions as embedding_functions
from desilofhe import Engine


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPEAT = 5
BYTES_PER_ELEM = 8  # float64


def calc_mb(rows: int, cols: int, bytes_per_elem: int = BYTES_PER_ELEM) -> float:
    return rows * cols * bytes_per_elem / (1024 ** 2)


def load_ciphertexts_sqlite(db_path: Path, engine: Engine, table_name: str = "encrypted_vectors"):
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute(f"SELECT id, ciphertext FROM {table_name} ORDER BY CAST(id AS INTEGER)")
    rows = cursor.fetchall()
    conn.close()

    cts = []
    for _, blob in rows:
        ct = engine.deserialize_ciphertext(blob)
        engine.to_cuda(ct)
        cts.append(ct)
    return cts


def get_query_embedding(dim: int, query: str):
    model = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="nomic-ai/nomic-embed-text-v1.5",
        device="cpu",
        normalize_embeddings=True,
        trust_remote_code=True,
        truncate_dim=dim,
    )

    emb = model([query])[0]
    return np.array(emb, dtype=np.float64)


def decrypt_first_scalar(ct_obj, engine: Engine, secret_key):
    arr = engine.decrypt(ct_obj, secret_key)
    return float(arr[0])


def sum_first_n_slots(ct, n, engine, rot_key):
    # Sum the first n slots via repeated rotation-and-add (log2 steps).
    # Rotates by powers of 2 up to the next power of 2 >= n.
    dim2 = 1 << (n - 1).bit_length()
    acc = ct
    step = 1
    while step < dim2:
        acc = engine.add(acc, engine.rotate(acc, rot_key, step))
        step <<= 1
    return acc


def encrypted_dot(ct_a, ct_b, n, engine, rel_key, rot_key):
    ct_mul = engine.multiply(ct_a, ct_b, rel_key)
    ct_sum = sum_first_n_slots(ct_mul, n, engine, rot_key)
    return ct_sum


def dot_product_clean(ct_query, row_db_cts, dim, engine, rel_key, rot_key, show_progress=False):
    # Only the first and last ciphertext results are retained to avoid
    # holding all intermediate results in GPU memory simultaneously.
    first_result = None
    last_result = None

    total = len(row_db_cts)
    for idx, ct_db in enumerate(row_db_cts):
        if show_progress and idx % 32 == 0:
            print(f"  [dot] progress: {idx}/{total}", flush=True)

        result = encrypted_dot(ct_query, ct_db, dim, engine, rel_key, rot_key)

        if first_result is None:
            first_result = result
        last_result = result

    return first_result, last_result


def benchmark_dot_single(dim: int, top_k: int, repeat: int, query: str = "What is bezoar?"):
    base = ROOT / "data" / str(dim)
    key_dir = base / "keys"
    db_dir = base / "db"
    row_db_path = db_dir / "row_encrypted.db"

    if not row_db_path.exists():
        raise FileNotFoundError(f"Missing DB file: {row_db_path}")

    engine = Engine(slot_count=dim, mode="gpu")
    secret_key = engine.read_secret_key(str(key_dir / "secret.key"))
    public_key = engine.read_public_key(str(key_dir / "public.key"))
    rel_key = engine.read_relinearization_key(str(key_dir / "relin.key"))
    rot_key = engine.read_rotation_key(str(key_dir / "rotation.key"))

    print(f"\n[DIM={dim}] loading row-oriented encrypted document DB...")
    row_db_cts = load_ciphertexts_sqlite(row_db_path, engine, "encrypted_vectors")
    doc_count = len(row_db_cts)
    print(f"[DIM={dim}] loaded document ciphertexts: {doc_count}")
    print("[note] Top-k does not affect encrypted similarity computation time.")

    if top_k > doc_count:
        raise ValueError(f"top_k={top_k} > doc_count={doc_count}")

    q = get_query_embedding(dim, query)
    ct_query = engine.encrypt(q.tolist(), public_key)
    engine.to_cuda(ct_query)

    dot_times = []
    warmup_time = None

    for i in range(repeat):
        is_warmup = (i == 0)
        tag = "warm-up" if is_warmup else f"run {i}/{repeat-1}"
        print(f"[{tag}] dot product", flush=True)

        start = time.perf_counter()
        dot_first, dot_last = dot_product_clean(
            ct_query,
            row_db_cts,
            dim,
            engine,
            rel_key,
            rot_key,
            show_progress=is_warmup,
        )
        end = time.perf_counter()
        elapsed = end - start

        first_val = decrypt_first_scalar(dot_first, engine, secret_key)
        last_val = decrypt_first_scalar(dot_last, engine, secret_key)
        print(f"  [dot] decrypt sanity: first={first_val}, last={last_val}", flush=True)
        print(f"  [dot] elapsed: {elapsed:.6f} sec", flush=True)

        if is_warmup:
            warmup_time = elapsed
        else:
            dot_times.append(elapsed)

    if not dot_times:
        raise RuntimeError("No measured runs available. Use repeat >= 2.")

    avg_dot = sum(dot_times) / len(dot_times)
    median_dot = statistics.median(dot_times)

    s_mb = calc_mb(top_k, doc_count)
    k_mb = calc_mb(dim, dim)

    print(
        f"\n[FINAL] DIM={dim} | top_k={top_k} | "
        f"s={s_mb:.6f} MB | K={k_mb:.6f} MB | "
        f"warmup={warmup_time:.6f} sec | avg={avg_dot:.6f} sec | median={median_dot:.6f} sec"
    )

    print("\n| Embedding dimension | Top-k | s (MB) | K (MB) | Dot product avg (sec) | Dot product median (sec) |")
    print("|---:|---:|---:|---:|---:|---:|")
    print(f"| {dim} | {top_k} | {s_mb:.6f} | {k_mb:.6f} | {avg_dot:.6f} | {median_dot:.6f} |")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, required=True, choices=[256, 512])
    parser.add_argument("--topk", type=int, required=True)
    parser.add_argument("--repeat", type=int, default=DEFAULT_REPEAT,
                        help="Total runs including 1 warm-up. Use at least 2.")
    parser.add_argument("--query", type=str, default="What is bezoar?",
                        help="Query text to embed and use as the encrypted search vector.")
    args = parser.parse_args()

    benchmark_dot_single(args.dim, args.topk, args.repeat, args.query)


if __name__ == "__main__":
    main()
