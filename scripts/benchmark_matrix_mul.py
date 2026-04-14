from pathlib import Path
import sqlite3
import time
import argparse
import numpy as np
from desilofhe import Engine


ROOT = Path(__file__).resolve().parent.parent
REPEAT = 1
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


def create_selector_vector_row(i: int, n: int):
    vec = np.zeros(n, dtype=np.float64)
    vec[i] = 1.0
    return vec


def create_selector_vector_encrypted(target_indices, n: int, engine, public_key):
    selectors = []
    for i in target_indices:
        vec = create_selector_vector_row(i, n)
        ct = engine.encrypt(vec.tolist(), public_key)
        engine.to_cuda(ct)
        selectors.append(ct)
    return selectors


def decrypt_first_scalar(ct_obj, engine: Engine, secret_key):
    arr = engine.decrypt(ct_obj, secret_key)
    return float(arr[0])


def sum_first_n_slots(ct, n, engine, rot_key):
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


def mm_naive_clean(encrypted_selectors, t_encrypted_db_cts, dim, engine, rel_key, rot_key):
    # Only the first and last ciphertext results are retained to avoid
    # holding all intermediate results in GPU memory simultaneously.
    first_result = None
    last_result = None

    for r, enc_sel in enumerate(encrypted_selectors):
        row_start = time.perf_counter()

        for ct_db in t_encrypted_db_cts:
            result = encrypted_dot(enc_sel, ct_db, dim, engine, rel_key, rot_key)

            if first_result is None:
                first_result = result
            last_result = result

        print(f"  [mm] row {r} done in {time.perf_counter() - row_start:.2f}s", flush=True)

    return first_result, last_result


def benchmark_mm_single(dim: int, top_k: int, repeat: int):
    base = ROOT / "data" / str(dim)
    key_dir = base / "keys"
    db_dir = base / "db"
    trow_db_path = db_dir / "t_row_encrypted.db"

    if not trow_db_path.exists():
        raise FileNotFoundError(f"Missing DB file: {trow_db_path}")

    engine = Engine(slot_count=dim, mode="gpu")
    secret_key = engine.read_secret_key(str(key_dir / "secret.key"))
    public_key = engine.read_public_key(str(key_dir / "public.key"))
    rel_key = engine.read_relinearization_key(str(key_dir / "relin.key"))
    rot_key = engine.read_rotation_key(str(key_dir / "rotation.key"))

    print(f"\n[DIM={dim}] loading transposed encrypted DB...")
    t_encrypted_db_cts = load_ciphertexts_sqlite(trow_db_path, engine, "encrypted_vectors")
    doc_count = len(t_encrypted_db_cts)
    print(f"[DIM={dim}] loaded ciphertext rows: {doc_count}")

    if top_k > doc_count:
        raise ValueError(f"top_k={top_k} > doc_count={doc_count}")

    top_k_indices = list(range(top_k))
    enc_selectors = create_selector_vector_encrypted(top_k_indices, doc_count, engine, public_key)

    s_mb = calc_mb(top_k, doc_count)
    k_mb = calc_mb(dim, dim)

    mm_times = []
    for i in range(repeat):
        print(f"[run {i+1}/{repeat}] mm naive", flush=True)
        start = time.perf_counter()
        mm_first, mm_last = mm_naive_clean(
            enc_selectors,
            t_encrypted_db_cts,
            dim,
            engine,
            rel_key,
            rot_key,
        )
        end = time.perf_counter()
        mm_times.append(end - start)

        first_val = decrypt_first_scalar(mm_first, engine, secret_key)
        last_val = decrypt_first_scalar(mm_last, engine, secret_key)
        print(f"  [mm] decrypt sanity: first={first_val}, last={last_val}", flush=True)

    avg_mm = sum(mm_times) / len(mm_times)

    print(
        f"\n[FINAL] DIM={dim} | top_k={top_k} | "
        f"s={s_mb:.6f} MB | K={k_mb:.6f} MB | MM Naive={avg_mm:.6f} sec"
    )

    print("\n| Embedding dimension | Top-k | s (MB) | K (MB) | MM Naive (sec) |")
    print("|---:|---:|---:|---:|---:|")
    print(f"| {dim} | {top_k} | {s_mb:.6f} | {k_mb:.6f} | {avg_mm:.6f} |")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, required=True, choices=[256, 512])
    parser.add_argument("--topk", type=int, required=True)
    parser.add_argument("--repeat", type=int, default=REPEAT)
    args = parser.parse_args()

    benchmark_mm_single(args.dim, args.topk, args.repeat)


if __name__ == "__main__":
    main()