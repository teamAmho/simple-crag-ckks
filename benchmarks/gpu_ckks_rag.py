"""
gpu_ckks_rag.py — Privacy-preserving RAG with CKKS homomorphic encryption (local version)

Subcommands:
  build     Chunk doc → embed → ChromaDB → encrypt → SQLite DBs
  retrieve  Encrypt query → encrypted inner products → top-k → encrypted matmul → decrypt

Usage:
  python gpu_ckks_rag.py build --doc cat-facts.txt
  python gpu_ckks_rag.py retrieve --query "What is bezoar?" --topk 4 --save-vectors enc_matmul.npy
"""
from pathlib import Path
import argparse
import os
import sqlite3

import numpy as np
import torch
from desilofhe import Engine
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
import chromadb.utils.embedding_functions as embedding_functions


MODEL_NAME            = "nomic-ai/nomic-embed-text-v1.5"
DEFAULT_DIM           = 256
DEFAULT_TOPK          = 4
DEFAULT_CHUNK_SIZE    = 600
DEFAULT_CHUNK_OVERLAP = 0
COLLECTION_NAME       = "nomic_v1.5_db_collection"


# ── SQLite helpers ────────────────────────────────────────────────────────────

def save_ciphertexts_sqlite(db_path: Path, ids, blobs, table="encrypted_vectors"):
    conn = sqlite3.connect(str(db_path))
    cur  = conn.cursor()
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {table} (
            id TEXT PRIMARY KEY,
            ciphertext BLOB
        )
    """)
    cur.executemany(
        f"INSERT OR REPLACE INTO {table} (id, ciphertext) VALUES (?, ?)",
        [(doc_id, sqlite3.Binary(b)) for doc_id, b in zip(ids, blobs)],
    )
    conn.commit()
    conn.close()


def load_ciphertexts_sqlite(db_path: Path, engine: Engine, table="encrypted_vectors"):
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur  = conn.cursor()
    cur.execute(f"SELECT id, ciphertext FROM {table} ORDER BY CAST(id AS INTEGER)")
    out = []
    for row in cur.fetchall():
        ct = engine.deserialize_ciphertext(row["ciphertext"])
        engine.to_cuda(ct)
        out.append(ct)
    conn.close()
    return out


# ── Key management ────────────────────────────────────────────────────────────

def _load_or_create(path: Path, load_fn, create_fn, save_fn):
    if path.exists():
        return load_fn(path.read_bytes())
    obj = create_fn()
    path.write_bytes(save_fn(obj))
    return obj


def load_or_create_keys(key_dir: Path, engine: Engine):
    key_dir.mkdir(parents=True, exist_ok=True)
    secret_key = _load_or_create(
        key_dir / "secret.key",
        engine.deserialize_secret_key,
        engine.create_secret_key,
        engine.serialize_secret_key,
    )
    public_key = _load_or_create(
        key_dir / "public.key",
        engine.deserialize_public_key,
        lambda: engine.create_public_key(secret_key),
        engine.serialize_public_key,
    )
    rel_key = _load_or_create(
        key_dir / "relin.key",
        engine.deserialize_relinearization_key,
        lambda: engine.create_relinearization_key(secret_key),
        engine.serialize_relinearization_key,
    )
    rot_key = _load_or_create(
        key_dir / "rotation.key",
        engine.deserialize_rotation_key,
        lambda: engine.create_rotation_key(secret_key),
        engine.serialize_rotation_key,
    )
    return secret_key, public_key, rel_key, rot_key


# ── Embedding ─────────────────────────────────────────────────────────────────

def get_emb_fn(dim: int, device: str):
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=MODEL_NAME,
        device=device,
        normalize_embeddings=True,
        trust_remote_code=True,
        truncate_dim=dim,
    )


# ── FHE core operations ───────────────────────────────────────────────────────

def sum_first_n_slots(ct, n: int, engine: Engine, rot_key):
    """Rotate-and-add: broadcasts sum of first n slots to slot 0."""
    dim2 = 1 << (n - 1).bit_length()
    acc  = ct
    step = 1
    while step < dim2:
        acc  = engine.add(acc, engine.rotate(acc, rot_key, step))
        step <<= 1
    return acc


def encrypted_dot(ct_a, ct_b, n: int, engine: Engine, rel_key, rot_key):
    ct_mul = engine.multiply(ct_a, ct_b, rel_key)
    return sum_first_n_slots(ct_mul, n, engine, rot_key)


def encrypted_inner_product(ct_q, ct_db, dim: int, engine: Engine, rel_key, rot_key):
    """Element-wise multiply then rotate-and-add to get dot product in slot 0."""
    ct_mul = engine.multiply(ct_q, ct_db, rel_key)
    dim2   = 1 << (dim - 1).bit_length()
    step   = 1
    while step < dim2:
        ct_rot = engine.rotate(ct_mul, rot_key, step)
        ct_mul = engine.add(ct_mul, ct_rot)
        step <<= 1
    return ct_mul


def encrypted_matmul(sel_cts, col_db_cts, n: int, engine: Engine, rel_key, rot_key):
    """
    Encrypted (k × slot_count selector) × (slot_count col-DB) → C[r][c] ciphertext.
    Slot 0 of C[r][c] holds doc[top_k_indices[r]][c] after decryption.
    """
    k = len(sel_cts)
    m = len(col_db_cts)
    C = [[None] * m for _ in range(k)]
    for r in range(k):
        for c in range(m):
            if c % 64 == 0:
                print(f"  [matmul] row {r+1}/{k}  col {c}/{m}", flush=True)
            C[r][c] = encrypted_dot(sel_cts[r], col_db_cts[c], n, engine, rel_key, rot_key)
    return C


def create_encrypted_selector(top_k_indices, n: int, engine: Engine, public_key):
    """Encrypt one-hot selector vectors for each top-k document index."""
    sel_cts = []
    for idx in top_k_indices:
        vec      = np.zeros(n, dtype=np.float64)
        vec[idx] = 1.0
        ct = engine.encrypt(vec.tolist(), public_key)
        engine.to_cuda(ct)
        sel_cts.append(ct)
    return sel_cts


# ── build subcommand ──────────────────────────────────────────────────────────

def cmd_build(args):
    data_dir   = args.data_dir
    key_dir    = data_dir / "keys"
    db_dir     = data_dir / "db"
    chroma_dir = data_dir / "chroma"
    db_dir.mkdir(parents=True, exist_ok=True)
    chroma_dir.mkdir(parents=True, exist_ok=True)

    dim        = args.dim
    slot_count = dim
    device     = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[build] device={device}  dim={dim}")
    print(f"[build] doc={args.doc}")

    engine = Engine(slot_count=dim, mode="gpu")
    secret_key, public_key, _, _ = load_or_create_keys(key_dir, engine)
    print(f"[build] keys ready  ({key_dir})")

    # Load and chunk
    loader   = TextLoader(str(args.doc))
    raw      = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    docs   = splitter.split_documents(raw)
    texts  = [d.page_content for d in docs]
    n_docs = len(texts)
    print(f"[build] {n_docs} chunks")

    # Embed
    emb_fn     = get_emb_fn(dim, device)
    embeddings = emb_fn(texts)   # (n_docs, dim), L2-normalized
    print(f"[build] embedded {n_docs} docs  (dim={dim})")

    # ChromaDB
    client = chromadb.PersistentClient(path=str(chroma_dir))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    col = client.create_collection(name=COLLECTION_NAME, embedding_function=emb_fn)
    col.add(ids=[str(i) for i in range(n_docs)], documents=texts)
    print(f"[build] ChromaDB saved  ({chroma_dir})")

    # Pad to (n_docs, slot_count)
    padded = np.zeros((n_docs, slot_count), dtype=np.float64)
    for i, emb in enumerate(embeddings):
        padded[i, : len(emb)] = emb

    # Row DB: ciphertext i = document i's full embedding
    row_db_path = db_dir / "row_encrypted.db"
    if row_db_path.exists():
        row_db_path.unlink()
    row_blobs = []
    for row in padded:
        ct   = engine.encrypt(row.tolist(), public_key)
        row_blobs.append(engine.serialize_ciphertext(ct))
    save_ciphertexts_sqlite(row_db_path, [str(i) for i in range(n_docs)], row_blobs)
    print(f"[build] row DB  → {row_db_path}  ({n_docs} ciphertexts)")

    # Col DB: ciphertext j = all documents' j-th dimension value (padded to slot_count)
    col_db_path = db_dir / "col_encrypted.db"
    if col_db_path.exists():
        col_db_path.unlink()
    col_blobs = []
    for j in range(slot_count):
        col_vec          = np.zeros(slot_count, dtype=np.float64)
        col_vec[:n_docs] = padded[:, j]
        ct               = engine.encrypt(col_vec.tolist(), public_key)
        col_blobs.append(engine.serialize_ciphertext(ct))
    save_ciphertexts_sqlite(col_db_path, [str(j) for j in range(slot_count)], col_blobs)
    print(f"[build] col DB  → {col_db_path}  ({slot_count} ciphertexts)")
    print("[build] done.")


# ── retrieve subcommand ───────────────────────────────────────────────────────

def cmd_retrieve(args):
    data_dir    = args.data_dir
    key_dir     = data_dir / "keys"
    db_dir      = data_dir / "db"
    row_db_path = db_dir / "row_encrypted.db"
    col_db_path = db_dir / "col_encrypted.db"
    dim         = args.dim
    top_k       = args.topk
    slot_count  = dim
    device      = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[retrieve] device={device}  dim={dim}  top_k={top_k}")
    print(f"[retrieve] query={args.query!r}\n")

    engine = Engine(slot_count=dim, mode="gpu")
    secret_key, public_key, rel_key, rot_key = load_or_create_keys(key_dir, engine)

    print("[retrieve] loading row DB...")
    row_db_cts = load_ciphertexts_sqlite(row_db_path, engine)
    n_docs     = len(row_db_cts)
    print(f"[retrieve] {n_docs} document ciphertexts loaded")
    if not 1 <= top_k <= n_docs:
        raise ValueError(f"top_k must be between 1 and {n_docs}, got {top_k}")

    print("[retrieve] loading col DB...")
    col_db_cts = load_ciphertexts_sqlite(col_db_path, engine)
    print(f"[retrieve] {len(col_db_cts)} column ciphertexts loaded\n")

    # Step 1: Encrypt query
    emb_fn  = get_emb_fn(dim, device)
    q_emb   = np.array(emb_fn([args.query])[0], dtype=np.float64)
    ct_q    = engine.encrypt(q_emb.tolist(), public_key)
    engine.to_cuda(ct_q)
    print("[retrieve] query encrypted")

    # Step 2: Encrypted inner products (query × each document)
    print(f"[retrieve] computing encrypted inner products ({n_docs} docs)...")
    scores = []
    for i, ct_db in enumerate(row_db_cts):
        if i % 16 == 0:
            print(f"  [ip] {i}/{n_docs}", flush=True)
        ct_ip = encrypted_inner_product(ct_q, ct_db, dim, engine, rel_key, rot_key)
        scores.append(float(engine.decrypt(ct_ip, secret_key)[0]))

    # Step 3: Top-K selection (client-side, decrypt scores)
    ranked      = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
    top_indices = [idx for idx, _ in ranked]
    top_scores  = [s   for _,   s in ranked]

    print(f"\n=== Step 3: Top-{top_k} (encrypted inner product) ===")
    for rank, (idx, score) in enumerate(ranked, 1):
        print(f"  [{rank:2d}] doc_idx={idx:3d}  score={score:.6f}")

    # Step 4: Encrypted selector vectors
    print(f"\n[retrieve] creating {top_k} encrypted selector vectors...")
    # Selector length = slot_count; doc indices are within [0, n_docs-1] < slot_count
    sel_cts = create_encrypted_selector(top_indices, slot_count, engine, public_key)

    # Step 5: Encrypted matmul → retrieve top-k document vectors
    print(f"[retrieve] encrypted matmul ({top_k} × {slot_count})...")
    matmul_res = encrypted_matmul(sel_cts, col_db_cts, slot_count, engine, rel_key, rot_key)

    # Decrypt matmul results → (k, dim)
    print("[retrieve] decrypting retrieved vectors...")
    retrieved = np.zeros((top_k, dim), dtype=np.float64)
    for r, row_cts in enumerate(matmul_res):
        for c, ct in enumerate(row_cts):
            retrieved[r, c] = engine.decrypt(ct, secret_key)[0]

    print(f"\n=== Step 5: Decrypted top-{top_k} document vectors ===")
    print(f"  shape: {retrieved.shape}")
    for rank in range(top_k):
        print(f"  [{rank+1:2d}] doc_idx={top_indices[rank]:3d}  first 8 dims: {retrieved[rank, :8]}")

    # Save for compare.py
    if args.save_vectors is not None:
        out = {
            "vectors": retrieved,
            "indices": np.array(top_indices, dtype=np.int64),
            "scores":  np.array(top_scores,  dtype=np.float64),
        }
        np.save(str(args.save_vectors), out, allow_pickle=True)
        print(f"\n[save-vectors] saved → {args.save_vectors}  shape={retrieved.shape}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    default_data = Path(__file__).resolve().parent / "data" / str(DEFAULT_DIM)

    parser = argparse.ArgumentParser(
        description="Privacy-preserving RAG with CKKS homomorphic encryption"
    )
    parser.add_argument("--data-dir", type=Path, default=default_data,
                        help=f"Root data directory (default: {default_data})")
    parser.add_argument("--dim", type=int, default=DEFAULT_DIM,
                        help=f"Embedding dim / CKKS slot count (default: {DEFAULT_DIM})")

    sub = parser.add_subparsers(dest="cmd", required=True)

    # build
    bp = sub.add_parser("build", help="Build encrypted DBs from a text document")
    bp.add_argument("--doc", type=Path, required=True,
                    help="Input .txt file (e.g. cat-facts.txt)")
    bp.add_argument("--chunk-size",    type=int, default=DEFAULT_CHUNK_SIZE,
                    help=f"LangChain chunk size (default: {DEFAULT_CHUNK_SIZE})")
    bp.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP,
                    help=f"LangChain chunk overlap (default: {DEFAULT_CHUNK_OVERLAP})")

    # retrieve
    rp = sub.add_parser("retrieve", help="Run encrypted retrieval against existing DBs")
    rp.add_argument("--query", type=str, default="What is bezoar?",
                    help="Query text (default: 'What is bezoar?')")
    rp.add_argument("--topk", type=int, default=DEFAULT_TOPK,
                    help=f"Number of results to retrieve (default: {DEFAULT_TOPK})")
    rp.add_argument("--save-vectors", type=Path, default=None, metavar="PATH",
                    help="Save decrypted top-k vectors to .npy for compare.py")

    args = parser.parse_args()
    if args.cmd == "build":
        cmd_build(args)
    else:
        cmd_retrieve(args)


if __name__ == "__main__":
    main()
