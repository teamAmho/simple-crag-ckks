from pathlib import Path
import argparse
import numpy as np
import torch
import chromadb
import chromadb.utils.embedding_functions as embedding_functions


DEFAULT_DIM = 256
DEFAULT_TOP_K = 10
DEFAULT_COLLECTION = "nomic_v1.5_db_collection"


def get_embedding_function(dim: int, device: str):
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="nomic-ai/nomic-embed-text-v1.5",
        device=device,
        normalize_embeddings=True,
        trust_remote_code=True,
        truncate_dim=dim,
    )


def load_chroma_collection(chroma_dir: Path, collection_name: str, emb_fn):
    client = chromadb.PersistentClient(path=str(chroma_dir))
    return client.get_collection(name=collection_name, embedding_function=emb_fn)


def create_selector_matrix(indices, k: int, n: int) -> np.ndarray:
    mat = np.zeros((k, n), dtype=np.float64)
    for i, idx in enumerate(indices):
        if idx < n:
            mat[i, idx] = 1.0
    return mat


def dot_product_topk(embeddings, query_emb: np.ndarray, dim: int, top_k: int, device: str):
    db_t = torch.tensor(embeddings, dtype=torch.float32).to(device)
    q_t = torch.tensor(query_emb[:dim], dtype=torch.float32).to(device)
    scores = (db_t @ q_t).cpu().tolist()
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


def cosine_topk(embeddings, query_emb: np.ndarray, top_k: int, device: str):
    db_t = torch.tensor(embeddings, dtype=torch.float32).to(device)
    q_t = torch.tensor(query_emb, dtype=torch.float32).to(device)
    sims = torch.nn.functional.cosine_similarity(db_t, q_t.unsqueeze(0), dim=1, eps=1e-12)
    top_scores, top_idxs = torch.topk(sims, min(top_k, len(embeddings)))
    return [(i.item(), s.item()) for i, s in zip(top_idxs, top_scores)]


def selector_matmul(embeddings, top_indices: list[int], k: int, dim: int, device: str) -> np.ndarray:
    num_docs = len(embeddings)
    db_t = torch.tensor(embeddings, dtype=torch.float32).to(device)

    db_padded = torch.zeros((num_docs, dim), dtype=torch.float32, device=device)
    db_padded[:, :db_t.shape[1]] = db_t

    sel = create_selector_matrix(top_indices, k, num_docs)
    sel_t = torch.tensor(sel, dtype=torch.float32).to(device)

    # (k, num_docs) @ (num_docs, dim) -> (k, dim)
    return torch.matmul(sel_t, db_padded).cpu().numpy()


def run(chroma_dir: Path, collection_name: str, query: str, dim: int, top_k: int, device: str,
        save_vectors: Path | None = None):
    print(f"ChromaDB : {chroma_dir}")
    print(f"Query    : {query!r}")
    print(f"dim={dim}  top_k={top_k}  device={device}\n")

    emb_fn = get_embedding_function(dim, device)
    collection = load_chroma_collection(chroma_dir, collection_name, emb_fn)

    vec_db = collection.get(include=["embeddings"])
    embeddings = vec_db["embeddings"]
    print(f"Loaded {len(embeddings)} document embeddings\n")

    query_emb = np.array(emb_fn([query])[0], dtype=np.float64)

    # --- Plaintext dot product ---
    print("=== Plaintext Dot Product (top-k) ===")
    dot_results = dot_product_topk(embeddings, query_emb, dim, top_k, device)
    top_indices = [idx for idx, _ in dot_results]
    top_scores = [score for _, score in dot_results]
    for rank, (idx, score) in enumerate(dot_results, 1):
        print(f"  [{rank:2d}] doc_idx={idx:3d}  score={score:.6f}")

    # --- Cosine similarity ---
    print("\n=== Cosine Similarity (top-k) ===")
    cos_results = cosine_topk(embeddings, query_emb, top_k, device)
    for rank, (idx, score) in enumerate(cos_results, 1):
        print(f"  [{rank:2d}] doc_idx={idx:3d}  score={score:.6f}")

    # --- Selector matrix retrieval ---
    print("\n=== Selector Matrix Retrieval ===")
    retrieved = selector_matmul(embeddings, top_indices, top_k, dim, device)
    print(f"  Retrieved shape : {retrieved.shape}  (k={top_k}, dim={dim})")
    print(f"  First doc [0:8] : {retrieved[0][:8]}")

    if save_vectors is not None:
        out = {
            "vectors": retrieved,
            "indices": np.array(top_indices, dtype=np.int64),
            "scores": np.array(top_scores, dtype=np.float64),
        }
        np.save(str(save_vectors), out, allow_pickle=True)
        print(f"\n[save-vectors] saved to {save_vectors}  shape={retrieved.shape}")


def main():
    parser = argparse.ArgumentParser(description="Plaintext RAG with ChromaDB + nomic-embed-text-v1.5")
    parser.add_argument("--chroma-dir", type=Path, required=True,
                        help="Path to ChromaDB persist directory")
    parser.add_argument("--collection", type=str, default=DEFAULT_COLLECTION,
                        help=f"ChromaDB collection name (default: {DEFAULT_COLLECTION})")
    parser.add_argument("--query", type=str, default="What is bezoar?",
                        help="Query text (default: 'What is bezoar?')")
    parser.add_argument("--dim", type=int, default=DEFAULT_DIM,
                        help=f"Embedding dimension (default: {DEFAULT_DIM})")
    parser.add_argument("--topk", type=int, default=DEFAULT_TOP_K,
                        help=f"Number of top results (default: {DEFAULT_TOP_K})")
    parser.add_argument("--save-vectors", type=Path, default=None, metavar="PATH",
                        help="Save plaintext top-k document vectors to this .npy file for comparison.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run(args.chroma_dir, args.collection, args.query, args.dim, args.topk, device,
        save_vectors=args.save_vectors)


if __name__ == "__main__":
    main()
