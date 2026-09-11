"""
src/similarity.py
-----------------
Near-Duplicate Detection (Module 5) & Similar Ticket Retrieval Index (Module 8).

Features:
1. Near-duplicate detection using TF-IDF and cosine similarity to identify
   data contamination risks and quantify test-set evaluation inflation.
2. Semantic ticket retrieval using dense Sentence Transformers ("all-MiniLM-L6-v2")
   running on CUDA GPU (with CPU fallback) to retrieve top-k historically similar
   tickets for incoming support requests.

Usage:
    # Module 5: Near-duplicate detection & data contamination audit
    python src/similarity.py --threshold 0.85
    python src/similarity.py --explain-only

    # Module 8: Build dense retrieval index on GPU
    python src/similarity.py --build-index

    # Module 8: Run semantic retrieval demonstration
    python src/similarity.py --demo-retrieval

    # Module 8: Search for similar tickets given an ad-hoc query
    python src/similarity.py --query "I was charged twice for the same order."
    python src/similarity.py --document-limitations
"""

import argparse
from datetime import datetime
from pathlib import Path
import sys
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Ensure repo root is on sys.path for direct script execution
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Cached SentenceTransformer model instances: {cache_key: model}
_MODEL_CACHE: dict[str, Any] = {}


# ===========================================================================
# DEVICE SELECTION & MODEL CACHING
# ===========================================================================

def get_device(requested_device: str | None = None) -> str:
    """
    Resolve target device ('cuda' or 'cpu').
    Defaults to CUDA if an NVIDIA GPU is detected via PyTorch.
    """
    if requested_device is not None:
        return requested_device.lower()
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def get_sentence_transformer(
    model_name: str = "all-MiniLM-L6-v2",
    device: str | None = None,
) -> Any:
    """
    Load or retrieve cached SentenceTransformer instance on the target device.
    """
    device_str = get_device(device)
    cache_key = f"{model_name}_{device_str}"

    if cache_key not in _MODEL_CACHE:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for semantic retrieval. "
                "Please run: pip install sentence-transformers"
            ) from e

        device_info = device_str.upper()
        if device_str == "cuda":
            try:
                import torch
                device_info = f"CUDA ({torch.cuda.get_device_name(0)})"
            except Exception:
                device_info = "CUDA"

        print(f"[similarity.py] Loading SentenceTransformer('{model_name}') on {device_info}...")
        model = SentenceTransformer(model_name, device=device_str)
        _MODEL_CACHE[cache_key] = model

    return _MODEL_CACHE[cache_key]


# ===========================================================================
# 1. TF-IDF SIMILARITY MATRIX (MODULE 5)
# ===========================================================================

def compute_tfidf_similarity_matrix(
    texts: Sequence[str],
    max_features: int = 5000,
) -> tuple[np.ndarray, TfidfVectorizer]:
    """
    Fit TF-IDF on ticket texts and compute pairwise cosine similarity matrix.

    Configuration:
        - max_features: 5,000 unigrams and bigrams.
        - sublinear_tf: True applies logarithmic term frequency scaling.
        - strip_accents: 'unicode' for character normalization.

    Parameters:
        texts: Sequence of ticket complaint strings.
        max_features: Vocabulary capacity limit.

    Returns:
        (similarity_matrix, fitted_vectorizer)
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        sublinear_tf=True,
        strip_accents="unicode",
        min_df=2,
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    sim_matrix = cosine_similarity(tfidf_matrix)
    return sim_matrix, vectorizer


# ===========================================================================
# 2. NEAR-DUPLICATE DETECTION (MODULE 5)
# ===========================================================================

def find_near_duplicates(
    df: pd.DataFrame,
    threshold: float = 0.90,
    max_features: int = 5000,
) -> pd.DataFrame:
    """
    Find ticket pairs where cosine similarity >= threshold.

    Extracts the strict upper triangle (i < j) to avoid comparing tickets to
    themselves or generating redundant reversed pairs.

    Parameters:
        df: Cleaned tickets DataFrame.
        threshold: Minimum cosine similarity required to flag as near-duplicate.
        max_features: Vocabulary size for TF-IDF.

    Returns:
        DataFrame with columns:
            [ticket_id_1, ticket_id_2, customer_id_1, customer_id_2,
             text_1, text_2, similarity_score, same_customer]
    """
    if "ticket_text" not in df.columns:
        raise ValueError("DataFrame must contain 'ticket_text' column.")

    texts = df["ticket_text"].fillna("").astype(str).tolist()
    sim_matrix, _ = compute_tfidf_similarity_matrix(texts, max_features=max_features)

    # Extract upper triangle indices (i < j)
    row_idx, col_idx = np.triu_indices(len(df), k=1)
    pair_sims = sim_matrix[row_idx, col_idx]

    # Filter pairs exceeding threshold
    mask = pair_sims >= threshold
    matched_rows = row_idx[mask]
    matched_cols = col_idx[mask]
    matched_sims = pair_sims[mask]

    # Sort descending by similarity score
    sort_order = np.argsort(-matched_sims)
    matched_rows = matched_rows[sort_order]
    matched_cols = matched_cols[sort_order]
    matched_sims = matched_sims[sort_order]

    cust1 = df["customer_id"].iloc[matched_rows].values
    cust2 = df["customer_id"].iloc[matched_cols].values

    near_dupes_df = pd.DataFrame({
        "ticket_id_1": df["ticket_id"].iloc[matched_rows].values,
        "ticket_id_2": df["ticket_id"].iloc[matched_cols].values,
        "customer_id_1": cust1,
        "customer_id_2": cust2,
        "text_1": df["ticket_text"].iloc[matched_rows].values,
        "text_2": df["ticket_text"].iloc[matched_cols].values,
        "similarity_score": np.round(matched_sims, 4),
        "same_customer": (cust1 == cust2),
    })

    return near_dupes_df


# ===========================================================================
# 3. DEMONSTRATION & DIAGNOSTIC REPORTING (MODULE 5)
# ===========================================================================

def demonstrate_near_duplicates(
    df: pd.DataFrame,
    threshold: float = 0.85,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Display representative near-duplicate pairs and compare lexical vs. semantic similarity.

    Includes:
    - Top near-identical pairs discovered via TF-IDF cosine similarity.
    - An injected rephrased pair (e.g. 'charged twice' vs 'billed twice') to contrast
      lexical TF-IDF matching against semantic intent.
    """
    print("\n" + "=" * 90)
    print(" " * 24 + "NEAR-DUPLICATE AUDIT & DEMO")
    print("=" * 90)
    print(f"Total Clean Tickets Scanned: {len(df):,}")
    print(f"Cosine Similarity Threshold: >= {threshold:.2f}\n")

    dupes_df = find_near_duplicates(df, threshold=threshold)
    total_found = len(dupes_df)
    same_cust_count = int(dupes_df["same_customer"].sum()) if total_found > 0 else 0
    cross_cust_count = total_found - same_cust_count

    print(f"Found {total_found:,} near-duplicate pairs meeting threshold >= {threshold:.2f}:")
    print(f"  - Same Customer:  {same_cust_count:>5,} pairs ({same_cust_count/max(1, total_found)*100:5.1f}%)")
    print(f"  - Cross Customer: {cross_cust_count:>5,} pairs ({cross_cust_count/max(1, total_found)*100:5.1f}%)")
    print("-" * 90)

    # Display top representative pairs
    display_n = min(top_n, total_found)
    print(f"Displaying Top {display_n} Representative High-Similarity Pairs:\n")

    for idx in range(display_n):
        row = dupes_df.iloc[idx]
        cust_rel = "SAME CUSTOMER" if row["same_customer"] else "DIFFERENT CUSTOMERS"
        print(f"Pair #{idx + 1} | Similarity: {row['similarity_score']:.4f} | [{cust_rel}]")
        print(f"  Ticket A ({row['ticket_id_1']} / {row['customer_id_1']}): \"{row['text_1']}\"")
        print(f"  Ticket B ({row['ticket_id_2']} / {row['customer_id_2']}): \"{row['text_2']}\"")
        print("  " + "-" * 86)

    # Demonstrate the Lexical vs. Semantic gap on rephrased pairs
    print("\n" + "=" * 90)
    print(" " * 18 + "LEXICAL (TF-IDF) vs. SEMANTIC SIMILARITY COMPARISON")
    print("=" * 90)
    rephrased_sample_1 = "i was charged twice for the same order."
    rephrased_sample_2 = "my credit card was billed twice for a single purchase."

    # Compute cosine similarity between these two specific sentences
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    pair_tfidf = vec.fit_transform([rephrased_sample_1, rephrased_sample_2])
    pair_sim = float(cosine_similarity(pair_tfidf[0:1], pair_tfidf[1:2])[0][0])
    shared_terms = set(rephrased_sample_1.split()).intersection(set(rephrased_sample_2.split()))

    print("Example Rephrased Near-Duplicate Pair (Injected Semantics):")
    print(f"  Text 1: \"{rephrased_sample_1}\"")
    print(f"  Text 2: \"{rephrased_sample_2}\"")
    print(f"  Shared Words:              {sorted(list(shared_terms))}")
    print(f"  TF-IDF Cosine Similarity:   {pair_sim:.4f}  <-- Missed by strict lexical matching (>0.85)")
    print("\nKey Takeaway:")
    print("TF-IDF excels at detecting identical templates with varying IDs, but fails on paraphrased")
    print("tickets where different words convey identical meaning ('charged' vs 'billed', 'order' vs 'purchase').")
    print("This is why semantic ticket retrieval upgrades to dense Sentence Transformer embeddings.")
    print("=" * 90 + "\n")

    return dupes_df


# ===========================================================================
# 4. EXPLANATION OF DUPLICATE EVALUATION INFLATION (MODULE 5)
# ===========================================================================

def explain_duplicate_inflation() -> str:
    """
    Return and print comprehensive explanation of near-duplicate data contamination.
    """
    explanation = """
========================================================================================
             DATA CONTAMINATION: WHY NEAR-DUPLICATES INFLATE EVALUATION SCORES
========================================================================================

1. How Near-Duplicates Inflate Metrics:
   - When near-identical complaints exist in both the training set and the test set,
     the evaluation benchmark is no longer testing generalization to unseen situations.
   - Instead, the model is tested on near-exact copies of text it has already memorized.
   - This inflates accuracy, precision, and recall, creating a false sense of security.

2. Why Random Row-Level Splitting Is Particularly Dangerous:
   - In standard random splitting (train_test_split), tickets from the same player or
     tickets generated from identical templates are randomly distributed between train and test.
   - For example, if a player submitted 5 near-identical tickets about a login failure,
     3 might land in train and 2 in test.
   - The test set becomes contaminated with 'leaked' questions from the training set.

3. What Customer-Aware Splitting Solves (and What It Leaves Unsolved):
   - What it SOLVES: Customer-aware splitting places ALL tickets from customer X into
     either train or test. It completely prevents SAME-CUSTOMER duplicate leakage.
   - What it LEAVES UNSOLVED: When two DIFFERENT customers submit identical or
     template-based complaints (e.g. 'I was banned for toxic behaviour', or widespread
     server crash reports), customer-aware splitting cannot prevent them from crossing
     splits. Cross-customer near-duplicates remain a subtle form of data contamination.
========================================================================================
"""
    print(explanation)
    return explanation


# ===========================================================================
# 5. DENSE RETRIEVAL INDEX BUILDER (MODULE 8)
# ===========================================================================

def build_retrieval_index(
    df: pd.DataFrame,
    model_name: str = "all-MiniLM-L6-v2",
    device: str | None = None,
    batch_size: int = 64,
) -> dict[str, Any]:
    """
    Encode all historical ticket texts into dense 384-dimensional semantic vectors
    using Sentence Transformers on GPU (with CPU fallback).

    Parameters:
        df: Cleaned tickets DataFrame.
        model_name: Hugging Face model identifier (default: "all-MiniLM-L6-v2").
        device: 'cuda' or 'cpu' (default: auto-detected).
        batch_size: Batch size for dense embedding inference.

    Returns:
        Dense index dictionary containing normalized embeddings and metadata.
    """
    if "ticket_text" not in df.columns:
        raise ValueError("DataFrame must contain 'ticket_text' column.")

    device_str = get_device(device)
    model = get_sentence_transformer(model_name, device=device_str)

    texts = df["ticket_text"].fillna("").astype(str).tolist()
    ticket_ids = df["ticket_id"].astype(str).tolist()
    products = df["product"].astype(str).tolist() if "product" in df.columns else ["Unknown"] * len(df)
    categories = df["category"].astype(str).tolist() if "category" in df.columns else ["Unknown"] * len(df)
    priorities = df["priority"].astype(str).tolist() if "priority" in df.columns else ["Unknown"] * len(df)

    print(f"\n[similarity.py] Encoding {len(texts):,} historical tickets using {model_name}...")
    print(f"[similarity.py] Target device: {device_str.upper()} | Batch size: {batch_size}")

    # Encode texts with L2 normalization enabled -> Cosine similarity becomes a simple dot product
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    index: dict[str, Any] = {
        "embeddings": embeddings,
        "ticket_ids": ticket_ids,
        "texts": texts,
        "products": products,
        "categories": categories,
        "priorities": priorities,
        "model_name": model_name,
        "embedding_dim": int(embeddings.shape[1]),
        "total_records": len(ticket_ids),
        "created_at": datetime.now().isoformat(),
    }

    print(f"[similarity.py] Successfully generated embedding matrix: shape {embeddings.shape}, dtype {embeddings.dtype}")
    return index


# ===========================================================================
# 6. SERIALIZATION & PERSISTENCE (MODULE 8)
# ===========================================================================

def save_retrieval_index(
    index: dict[str, Any],
    path: str | Path = "models/retrieval_index.joblib",
) -> Path:
    """
    Serialize the dense retrieval index to disk for downstream inference / API serving.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(index, out_path)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"[similarity.py] Saved retrieval index to {out_path} ({size_mb:.2f} MB, {index['total_records']:,} records).")
    return out_path


def load_retrieval_index(
    path: str | Path = "models/retrieval_index.joblib",
) -> dict[str, Any]:
    """
    Load pre-built retrieval index from disk.
    """
    target_path = Path(path)
    if not target_path.exists():
        raise FileNotFoundError(
            f"Retrieval index not found at '{target_path}'. "
            "Please build the index first using 'python src/similarity.py --build-index'."
        )
    index = joblib.load(target_path)
    return index


# ===========================================================================
# 7. SEMANTIC QUERY RETRIEVAL (MODULE 8)
# ===========================================================================

def find_similar_tickets(
    query_text: str,
    index: dict[str, Any],
    top_k: int = 5,
    model: Any = None,
    device: str | None = None,
) -> list[dict[str, Any]]:
    """
    Retrieve top_k most semantically similar tickets for an incoming query.

    Parameters:
        query_text: Raw incoming player complaint text.
        index: Dense retrieval index dictionary.
        top_k: Number of nearest tickets to return.
        model: Pre-loaded SentenceTransformer instance (optional).
        device: 'cuda' or 'cpu'.

    Returns:
        List of dicts:
        [
            {
                "rank": int,
                "ticket_id": str,
                "similarity": float,
                "category": str,
                "priority": str,
                "product": str,
                "preview": str,
                "text": str,
            },
            ...
        ]
    """
    if not query_text or not query_text.strip():
        return []

    if model is None:
        model_name = index.get("model_name", "all-MiniLM-L6-v2")
        model = get_sentence_transformer(model_name, device=device)

    # Encode query into normalized vector
    query_vec = model.encode(
        [query_text.strip()],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0].astype(np.float32)

    embeddings = index["embeddings"]  # Shape: (N, 384)
    # Cosine similarity is inner dot product of unit-length vectors
    similarities = np.dot(embeddings, query_vec)

    # Extract top k indices sorted descending
    k = min(top_k, len(similarities))
    top_indices = np.argpartition(-similarities, k)[:k]
    top_indices = top_indices[np.argsort(-similarities[top_indices])]

    results: list[dict[str, Any]] = []
    for rank, idx in enumerate(top_indices, start=1):
        sim_score = float(similarities[idx])
        ticket_id = index["ticket_ids"][idx]
        text = index["texts"][idx]
        cat = index["categories"][idx] if idx < len(index.get("categories", [])) else "Unknown"
        prio = index["priorities"][idx] if idx < len(index.get("priorities", [])) else "Unknown"
        prod = index["products"][idx] if idx < len(index.get("products", [])) else "Unknown"
        preview = text[:80] + ("..." if len(text) > 80 else "")

        results.append({
            "rank": rank,
            "ticket_id": ticket_id,
            "similarity": round(sim_score, 4),
            "category": cat,
            "priority": prio,
            "product": prod,
            "preview": preview,
            "text": text,
        })

    return results


# ===========================================================================
# 8. DEMONSTRATION & BENCHMARKING (MODULE 8)
# ===========================================================================

def demonstrate_retrieval(
    index: dict[str, Any] | None = None,
    top_k: int = 5,
    index_path: str | Path = "models/retrieval_index.joblib",
    device: str | None = None,
) -> None:
    """
    Run 3 diverse domain queries to demonstrate dense semantic ticket retrieval.
    """
    if index is None:
        index = load_retrieval_index(index_path)

    sample_queries = [
        "I was charged twice for the same order.",
        "Game freezes and crashes during loading screen every time.",
        "My account was permanently suspended for scripts but I never cheated.",
    ]

    print("\n" + "=" * 95)
    print(" " * 22 + "DEMONSTRATION: SEMANTIC TICKET RETRIEVAL")
    print("=" * 95)
    print(f"Total Indexed Historical Tickets: {index['total_records']:,}")
    print(f"Embedding Architecture:           {index.get('model_name', 'all-MiniLM-L6-v2')} ({index.get('embedding_dim', 384)} dims, dense cosine)")
    print("=" * 95 + "\n")

    model = get_sentence_transformer(index.get("model_name", "all-MiniLM-L6-v2"), device=device)

    for q_idx, query in enumerate(sample_queries, start=1):
        print(f"Query #{q_idx}: \"{query}\"")
        print("-" * 95)
        matches = find_similar_tickets(query, index, top_k=top_k, model=model, device=device)
        for m in matches:
            print(f"  {m['rank']}. [{m['similarity']:.4f}] {m['ticket_id']} | {m['product']:<18} | {m['category']} ({m['priority']})")
            print(f"     \"{m['preview']}\"")
        print("-" * 95 + "\n")


# ===========================================================================
# 9. LIMITATIONS DOCUMENTATION (MODULE 8)
# ===========================================================================

def document_limitations() -> str:
    """
    Print and return technical analysis of retrieval limitations:
    TF-IDF vs. Dense Transformer Embeddings.
    """
    limitations = """
========================================================================================
         LIMITATIONS OF RETRIEVAL METHODS: TF-IDF vs. DENSE SENTENCE TRANSFORMERS
========================================================================================

1. TF-IDF Lexical Matching Limitations:
   - Exact Term Dependency: TF-IDF relies entirely on shared vocabulary (unigrams/bigrams).
     Paraphrased or synonymous complaints (e.g. 'charged twice' vs. 'billed twice') share
     virtually zero distinctive keywords, causing severe similarity degradation (0.15 vs 0.85+).
   - Inability to Discern Context & Negation: TF-IDF treats words independently (bag-of-words),
     failing to distinguish negated phrases ('cannot log in' vs. 'can log in').

2. Dense Sentence Transformer (all-MiniLM-L6-v2) Trade-offs:
   - Domain & Gaming Slang Gaps: General web-trained models may lack specialized gaming
     telemetry or Riot slang ('inting', 'smurf', 'hardstuck', 'MMR tanked') unless fine-tuned.
   - Inference Latency: While TF-IDF sparse matrix multiplication takes microseconds, dense
     transformer inference requires deep neural forward passes (~15-40ms on CPU, ~2-5ms on GPU).
   - Information Bottleneck: Compressing complex, multi-sentence paragraphs into a single
     384-dimensional vector inherently loses fine-grained technical diagnostics (e.g., specific
     DirectX error codes or kernel crash stack traces).

3. Shared Practical Limitations:
   - Extreme Query Brevity: Ultra-short inputs ('ban', 'help', 'lag') lack contextual entropy,
     leading to diffuse or ambiguous retrieval neighborhoods.
   - Static Index Rebuild Requirement: The vector index is a point-in-time snapshot. When new
     tickets, new game patches, or new bug classes emerge, the embedding index must be rebuilt
     or augmented via dynamic vector databases (e.g., FAISS, ChromaDB).
========================================================================================
"""
    print(limitations)
    return limitations


# ===========================================================================
# 10. CLI ENTRYPOINT
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Near-Duplicate Detection (Module 5) & Similar Ticket Retrieval Index (Module 8)"
    )
    # Common arguments
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/tickets_clean.csv",
        help="Path to cleaned dataset CSV (default: data/processed/tickets_clean.csv)",
    )
    # Module 5 arguments
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Cosine similarity threshold for near-duplicate detection (default: 0.85)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of representative pairs to display for near-duplicates (default: 5)",
    )
    parser.add_argument(
        "--explain-only",
        action="store_true",
        help="Print only the educational explanation on duplicate evaluation inflation",
    )
    # Module 8 arguments
    parser.add_argument(
        "--build-index",
        action="store_true",
        help="Build and serialize dense retrieval index on GPU using Sentence Transformers",
    )
    parser.add_argument(
        "--demo-retrieval",
        action="store_true",
        help="Run semantic retrieval demonstration using sample queries",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Execute semantic search for an ad-hoc query string",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of nearest tickets to retrieve (default: 5)",
    )
    parser.add_argument(
        "--index-path",
        type=str,
        default="models/retrieval_index.joblib",
        help="Path to save or load retrieval index (default: models/retrieval_index.joblib)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model name (default: all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Target device ('cuda' or 'cpu', default: auto-detect)",
    )
    parser.add_argument(
        "--document-limitations",
        action="store_true",
        help="Print written comparison of TF-IDF vs. Dense Transformer limitations",
    )

    args = parser.parse_args()

    # 1. Print limitations if requested
    if args.document_limitations:
        document_limitations()
        return

    # 2. Print Module 5 explanation if requested
    if args.explain_only:
        explain_duplicate_inflation()
        return

    # 3. Handle Ad-hoc Query Search
    if args.query:
        print(f"[similarity.py] Loading index from {args.index_path}...")
        index = load_retrieval_index(args.index_path)
        print(f"\n[similarity.py] Searching top {args.top_k} matches for query: \"{args.query}\"")
        print("-" * 90)
        results = find_similar_tickets(args.query, index, top_k=args.top_k, device=args.device)
        for r in results:
            print(f"  {r['rank']}. [{r['similarity']:.4f}] {r['ticket_id']} | {r['product']} | {r['category']} ({r['priority']})")
            print(f"     \"{r['preview']}\"")
        print("-" * 90)
        return

    # 4. Handle Demonstration
    if args.demo_retrieval:
        demonstrate_retrieval(top_k=args.top_k, index_path=args.index_path, device=args.device)
        document_limitations()
        return

    # 5. Handle Build Index
    if args.build_index:
        csv_path = Path(args.input)
        if not csv_path.exists():
            print(f"[Error] Dataset not found at: {csv_path.resolve()}")
            sys.exit(1)

        print(f"[similarity.py] Loading cleaned tickets from {csv_path}...")
        df = pd.read_csv(csv_path)

        index = build_retrieval_index(
            df=df,
            model_name=args.model_name,
            device=args.device,
        )
        save_retrieval_index(index, path=args.index_path)

        # Run demonstration immediately after building index
        demonstrate_retrieval(index=index, top_k=args.top_k, device=args.device)
        document_limitations()
        return

    # Default fallback: Module 5 Near-duplicate detection
    csv_path = Path(args.input)
    if not csv_path.exists():
        print(f"[Error] Dataset not found at: {csv_path.resolve()}")
        sys.exit(1)

    print(f"[similarity.py] Loading cleaned tickets from {csv_path}...")
    df = pd.read_csv(csv_path)
    demonstrate_near_duplicates(df, threshold=args.threshold, top_n=args.top_n)
    explain_duplicate_inflation()


if __name__ == "__main__":
    main()
