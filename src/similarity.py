"""
src/similarity.py
-----------------
Module 5: Near-Duplicate Detection & Similarity Analysis.
(Will also house Module 8: Similar Ticket Retrieval Index).

Detects lexical and semantic near-duplicate support tickets using TF-IDF
and cosine similarity to identify data contamination risks and quantify
test-set evaluation inflation.

Usage:
    python src/similarity.py
    python src/similarity.py --threshold 0.85
    python src/similarity.py --top-n 10
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Ensure repo root is on sys.path for direct script execution
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# 1. TF-IDF SIMILARITY MATRIX
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 2. NEAR-DUPLICATE DETECTION
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 3. DEMONSTRATION & DIAGNOSTIC REPORTING
# ---------------------------------------------------------------------------

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
    print(" " * 24 + "MODULE 5: NEAR-DUPLICATE AUDIT & DEMO")
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
    print("This is why Module 8 upgrades similarity retrieval to dense Sentence Transformer embeddings.")
    print("=" * 90 + "\n")

    return dupes_df


# ---------------------------------------------------------------------------
# 4. EXPLANATION OF DUPLICATE EVALUATION INFLATION
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 5. CLI ENTRYPOINT
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Module 5: Near-Duplicate Detection & Contamination Analysis")
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/tickets_clean.csv",
        help="Path to cleaned dataset CSV",
    )
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
        help="Number of representative pairs to display (default: 5)",
    )
    parser.add_argument(
        "--explain-only",
        action="store_true",
        help="Print only the educational explanation on duplicate evaluation inflation",
    )
    args = parser.parse_args()

    if args.explain_only:
        explain_duplicate_inflation()
        return

    csv_path = Path(args.input)
    if not csv_path.exists():
        print(f"[Error] Dataset not found at: {csv_path.resolve()}")
        print("Please run 'python src/preprocessing.py' first.")
        sys.exit(1)

    print(f"[similarity.py] Loading cleaned tickets from {csv_path}...")
    df = pd.read_csv(csv_path)

    # 1. Demonstrate near-duplicates and semantic vs lexical gap
    dupes_df = demonstrate_near_duplicates(df, threshold=args.threshold, top_n=args.top_n)

    # 2. Print educational explanation
    explain_duplicate_inflation()


if __name__ == "__main__":
    main()
