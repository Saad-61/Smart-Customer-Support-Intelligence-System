"""
src/features.py
---------------
Feature engineering and transformation pipelines for Riot Games support tickets.

Provides reusable scikit-learn transformers that transform raw ticket attributes
into numeric feature representations:
- Text: TF-IDF vectorization with unigrams, bigrams, and sublinear scaling.
- Categorical: One-Hot Encoding for game titles ('product').
- Numerical: Imputation and Standard Scaling for operational/temporal counts.

CRITICAL ARCHITECTURAL RULE:
Transformers are instantiated here but MUST NOT be fit upon module import.
Fitting happens strictly on the training partition inside train.py / evaluate.py
to prevent data leakage from the validation/test set.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------------------------------------------------------------------------
# 1. FEATURE DEFINITIONS
# ---------------------------------------------------------------------------

TEXT_FEATURES = ["ticket_text"]
CATEGORICAL_FEATURES = ["product"]
NUMERIC_FEATURES = ["previous_tickets", "hour_of_day", "day_of_week", "month"]

# Post-outcome fields that MUST be excluded from all model inputs
LEAKY_FEATURES = ["resolution_time", "resolved"]


# ---------------------------------------------------------------------------
# 2. SUB-PIPELINES
# ---------------------------------------------------------------------------

def build_text_pipeline(max_features: int = 15000) -> Pipeline:
    """
    Build TF-IDF text vectorization sub-pipeline.
    
    Configuration:
    - max_features: 15,000 top n-grams to balance vocabulary breadth and memory.
    - ngram_range: (1, 2) extracts single words and two-word combinations (e.g. 'charged twice').
    - sublinear_tf: True applies logarithmic sublinear scaling (1 + log(tf)) to dampen
      the influence of repeated words in long complaints.
    - strip_accents: 'unicode' normalizes international player characters.
    - min_df: 2 ignores words/bigrams that appear only once in the corpus.
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            sublinear_tf=True,
            strip_accents="unicode",
            analyzer="word",
            min_df=2,
        ))
    ])


def build_categorical_pipeline() -> Pipeline:
    """
    Build categorical encoding sub-pipeline for game titles.
    
    Configuration:
    - OneHotEncoder(handle_unknown='ignore', sparse_output=False):
      Ensures that if an unseen game/service appears in production API queries,
      it encodes as all zeros rather than crashing inference.
    """
    return Pipeline([
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])


def build_numeric_pipeline() -> Pipeline:
    """
    Build numeric preprocessing sub-pipeline.
    
    Configuration:
    - SimpleImputer(strategy='median'): Safe fallback for missing operational counts.
    - StandardScaler(): Centers data to zero mean and unit variance, essential for
      linear models (LogisticRegression, LinearSVC).
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])


# ---------------------------------------------------------------------------
# 3. COMPOSITE FEATURE TRANSFORMER
# ---------------------------------------------------------------------------

def build_full_pipeline(max_tfidf_features: int = 15000) -> ColumnTransformer:
    """
    Combine text, categorical, and numeric sub-pipelines into a single ColumnTransformer.
    
    Returns an unfitted ColumnTransformer ready to be integrated into an estimator
    pipeline in train.py.
    """
    return ColumnTransformer(
        transformers=[
            ("text", build_text_pipeline(max_features=max_tfidf_features), "ticket_text"),
            ("cat", build_categorical_pipeline(), CATEGORICAL_FEATURES),
            ("num", build_numeric_pipeline(), NUMERIC_FEATURES),
        ],
        remainder="drop",
    )


# ---------------------------------------------------------------------------
# 4. EXPLAINABILITY & INSPECTION HELPERS
# ---------------------------------------------------------------------------

def get_feature_names(fitted_pipeline: ColumnTransformer) -> list[str]:
    """
    Extract human-readable feature names from a fitted ColumnTransformer.
    
    Used by explainability modules to map model coefficients and tree split
    indices back to original words, products, or metadata features.
    """
    if not hasattr(fitted_pipeline, "get_feature_names_out"):
        raise ValueError("Pipeline has not been fitted yet. Call .fit() before extracting feature names.")
    
    raw_names = list(fitted_pipeline.get_feature_names_out())
    # Clean prefixes (e.g. 'text__tfidf__word' -> 'word')
    cleaned_names = []
    for name in raw_names:
        if name.startswith("text__tfidf__"):
            cleaned_names.append(name.replace("text__tfidf__", "word:"))
        elif name.startswith("text__"):
            cleaned_names.append(name.replace("text__", "word:"))
        elif name.startswith("cat__ohe__"):
            cleaned_names.append(name.replace("cat__ohe__", "cat:"))
        elif name.startswith("cat__"):
            cleaned_names.append(name.replace("cat__", "cat:"))
        elif name.startswith("num__scaler__"):
            cleaned_names.append(name.replace("num__scaler__", "num:"))
        elif name.startswith("num__"):
            cleaned_names.append(name.replace("num__", "num:"))
        else:
            cleaned_names.append(name)
            
    return cleaned_names


def print_leakage_report() -> None:
    """
    Print formal data leakage report documenting excluded post-outcome features.
    """
    report = """
========================================================================================
                               DATA LEAKAGE AUDIT REPORT
========================================================================================
| Feature          | Leakage Type       | Reason                                       |
|------------------|--------------------|----------------------------------------------|
| resolution_time  | Post-outcome       | Unknown at ticket creation time              |
| resolved         | Post-outcome       | Unknown at ticket creation time              |
========================================================================================
CRITICAL RULE:
These features are recorded only AFTER human agents resolve or close the ticket.
Using them as inputs to predict ticket category or priority would create catastrophic
target leakage (achieving near-100% synthetic accuracy that fails completely in production).
Both features are strictly excluded from all training and feature pipelines.
"""
    print(report)


# ---------------------------------------------------------------------------
# 5. VERIFICATION / CLI ENTRYPOINT
# ---------------------------------------------------------------------------

def main() -> None:
    """Demonstrate pipeline building and verify transformation integrity."""
    print("=" * 65)
    print(" " * 15 + "FEATURE ENGINEERING PIPELINE")
    print("=" * 65)

    # 1. Print Data Leakage Report
    print_leakage_report()

    # 2. Load dataset to demonstrate transformer assembly
    data_path = Path("data/processed/tickets_clean.csv")
    if not data_path.exists():
        print(f"[Warning] {data_path} not found. Run src/preprocessing.py first.")
        return

    df = pd.read_csv(data_path)
    print(f"\n[features.py] Loaded {len(df):,} cleaned tickets from {data_path}")

    # 3. Instantiate pipeline (unfitted)
    pipeline = build_full_pipeline(max_tfidf_features=15000)
    print("[features.py] Instantiated composite ColumnTransformer successfully:")
    print("  - Text Sub-pipeline:        TfidfVectorizer(max_features=15000, ngrams=(1,2))")
    print("  - Categorical Sub-pipeline: OneHotEncoder(product)")
    print("  - Numeric Sub-pipeline:     SimpleImputer(median) -> StandardScaler()")

    # 4. Demonstrate fit_transform on a training slice (proving transformation without modifying global state)
    print("\n[features.py] Testing transformation on training sample (500 rows)...")
    sample_train = df.iloc[:500]
    X_sample = pipeline.fit_transform(sample_train)
    feature_names = get_feature_names(pipeline)

    print(f"[features.py] Transformed Feature Matrix Shape: {X_sample.shape}")
    print(f"[features.py] Total Extracted Features:       {len(feature_names):,}")
    print(f"[features.py] Sample Word Features (first 5):  {feature_names[:5]}")
    print(f"[features.py] Sample Categorical Features:     {[f for f in feature_names if f.startswith('cat:')][:5]}")
    print(f"[features.py] Sample Numeric Features:         {[f for f in feature_names if f.startswith('num:')]}")
    print("\n[features.py] Verification complete. Pipeline is ready for train.py.")


if __name__ == "__main__":
    main()
