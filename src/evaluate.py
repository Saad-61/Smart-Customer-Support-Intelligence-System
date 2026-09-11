"""
src/evaluate.py
---------------
Module 4: Leakage Analysis & Evaluation Strategy.

Implements rigorous evaluation methodologies to prevent data leakage and benchmark
model generalization:
1. Leakage Experiment: Demonstrates artificial score inflation when post-outcome
   features ('resolution_time', 'resolved') are improperly included in training.
2. Split Strategies:
   - random_split: Standard row-level random train/test split.
   - customer_aware_split: Groups tickets by unique customer_id to guarantee that
     no customer in the test set ever appeared during training.
3. Split Comparison: Directly measures the performance drop on unseen customers
   (the true, honest generalization benchmark).
4. Diagnostic Reporting: Outputs multi-class classification reports, macro F1,
   and formatted text confusion matrices.

Usage:
    python src/evaluate.py
    python src/evaluate.py --leakage-only
    python src/evaluate.py --compare-splits
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Ensure repo root is on sys.path for direct script execution
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.features import (
    CATEGORICAL_FEATURES,
    LEAKY_FEATURES,
    NUMERIC_FEATURES,
    TEXT_FEATURES,
    build_categorical_pipeline,
    build_full_pipeline,
    build_text_pipeline,
)

# Valid model features (strictly excluding LEAKY_FEATURES)
FEATURE_COLS = TEXT_FEATURES + CATEGORICAL_FEATURES + NUMERIC_FEATURES


# ---------------------------------------------------------------------------
# 1. SPLIT STRATEGIES
# ---------------------------------------------------------------------------

def random_split(
    df: pd.DataFrame,
    target_col: str = "category",
    test_size: float = 0.2,
    seed: int = 42,
    feature_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Standard row-level train/test split.
    
    Parameters:
        df: Cleaned tickets DataFrame.
        target_col: Target column to predict ('category' or 'priority').
        test_size: Proportion of rows allocated to the test split.
        seed: Random state for reproducibility.
        feature_cols: Features to include in X (defaults to FEATURE_COLS).
        
    Returns:
        (X_train, X_test, y_train, y_test)
    """
    cols = feature_cols if feature_cols is not None else FEATURE_COLS
    X = df[cols]
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=seed)


def customer_aware_split(
    df: pd.DataFrame,
    target_col: str = "category",
    test_size: float = 0.2,
    seed: int = 42,
    feature_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Partition tickets such that NO customer in the test set appears in training.
    
    Algorithm:
        1. Extract all unique customer_ids.
        2. Shuffle them using a reproducible seed.
        3. Allocate test_size (e.g. 20%) of unique customers to the test cohort.
        4. Assign all tickets belonging to test customers to the test set;
           all remaining tickets form the training set.
           
    Parameters:
        df: Cleaned tickets DataFrame containing 'customer_id'.
        target_col: Target column to predict ('category' or 'priority').
        test_size: Proportion of unique customers assigned to test.
        seed: Random state for customer shuffling.
        feature_cols: Features to include in X (defaults to FEATURE_COLS).
        
    Returns:
        (X_train, X_test, y_train, y_test)
    """
    if "customer_id" not in df.columns:
        raise ValueError("DataFrame must contain 'customer_id' column for customer-aware splitting.")

    unique_customers = np.array(sorted(df["customer_id"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_customers)

    n_test_customers = int(len(unique_customers) * test_size)
    test_customers = set(unique_customers[:n_test_customers])

    train_mask = ~df["customer_id"].isin(test_customers)
    test_mask = df["customer_id"].isin(test_customers)

    train_df = df[train_mask]
    test_df = df[test_mask]

    cols = feature_cols if feature_cols is not None else FEATURE_COLS
    X_train = train_df[cols]
    X_test = test_df[cols]
    y_train = train_df[target_col]
    y_test = test_df[target_col]

    # Integrity verification
    overlap = set(train_df["customer_id"]).intersection(set(test_df["customer_id"]))
    assert len(overlap) == 0, f"Customer leakage detected! {len(overlap)} customers overlap between splits."

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# 2. DATA LEAKAGE EXPERIMENT
# ---------------------------------------------------------------------------

def leakage_experiment(
    df: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Train Logistic Regression models with and without leaky post-outcome features.
    
    Model A (Leaky):
        Includes 'ticket_text', 'product', 'previous_tickets', temporal features,
        AND leaky post-outcome features ('resolution_time', 'resolved').
    Model B (Clean):
        Strictly excludes 'resolution_time' and 'resolved'.
        
    Demonstrates how post-outcome features artificially inflate accuracy and macro F1,
    leading to fatal model failure in real production environments where post-outcome
    variables do not yet exist at ticket submission time.
    """
    print("\n" + "=" * 80)
    print(" " * 22 + "MODULE 4: DATA LEAKAGE EXPERIMENT")
    print("=" * 80)
    print("Target Target: 'priority' (HIGH / MEDIUM / LOW)")
    print("Hypothesis: Post-outcome features ('resolution_time', 'resolved') cause severe")
    print("            target leakage, producing deceptive, artificially inflated performance.\n")

    target_col = "priority"
    
    # 1. Clean feature pipeline (Model B)
    clean_cols = FEATURE_COLS
    clean_preprocessor = build_full_pipeline(max_tfidf_features=5000)
    model_clean = Pipeline([
        ("features", clean_preprocessor),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)),
    ])

    # 2. Leaky feature pipeline (Model A)
    leaky_cols = FEATURE_COLS + LEAKY_FEATURES
    leaky_preprocessor = ColumnTransformer(
        transformers=[
            ("text", build_text_pipeline(max_features=5000), "ticket_text"),
            ("cat", build_categorical_pipeline(), CATEGORICAL_FEATURES),
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), NUMERIC_FEATURES + LEAKY_FEATURES),
        ],
        remainder="drop",
    )
    model_leaky = Pipeline([
        ("features", leaky_preprocessor),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)),
    ])

    # Split dataset (using same partition for fair side-by-side comparison)
    X_train_clean, X_test_clean, y_train, y_test = random_split(
        df, target_col=target_col, test_size=test_size, seed=seed, feature_cols=clean_cols
    )
    X_train_leaky, X_test_leaky, _, _ = random_split(
        df, target_col=target_col, test_size=test_size, seed=seed, feature_cols=leaky_cols
    )

    print(f"Training partition: {len(X_train_clean):,} tickets")
    print(f"Testing partition:  {len(X_test_clean):,} tickets")
    print("-" * 80)

    # Train Model A (Leaky)
    print("[1/2] Training Model A (LEAKY: includes 'resolution_time' and 'resolved')...")
    model_leaky.fit(X_train_leaky, y_train)
    y_pred_leaky = model_leaky.predict(X_test_leaky)
    acc_leaky = accuracy_score(y_test, y_pred_leaky)
    f1_leaky = f1_score(y_test, y_pred_leaky, average="macro")

    # Train Model B (Clean)
    print("[2/2] Training Model B (CLEAN: valid pre-resolution features only)...")
    model_clean.fit(X_train_clean, y_train)
    y_pred_clean = model_clean.predict(X_test_clean)
    acc_clean = accuracy_score(y_test, y_pred_clean)
    f1_clean = f1_score(y_test, y_pred_clean, average="macro")

    # Display comparison table
    print("\n" + "=" * 80)
    print(" " * 26 + "LEAKAGE COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'Configuration':<22} {'Features Included':<36} {'Accuracy':<10} {'Macro F1':<10}")
    print("-" * 80)
    print(f"{'Model A (LEAKY)':<22} {'Text + Prod + Meta + res_time + res':<36} {acc_leaky*100:>7.2f}% {f1_leaky*100:>8.2f}%")
    print(f"{'Model B (CLEAN)':<22} {'Text + Prod + Meta (Clean)':<36} {acc_clean*100:>7.2f}% {f1_clean*100:>8.2f}%")
    print("-" * 80)

    acc_diff = (acc_leaky - acc_clean) * 100
    f1_diff = (f1_leaky - f1_clean) * 100
    print(f"Artificial Metric Inflation: +{acc_diff:5.2f}% Accuracy | +{f1_diff:5.2f}% Macro F1")
    print("=" * 80)

    print("\n[Educational Breakdown: Why Leaky Features Are Catastrophic]")
    print("1. Mechanism: In data generation, 'resolution_time' was generated conditionally on priority")
    print("   (HIGH base=4h, MEDIUM base=24h, LOW base=72h). Model A trivially exploits this correlation.")
    print("2. Operational Reality: When a real customer submits a support ticket, the ticket has NOT yet")
    print("   been resolved. 'resolution_time' and 'resolved' are completely unknown at prediction time.")
    print("3. Consequence: A model trained with leaky features will show outstanding test scores in development")
    print("   (~80-90%+), but will catastrophically collapse to random guessing when deployed to production.\n")

    return {
        "model_leaky": {"accuracy": acc_leaky, "macro_f1": f1_leaky},
        "model_clean": {"accuracy": acc_clean, "macro_f1": f1_clean},
        "inflation": {"accuracy_diff": acc_diff, "macro_f1_diff": f1_diff},
    }


# ---------------------------------------------------------------------------
# 3. SPLIT COMPARISON: RANDOM vs. CUSTOMER-AWARE
# ---------------------------------------------------------------------------

def compare_splits(
    df: pd.DataFrame,
    model: Any = None,
    target_col: str = "category",
    seed: int = 42,
) -> dict[str, Any]:
    """
    Train and evaluate a classifier on both Random Split and Customer-Aware Split.
    
    Shows that random splitting produces inflated performance because repeat customers
    appear in both training and test sets. Customer-aware splitting evaluates the model
    on completely unseen customers, yielding honest, production-realistic metrics.
    """
    print("\n" + "=" * 80)
    print(" " * 18 + "MODULE 4: RANDOM vs. CUSTOMER-AWARE SPLIT")
    print("=" * 80)
    print(f"Target Column: '{target_col}'")
    print("Estimator:     LogisticRegression (class_weight='balanced')\n")

    if model is None:
        model = Pipeline([
            ("features", build_full_pipeline(max_tfidf_features=5000)),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)),
        ])

    # 1. Random Split
    print("[1/2] Evaluating Random Row Split (Naive Baseline)...")
    X_train_rand, X_test_rand, y_train_rand, y_test_rand = random_split(
        df, target_col=target_col, test_size=0.2, seed=seed
    )
    model_rand = clone(model)
    model_rand.fit(X_train_rand, y_train_rand)
    y_pred_rand = model_rand.predict(X_test_rand)
    acc_rand = accuracy_score(y_test_rand, y_pred_rand)
    macro_f1_rand = f1_score(y_test_rand, y_pred_rand, average="macro")

    # Count customer overlap in random split
    rand_train_cust = set(df.loc[X_train_rand.index, "customer_id"])
    rand_test_cust = set(df.loc[X_test_rand.index, "customer_id"])
    rand_cust_overlap = len(rand_train_cust.intersection(rand_test_cust))
    rand_overlap_pct = (rand_cust_overlap / len(rand_test_cust)) * 100

    # 2. Customer-Aware Split
    print("[2/2] Evaluating Customer-Aware Split (Leak-Free Generalization)...")
    X_train_cust, X_test_cust, y_train_cust, y_test_cust = customer_aware_split(
        df, target_col=target_col, test_size=0.2, seed=seed
    )
    model_cust = clone(model)
    model_cust.fit(X_train_cust, y_train_cust)
    y_pred_cust = model_cust.predict(X_test_cust)
    acc_cust = accuracy_score(y_test_cust, y_pred_cust)
    macro_f1_cust = f1_score(y_test_cust, y_pred_cust, average="macro")

    cust_train_cust = set(df.loc[X_train_cust.index, "customer_id"])
    cust_test_cust = set(df.loc[X_test_cust.index, "customer_id"])
    cust_cust_overlap = len(cust_train_cust.intersection(cust_test_cust))

    # Summary table
    print("\n" + "=" * 85)
    print(" " * 24 + "SPLIT EVALUATION BENCHMARK")
    print("=" * 85)
    print(f"{'Split Strategy':<24} {'Train/Test Rows':<18} {'Test Customers':<16} {'Accuracy':<11} {'Macro F1':<10}")
    print("-" * 85)
    print(
        f"{'Random Split (Naive)':<24} "
        f"{f'{len(X_train_rand)} / {len(X_test_rand)}':<18} "
        f"{f'{len(rand_test_cust)} ({rand_overlap_pct:.1f}% shared)':<16} "
        f"{acc_rand*100:>7.2f}%    {macro_f1_rand*100:>7.2f}%"
    )
    print(
        f"{'Customer-Aware':<24} "
        f"{f'{len(X_train_cust)} / {len(X_test_cust)}':<18} "
        f"{f'{len(cust_test_cust)} (0% shared)':<16} "
        f"{acc_cust*100:>7.2f}%    {macro_f1_cust*100:>7.2f}%"
    )
    print("-" * 85)

    gap_acc = (acc_rand - acc_cust) * 100
    gap_f1 = (macro_f1_rand - macro_f1_cust) * 100
    print(f"Generalization Gap: -{gap_acc:4.2f}% Accuracy | -{gap_f1:4.2f}% Macro F1 (Lower is more honest!)")
    print("=" * 85)

    print("\n[Educational Breakdown: Why Customer-Aware Split Gives Lower, Honest Scores]")
    print(f"1. In the Random Split, {rand_cust_overlap} of {len(rand_test_cust)} test customers ({rand_overlap_pct:.1f}%)")
    print("   also had tickets in the training set. The model partially memorized their individual phrasing,")
    print("   game preferences, and complaint frequencies.")
    print("2. In the Customer-Aware Split, exactly 0 customers overlap between train and test.")
    print("   The model is tested strictly on completely new, unseen players.")
    print("3. Lower evaluation metrics here are NOT a failure — they represent an HONEST measure")
    print("   of how the system will perform when real players submit tickets for the first time.\n")

    return {
        "random_split": {
            "accuracy": acc_rand,
            "macro_f1": macro_f1_rand,
            "train_rows": len(X_train_rand),
            "test_rows": len(X_test_rand),
            "overlapping_customers": rand_cust_overlap,
        },
        "customer_aware_split": {
            "accuracy": acc_cust,
            "macro_f1": macro_f1_cust,
            "train_rows": len(X_train_cust),
            "test_rows": len(X_test_cust),
            "overlapping_customers": cust_cust_overlap,
        },
        "generalization_gap": {"accuracy_drop": gap_acc, "macro_f1_drop": gap_f1},
        "predictions": {
            "y_test_cust": y_test_cust,
            "y_pred_cust": y_pred_cust,
        }
    }


# ---------------------------------------------------------------------------
# 4. DIAGNOSTIC EVALUATION REPORT
# ---------------------------------------------------------------------------

def print_evaluation_report(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    y_proba: np.ndarray | None = None,
    label: str = "",
) -> dict[str, Any]:
    """
    Print a complete classification performance report:
    - Overall accuracy and Macro F1 score
    - Per-class precision, recall, F1, and support table
    - Text-based confusion matrix with class row/column labels
    
    Parameters:
        y_true: Ground truth target labels.
        y_pred: Model predicted labels.
        y_proba: Optional predicted probabilities (for calibrated loss).
        label: Context label describing the evaluation condition.
    """
    title = f"EVALUATION REPORT: {label}" if label else "EVALUATION REPORT"
    print("\n" + "=" * 80)
    print(" " * max(0, (80 - len(title)) // 2) + title)
    print("=" * 80)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")

    print(f"Overall Accuracy:  {acc * 100:6.2f}%")
    print(f"Macro-Averaged F1: {macro_f1 * 100:6.2f}%  <-- Primary Metric for Imbalanced Classes")
    print(f"Weighted F1:       {weighted_f1 * 100:6.2f}%\n")

    # Classification report
    print("-" * 80)
    print("Per-Class Classification Metrics:")
    print("-" * 80)
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))

    # Confusion matrix
    classes = sorted(list(set(y_true) | set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    print("-" * 80)
    print("Confusion Matrix (Rows: Actual, Columns: Predicted):")
    print("-" * 80)

    # Clean text-based confusion matrix with truncated headers for readability
    max_label_len = max(len(str(c)) for c in classes)
    header_pad = max(max_label_len, 15)
    col_width = 8

    # Header row
    col_indices = "".join(f"{i:>{col_width}}" for i in range(len(classes)))
    print(f"{'Class Index / Label':<{header_pad}} {col_indices}")
    print("-" * (header_pad + col_width * len(classes) + 1))

    for idx, (cls_name, row) in enumerate(zip(classes, cm)):
        row_str = "".join(f"{val:>{col_width}}" for val in row)
        truncated_label = (cls_name[: header_pad - 4] + "...") if len(cls_name) > header_pad else cls_name
        print(f"[{idx}] {truncated_label:<{header_pad-4}} {row_str}")

    print("-" * (header_pad + col_width * len(classes) + 1))
    print("Legend:")
    for idx, cls_name in enumerate(classes):
        print(f"  [{idx}] {cls_name}")
    print("=" * 80 + "\n")

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "confusion_matrix": cm,
        "classes": classes,
    }


# ---------------------------------------------------------------------------
# 5. CLI ENTRYPOINT & SELF-VERIFICATION
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Module 4: Evaluation Strategy & Leakage Analysis")
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/tickets_clean.csv",
        help="Path to cleaned dataset CSV",
    )
    parser.add_argument(
        "--leakage-only",
        action="store_true",
        help="Run only the data leakage demonstration experiment",
    )
    parser.add_argument(
        "--compare-splits",
        action="store_true",
        help="Run only the Random vs. Customer-Aware split comparison",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="category",
        choices=["category", "priority"],
        help="Target column for split comparison ('category' or 'priority')",
    )
    args = parser.parse_args()

    csv_path = Path(args.input)
    if not csv_path.exists():
        print(f"[Error] Dataset not found at: {csv_path.resolve()}")
        print("Please run 'python src/preprocessing.py' first to generate clean data.")
        sys.exit(1)

    print(f"\n[evaluate.py] Loading cleaned dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"[evaluate.py] Successfully loaded {len(df):,} tickets ({df['customer_id'].nunique():,} unique customers).")

    run_all = not args.leakage_only and not args.compare_splits

    # 1. Leakage Experiment
    if args.leakage_only or run_all:
        leakage_results = leakage_experiment(df, seed=42)

    # 2. Split Comparison & Evaluation Reporting
    if args.compare_splits or run_all:
        split_results = compare_splits(df, target_col=args.target, seed=42)
        y_test = split_results["predictions"]["y_test_cust"]
        y_pred = split_results["predictions"]["y_pred_cust"]
        print_evaluation_report(
            y_test, y_pred, label=f"{args.target.capitalize()} Model (Customer-Aware Split Baseline)"
        )


if __name__ == "__main__":
    main()
