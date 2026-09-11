"""
src/evaluate.py
---------------
Evaluation Methodologies (Module 4) & Model Explainability Engine (Module 9).

Features:
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
5. Model Explainability (Module 9):
   - Extracts linear decision hyperplanes from CalibratedClassifierCV(LinearSVC).
   - Computes local active feature contributions (x_j * w_j) for words in player tickets.
   - Generates human-readable explanations formatted for API serving.
   - Documents explainability limitations (correlation vs causation, unigrams vs SHAP).

Usage:
    # Module 4: Leakage & Split Evaluation
    python src/evaluate.py
    python src/evaluate.py --leakage-only
    python src/evaluate.py --compare-splits

    # Module 9: Model Explainability Demonstration & Queries
    python src/evaluate.py --demo-explain
    python src/evaluate.py --explain "I was charged twice for the same RP bundle"
    python src/evaluate.py --document-explain-limitations
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import joblib

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


# ===========================================================================
# 5. CATEGORY EXPLAINABILITY ENGINE (MODULE 9)
# ===========================================================================

def explain_category_prediction(
    text: str,
    pipeline: Any = None,
    top_n: int = 10,
    product: str = "League of Legends",
    customer_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Generate feature attributions explaining why the category model classified a ticket.

    Extracts class coefficients from the LinearSVC pipeline (averaged across Platt scaling
    calibration folds) and computes input-specific contributions for words/bigrams present
    in the complaint.

    Parameters:
        text: Raw player complaint string.
        pipeline: Trained category pipeline (default: loaded from models/category_model.joblib).
        top_n: Number of top features to return.
        product: Game title for the ticket (default: "League of Legends").
        customer_metadata: Optional dict with keys: previous_tickets, hour_of_day, day_of_week, month.

    Returns:
        dict format:
        {
            "predicted_category": str,
            "confidence": float,
            "top_features": [
                {"feature": str, "weight": float, "contribution": float},
                ...
            ]
        }
    """
    if pipeline is None:
        model_path = REPO_ROOT / "models" / "category_model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Category model not found at {model_path}. "
                "Please run 'python src/train.py --model category' first."
            )
        pipeline = joblib.load(model_path)

    meta = customer_metadata or {}
    sample_df = pd.DataFrame([{
        "ticket_text": text.strip() if text else "",
        "product": product,
        "previous_tickets": meta.get("previous_tickets", 0),
        "hour_of_day": meta.get("hour_of_day", 12),
        "day_of_week": meta.get("day_of_week", 2),
        "month": meta.get("month", 6),
    }])

    # Predict category and confidence
    pred = pipeline.predict(sample_df)[0]
    predicted_category = str(pred)

    clf = pipeline.named_steps["clf"]
    features_step = pipeline.named_steps["features"]
    feature_names = features_step.get_feature_names_out()

    classes = list(clf.classes_)
    pred_idx = classes.index(predicted_category)

    if hasattr(pipeline, "predict_proba"):
        probas = pipeline.predict_proba(sample_df)[0]
        confidence = float(probas[pred_idx])
    else:
        confidence = 1.0

    # Extract linear coefficients across calibration folds
    if hasattr(clf, "calibrated_classifiers_"):
        coef_list = [cc.estimator.coef_ for cc in clf.calibrated_classifiers_]
        avg_coefs = np.mean(coef_list, axis=0)  # Shape: (n_classes, n_features)
    elif hasattr(clf, "coef_"):
        avg_coefs = clf.coef_
    else:
        raise ValueError(f"Classifier {type(clf)} does not expose linear coefficients.")

    class_weights = avg_coefs[pred_idx]  # Shape: (n_features,)

    # Transform input to identify active features (x_j)
    x_vec = features_step.transform(sample_df)
    active_indices = set(x_vec.nonzero()[1])

    def clean_feature_name(raw_name: str) -> str:
        return (
            raw_name.replace("text__", "")
            .replace("cat__product_", "Product: ")
            .replace("num__", "")
        )

    # 1. Collect features active in the input text with positive class weights
    active_features: list[dict[str, Any]] = []
    for idx in active_indices:
        w = float(class_weights[idx])
        val = float(x_vec[0, idx])
        contrib = val * w
        if w > 0 and contrib > 0:
            active_features.append({
                "feature": clean_feature_name(feature_names[idx]),
                "weight": round(w, 4),
                "contribution": round(contrib, 4),
            })
    # Sort active features by local contribution descending
    active_features.sort(key=lambda item: item["contribution"], reverse=True)

    # 2. If fewer active features than top_n, supplement with global class salient words
    if len(active_features) < top_n:
        sorted_global_indices = np.argsort(-class_weights)
        seen_names = {item["feature"] for item in active_features}
        for g_idx in sorted_global_indices:
            w = float(class_weights[g_idx])
            if w <= 0:
                break
            feat_name = clean_feature_name(feature_names[g_idx])
            if feat_name not in seen_names:
                active_features.append({
                    "feature": feat_name,
                    "weight": round(w, 4),
                    "contribution": 0.0,
                })
                seen_names.add(feat_name)
                if len(active_features) >= top_n:
                    break

    top_features = active_features[:top_n]

    return {
        "predicted_category": predicted_category,
        "confidence": round(confidence, 4),
        "top_features": top_features,
    }


def format_explanation(explanation_dict: dict[str, Any]) -> str:
    """
    Format explanation dictionary into a clean, human-readable string.
    """
    category = explanation_dict.get("predicted_category", "Unknown")
    confidence = explanation_dict.get("confidence", 0.0)
    top_features = explanation_dict.get("top_features", [])

    lines = [
        f"Predicted Category: {category} (Confidence: {confidence * 100:.1f}%)",
        "Key Contributing Words & Signals:",
    ]
    for idx, f in enumerate(top_features, start=1):
        feat_name = f["feature"]
        weight = f["weight"]
        contrib = f.get("contribution", 0.0)
        if contrib > 0:
            lines.append(f"  {idx:>2}. \"{feat_name}\" -> weight: {weight:+.4f} (local contribution: {contrib:+.4f})")
        else:
            lines.append(f"  {idx:>2}. \"{feat_name}\" -> class weight: {weight:+.4f}")
    return "\n".join(lines)


def demonstrate_explainability(pipeline: Any = None, top_n: int = 5) -> list[dict[str, Any]]:
    """
    Run explainability on 4 diverse support complaints and print formatted outputs.
    """
    if pipeline is None:
        model_path = REPO_ROOT / "models" / "category_model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Category model not found at {model_path}. "
                "Please run 'python src/train.py --model category' first."
            )
        pipeline = joblib.load(model_path)

    sample_complaints = [
        (
            "Account Ban Appeal",
            "I received a 14-day suspension for allegedly using scripts. I have never used an unauthorized program.",
            "League of Legends",
        ),
        (
            "Missing RP / Purchase Issue",
            "I purchased 1350 RP but my credit card was charged twice and no coins appeared in my account.",
            "League of Legends",
        ),
        (
            "Client Bug / Crash Report",
            "Game freezes and crashes during champion select every time with a fatal directx error.",
            "Valorant",
        ),
        (
            "Server Latency / Lag",
            "Constant high ping and severe packet loss every evening making ranked games unplayable.",
            "League of Legends",
        ),
    ]

    print("\n" + "=" * 90)
    print(" " * 26 + "DEMONSTRATION: MODEL EXPLAINABILITY")
    print("=" * 90)

    results = []
    for title, text, prod in sample_complaints:
        print(f"\n[{title}] (Product: {prod})")
        print(f"Complaint: \"{text}\"")
        print("-" * 90)
        explanation = explain_category_prediction(text, pipeline=pipeline, top_n=top_n, product=prod)
        print(format_explanation(explanation))
        print("-" * 90)
        results.append(explanation)

    return results


def document_explainability_limitations() -> str:
    """
    Return and print a comprehensive written explanation of linear model explainability limitations.
    """
    limitations = """
========================================================================================
                      MODEL EXPLAINABILITY: METHODOLOGICAL LIMITATIONS
========================================================================================

1. Correlation vs. Causation:
   - Linear coefficients reflect statistical correlation within the training dataset,
     NOT causal reasoning or true linguistic understanding.
   - For example, if the word 'ticket' appears frequently in purchase complaints in the
     synthetic training set, the model may assign it high positive weight, even though
     'ticket' is an operational artifact, not a causal indicator of a billing problem.

2. Unigrams and Bigrams Limitations:
   - TF-IDF with (1, 2) n-grams treats features as independent tokens or adjacent pairs.
   - It is completely blind to:
     * Long-range dependencies and sentence grammar.
     * Sarcasm or irony (e.g. 'Great job Riot, the client crashed again!').
     * Complex negations (e.g. 'I was NOT banned for toxicity, my brother was').

3. Explanation Applies Only to the Final Linear Layer:
   - The linear coefficients w_j explain how the model scores coordinates AFTER the
     multi-step feature pipeline (TF-IDF sublinear scaling, IDF weighting, OHE, standard scaling).
   - They do not explain the full nonlinear end-to-end data transformation.

4. Spurious Correlations and Class Imbalances:
   - In classes with small support (e.g. Server Latency with 28 samples), unique customer
     writing habits (e.g. specific player names, timestamps, ISPs) can become spurious
     high-weight features that overfit to those few samples.

5. Global Class Weights vs. Local SHAP Attribution:
   - LinearSVC decision hyperplanes (w_c) are inherently GLOBAL per class.
   - While instance-level weighting (x_j * w_j) pinpoints which words were active in a
     specific complaint, it treats each feature additively without interaction effects.
   - More rigorous game-theoretic frameworks (such as SHAP LinearExplainer or TreeExplainer)
     measure the marginal contribution of each feature across all possible feature subsets,
     providing true local attribution against a baseline background distribution.
========================================================================================
"""
    print(limitations)
    return limitations


# ===========================================================================
# 6. CLI ENTRYPOINT & SELF-VERIFICATION
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Module 4 (Evaluation & Leakage) and Module 9 (Model Explainability)"
    )
    # Module 4 arguments
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
    # Module 9 arguments
    parser.add_argument(
        "--demo-explain",
        action="store_true",
        help="Run explainability demonstration across diverse sample complaints",
    )
    parser.add_argument(
        "--explain",
        type=str,
        default=None,
        help="Explain category prediction for an ad-hoc customer complaint text",
    )
    parser.add_argument(
        "--product",
        type=str,
        default="League of Legends",
        help="Product name for ad-hoc complaint explanation (default: 'League of Legends')",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of top contributing features to return (default: 5)",
    )
    parser.add_argument(
        "--document-explain-limitations",
        action="store_true",
        help="Print written explanation of linear model explainability limitations",
    )

    args = parser.parse_args()

    # 1. Explainability limitations
    if args.document_explain_limitations:
        document_explainability_limitations()
        return

    # 2. Ad-hoc query explanation
    if args.explain:
        explanation = explain_category_prediction(
            text=args.explain,
            top_n=args.top_n,
            product=args.product,
        )
        print("\n" + "=" * 80)
        print(f"Complaint: \"{args.explain}\" (Product: {args.product})")
        print("-" * 80)
        print(format_explanation(explanation))
        print("=" * 80 + "\n")
        return

    # 3. Explainability demonstration
    if args.demo_explain:
        demonstrate_explainability(top_n=args.top_n)
        document_explainability_limitations()
        return

    # Default Module 4 workflow
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
