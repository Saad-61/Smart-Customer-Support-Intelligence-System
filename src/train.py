"""
src/train.py
------------
Modules 6 & 7: Model Training & Persistence Pipeline.

Implements model training, cross-validation, probability calibration,
evaluation, and artifact serialization:
- Module 6: Category Classification (Model A: LogisticRegression, Model B: Calibrated LinearSVC).
- Module 7: Priority Prediction (will be integrated here).

Hardware Acceleration:
Detects NVIDIA CUDA GPU and reports VRAM telemetry. Scikit-learn CPU algorithms
utilize multi-threaded vectorization while GPU context is initialized for downstream
deep learning pipelines (Module 8 retrieval).

Usage:
    python src/train.py --model category
    python src/train.py --model category --type both
    python src/train.py --model category --type svm
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

# Hardware check: Detect GPU and PyTorch CUDA
try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
    CUDA_DEVICE_NAME = torch.cuda.get_device_name(0) if HAS_CUDA else "None"
    CUDA_MEMORY_GB = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2) if HAS_CUDA else 0.0
except ImportError:
    HAS_CUDA = False
    CUDA_DEVICE_NAME = "PyTorch not installed"
    CUDA_MEMORY_GB = 0.0

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluate import (
    customer_aware_split,
    print_evaluation_report,
    random_split,
)
from src.features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TEXT_FEATURES,
    build_full_pipeline,
)

TARGET_CATEGORY = "category"
FEATURE_COLS = TEXT_FEATURES + CATEGORICAL_FEATURES + NUMERIC_FEATURES


# ---------------------------------------------------------------------------
# 1. HARDWARE DIAGNOSTIC DISPLAY
# ---------------------------------------------------------------------------

def print_hardware_summary() -> None:
    """Display system compute environment and GPU acceleration status."""
    print("=" * 85)
    print(" " * 28 + "COMPUTE ENGINE & HARDWARE STATUS")
    print("=" * 85)
    if HAS_CUDA:
        print(f"  [GPU Acceleration]: ENABLED")
        print(f"  [Device]:           {CUDA_DEVICE_NAME}")
        print(f"  [Dedicated VRAM]:   {CUDA_MEMORY_GB} GB")
        print(f"  [CUDA Driver/Lib]:  PyTorch {torch.__version__} (Active CUDA Backend)")
    else:
        print("  [Compute Device]:   CPU (CUDA not detected)")
    print(f"  [Feature Engine]:   Scikit-Learn Multi-Threaded Native Backend")
    print("=" * 85 + "\n")


# ---------------------------------------------------------------------------
# 2. PIPELINE FACTORY (CATEGORY CLASSIFIERS)
# ---------------------------------------------------------------------------

def build_category_pipeline(
    model_type: str = "svm",
    max_tfidf_features: int = 15000,
    seed: int = 42,
) -> Pipeline:
    """
    Construct composite Feature Transformer + Classifier Pipeline.
    
    Model A ('lr'):
        TfidfVectorizer (15k features, unigrams+bigrams, sublinear_tf)
        + OneHotEncoder(product) + StandardScaler(metadata)
        + LogisticRegression(class_weight='balanced', C=1.0, max_iter=1000)
        
    Model B ('svm' - Recommended):
        Same composite feature transformer
        + CalibratedClassifierCV(LinearSVC(class_weight='balanced', C=1.0), cv=3, method='sigmoid')
        Platt scaling yields reliable posterior class probabilities from raw SVM decision margins.
    """
    feature_transformer = build_full_pipeline(max_tfidf_features=max_tfidf_features)

    if model_type.lower() == "lr":
        classifier = LogisticRegression(
            class_weight="balanced",
            C=1.0,
            max_iter=1000,
            random_state=seed,
            n_jobs=-1,
        )
    elif model_type.lower() == "svm":
        base_svc = LinearSVC(
            class_weight="balanced",
            C=1.0,
            max_iter=2000,
            random_state=seed,
            dual="auto",
        )
        # CalibratedClassifierCV with sigmoid (Platt Scaling) produces calibrated probabilities
        classifier = CalibratedClassifierCV(
            estimator=base_svc,
            cv=3,
            method="sigmoid",
        )
    else:
        raise ValueError(f"Unknown model_type: '{model_type}'. Supported: 'svm', 'lr'")

    return Pipeline([
        ("features", feature_transformer),
        ("clf", classifier),
    ])


# ---------------------------------------------------------------------------
# 3. TRAINING & EVALUATION ROUTINES
# ---------------------------------------------------------------------------

def train_category_model(
    df: pd.DataFrame,
    model_type: str = "svm",
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[Pipeline, dict[str, Any]]:
    """
    Train category classifier on customer_aware_split and evaluate performance.
    
    Returns:
        (fitted_pipeline, metrics_dict)
    """
    X_train, X_test, y_train, y_test = customer_aware_split(
        df, target_col=TARGET_CATEGORY, test_size=test_size, seed=seed, feature_cols=FEATURE_COLS
    )

    model_label = "Model B (LinearSVC + Platt Scaling)" if model_type == "svm" else "Model A (Logistic Regression)"
    print(f"[train_category_model] Training {model_label} on {len(X_train):,} tickets...")

    pipeline = build_category_pipeline(model_type=model_type, seed=seed)

    t0 = time.perf_counter()
    pipeline.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    print(f"[train_category_model] Training completed in {train_time:.3f} seconds.")

    # Predict labels and class probabilities
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y_test, y_pred, average="weighted"))

    metrics = {
        "model_type": model_type,
        "train_time_sec": train_time,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "classes": list(pipeline.classes_),
    }

    return pipeline, metrics


def evaluate_both_models(df: pd.DataFrame, seed: int = 42) -> tuple[Pipeline, dict[str, Any]]:
    """
    Train both Model A (LR) and Model B (LinearSVC + Platt Scaling).
    Display a side-by-side performance comparison and return the winning pipeline.
    """
    print("\n" + "=" * 85)
    print(" " * 20 + "MODULE 6: CATEGORY CLASSIFIER BENCHMARK")
    print("=" * 85)
    print("Partition Strategy: Customer-Aware Split (Zero customer overlap)")
    print("Target Column:      'category' (10 customer support categories)\n")

    # Train Model A
    pipeline_lr, metrics_lr = train_category_model(df, model_type="lr", seed=seed)

    # Train Model B
    pipeline_svm, metrics_svm = train_category_model(df, model_type="svm", seed=seed)

    # Side-by-side comparison table
    print("\n" + "=" * 85)
    print(" " * 27 + "MODEL COMPARISON SUMMARY")
    print("=" * 85)
    print(f"{'Model Name':<28} {'Accuracy':<12} {'Macro F1':<12} {'Train Time':<14} {'Calibration':<15}")
    print("-" * 85)
    print(
        f"{'Model A: TF-IDF + LR':<28} "
        f"{metrics_lr['accuracy']*100:>8.2f}%    "
        f"{metrics_lr['macro_f1']*100:>8.2f}%    "
        f"{metrics_lr['train_time_sec']:>8.3f}s     "
        f"{'Softmax':<15}"
    )
    print(
        f"{'Model B: TF-IDF + LinearSVC':<28} "
        f"{metrics_svm['accuracy']*100:>8.2f}%    "
        f"{metrics_svm['macro_f1']*100:>8.2f}%    "
        f"{metrics_svm['train_time_sec']:>8.3f}s     "
        f"{'Platt Scaling':<15}"
    )
    print("=" * 85 + "\n")

    # Select best model based on Macro F1 (preferred for imbalanced distributions)
    if metrics_svm["macro_f1"] >= metrics_lr["macro_f1"]:
        print("[Selection]: Model B (Calibrated LinearSVC) selected as production model.")
        best_pipeline, best_metrics = pipeline_svm, metrics_svm
    else:
        print("[Selection]: Model A (Logistic Regression) selected as production model.")
        best_pipeline, best_metrics = pipeline_lr, metrics_lr

    # Display detailed classification report for the chosen model
    print_evaluation_report(
        best_metrics["y_test"],
        best_metrics["y_pred"],
        y_proba=best_metrics["y_proba"],
        label=f"Category Model ({best_metrics['model_type'].upper()} Customer-Aware)",
    )

    return best_pipeline, best_metrics


# ---------------------------------------------------------------------------
# 4. PERSISTENCE ROUTINES
# ---------------------------------------------------------------------------

def save_category_model(
    pipeline: Pipeline,
    path: str | Path = "models/category_model.joblib",
) -> Path:
    """Save the fitted end-to-end classification pipeline to disk."""
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, save_path)
    print(f"[save_category_model] Saved end-to-end model pipeline -> {save_path} ({save_path.stat().st_size / 1024:.1f} KB)")
    return save_path


def save_category_tfidf(
    pipeline: Pipeline,
    path: str | Path = "models/category_tfidf.joblib",
) -> Path:
    """
    Extract and save the fitted TfidfVectorizer object.
    Used for standalone inspection and explainability feature extraction (Module 9).
    """
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        tfidf = pipeline.named_steps["features"].named_transformers_["text"].named_steps["tfidf"]
        joblib.dump(tfidf, save_path)
        print(f"[save_category_tfidf] Saved fitted TF-IDF vectorizer -> {save_path} ({save_path.stat().st_size / 1024:.1f} KB)")
        return save_path
    except Exception as e:
        print(f"[Warning] Could not extract TF-IDF step: {e}")
        return save_path


def load_category_model(path: str | Path = "models/category_model.joblib") -> Pipeline:
    """Load a previously serialized model pipeline for API inference."""
    load_path = Path(path)
    if not load_path.exists():
        raise FileNotFoundError(f"Model artifact not found at: {load_path.resolve()}")
    pipeline = joblib.load(load_path)
    return pipeline


# ---------------------------------------------------------------------------
# 5. CLI ENTRYPOINT
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Module 6 & 7: Model Training Pipeline")
    parser.add_argument(
        "--model",
        type=str,
        default="category",
        choices=["category", "priority"],
        help="Target model to train ('category' [Module 6] or 'priority' [Module 7])",
    )
    parser.add_argument(
        "--type",
        type=str,
        default="both",
        choices=["both", "svm", "lr"],
        help="Model architecture ('both', 'svm' [Calibrated LinearSVC], 'lr' [Logistic Regression])",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/tickets_clean.csv",
        help="Path to clean dataset CSV",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Disable automatic joblib artifact saving",
    )
    args = parser.parse_args()

    # 1. Print hardware and GPU environment
    print_hardware_summary()

    csv_path = Path(args.input)
    if not csv_path.exists():
        print(f"[Error] Clean data file not found at: {csv_path.resolve()}")
        print("Please run 'python src/preprocessing.py' first.")
        sys.exit(1)

    print(f"[train.py] Loading cleaned dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"[train.py] Successfully loaded {len(df):,} tickets across {df['customer_id'].nunique():,} unique customers.\n")

    if args.model == "category":
        if args.type == "both":
            best_pipeline, _ = evaluate_both_models(df, seed=42)
        else:
            best_pipeline, best_metrics = train_category_model(df, model_type=args.type, seed=42)
            print_evaluation_report(
                best_metrics["y_test"],
                best_metrics["y_pred"],
                y_proba=best_metrics["y_proba"],
                label=f"Category Model ({args.type.upper()})",
            )

        if not args.no_save:
            print("-" * 85)
            save_category_model(best_pipeline, "models/category_model.joblib")
            save_category_tfidf(best_pipeline, "models/category_tfidf.joblib")
            print("-" * 85)

    elif args.model == "priority":
        print("[train.py] Priority prediction pipeline is scheduled for Module 7.")
        print("Please run Module 6 first: python src/train.py --model category")


if __name__ == "__main__":
    main()
