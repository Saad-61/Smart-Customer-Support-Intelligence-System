"""
src/train.py
------------
Model Training & Persistence Pipeline.

Implements model training, cross-validation, probability calibration,
evaluation, and artifact serialization:
- Category Classification: Model A (Logistic Regression) vs. Model B (Calibrated LinearSVC).
- Priority Prediction: Model A (GPU-Accelerated XGBoost) vs. Model B (Random Forest).

Hardware Acceleration:
Detects NVIDIA CUDA GPU and utilizes GPU acceleration via XGBoost (tree_method='hist',
device='cuda') and PyTorch. Scikit-learn CPU algorithms utilize native multi-threaded
backends.

Usage:
    python src/train.py --model category
    python src/train.py --model priority
    python src/train.py --model priority --type xgb
    python src/train.py --model priority --type rf
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_class_weight

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

# Check XGBoost availability
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

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
TARGET_PRIORITY = "priority"
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
    if HAS_XGB:
        print(f"  [Gradient Boost]:   XGBoost {xgb.__version__} (GPU Hist Enabled)")
    else:
        print(f"  [Gradient Boost]:   Random Forest Fallback (XGBoost not installed)")
    print(f"  [Linear & Forest]:  Scikit-Learn Multi-Threaded Native Backend")
    print("=" * 85 + "\n")


# ---------------------------------------------------------------------------
# 2. CATEGORY CLASSIFIERS (MODULE 6)
# ---------------------------------------------------------------------------

def build_category_pipeline(
    model_type: str = "svm",
    max_tfidf_features: int = 15000,
    seed: int = 42,
) -> Pipeline:
    """
    Construct composite Feature Transformer + Classifier Pipeline for Category.
    
    Model A ('lr'):
        TfidfVectorizer (15k features, unigrams+bigrams, sublinear_tf)
        + OneHotEncoder(product) + StandardScaler(metadata)
        + LogisticRegression(class_weight='balanced', C=1.0, max_iter=1000)
        
    Model B ('svm' - Recommended):
        Same composite feature transformer
        + CalibratedClassifierCV(LinearSVC(class_weight='balanced', C=1.0), cv=5, method='sigmoid')
        Platt scaling yields reliable posterior class probabilities from raw SVM decision margins.
    """
    feature_transformer = build_full_pipeline(max_tfidf_features=max_tfidf_features)

    if model_type.lower() == "lr":
        classifier = LogisticRegression(
            class_weight="balanced",
            C=1.0,
            max_iter=1000,
            random_state=seed,
        )
    elif model_type.lower() == "svm":
        base_svc = LinearSVC(
            class_weight="balanced",
            C=1.0,
            max_iter=2000,
            random_state=seed,
            dual="auto",
        )
        classifier = CalibratedClassifierCV(
            estimator=base_svc,
            cv=5,
            method="sigmoid",
        )
    else:
        raise ValueError(f"Unknown model_type: '{model_type}'. Supported: 'svm', 'lr'")

    return Pipeline([
        ("features", feature_transformer),
        ("clf", classifier),
    ])


def train_category_model(
    df: pd.DataFrame,
    model_type: str = "svm",
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[Pipeline, dict[str, Any]]:
    """
    Train category classifier on customer_aware_split and evaluate performance.
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
    print(" " * 20 + "CATEGORY CLASSIFIER BENCHMARK")
    print("=" * 85)
    print("Partition Strategy: Customer-Aware Split (Zero customer overlap)")
    print("Target Column:      'category' (10 customer support categories)\n")

    pipeline_lr, metrics_lr = train_category_model(df, model_type="lr", seed=seed)
    pipeline_svm, metrics_svm = train_category_model(df, model_type="svm", seed=seed)

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

    if metrics_svm["macro_f1"] >= metrics_lr["macro_f1"]:
        print("[Selection]: Model B (Calibrated LinearSVC) selected as production model.")
        best_pipeline, best_metrics = pipeline_svm, metrics_svm
    else:
        print("[Selection]: Model A (Logistic Regression) selected as production model.")
        best_pipeline, best_metrics = pipeline_lr, metrics_lr

    print_evaluation_report(
        best_metrics["y_test"],
        best_metrics["y_pred"],
        y_proba=best_metrics["y_proba"],
        label=f"Category Model ({best_metrics['model_type'].upper()} Customer-Aware)",
    )

    return best_pipeline, best_metrics


# ---------------------------------------------------------------------------
# 3. PRIORITY CLASSIFIERS (MODULE 7)
# ---------------------------------------------------------------------------

class PriorityXGBClassifier(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn compatible wrapper for GPU-accelerated XGBoost classifier.
    
    Features:
    - Encodes string priority labels ('HIGH', 'MEDIUM', 'LOW') into integer targets.
    - Computes balanced sample weights dynamically to handle class imbalance.
    - Utilizes 'device=cuda' with 'tree_method=hist' for hardware acceleration on NVIDIA GPU.
    - Outputs string class predictions and calibrated probability matrices.
    """
    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        max_depth: int = 5,
        device: str = "cuda",
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.device = device
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_

        # Dynamic sample weighting to counter priority imbalance (51% MEDIUM vs 19% HIGH)
        if sample_weight is None:
            classes = np.unique(y_encoded)
            weights = compute_class_weight("balanced", classes=classes, y=y_encoded)
            weight_dict = dict(zip(classes, weights))
            sample_weight = np.array([weight_dict[val] for val in y_encoded])

        # Verify device
        active_device = self.device if (self.device == "cuda" and HAS_CUDA) else "cpu"

        self.model_ = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            tree_method="hist",
            device=active_device,
            random_state=self.random_state,
            eval_metric="mlogloss",
        )
        self.model_.fit(X, y_encoded, sample_weight=sample_weight)
        return self

    def predict(self, X):
        preds = self.model_.predict(X)
        return self.label_encoder_.inverse_transform(preds)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


def build_priority_feature_pipeline(
    n_components: int = 50,
    max_tfidf_features: int = 5000,
    seed: int = 42,
) -> ColumnTransformer:
    """
    Construct dimensionality-reduced composite feature transformer for tree models.
    
    Gradient boosting algorithms suffer under high-dimensional sparse representations
    (e.g. 15,000 sparse TF-IDF columns). We compress ticket text into 50 latent semantic
    components using TruncatedSVD (LSA), fused with one-hot encoded product titles
    and standardized operational metadata (59 total dense numeric features).
    """
    text_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=max_tfidf_features,
            ngram_range=(1, 2),
            sublinear_tf=True,
            strip_accents="unicode",
            min_df=2,
        )),
        ("svd", TruncatedSVD(n_components=n_components, random_state=seed)),
    ])

    cat_pipeline = Pipeline([
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    return ColumnTransformer(
        transformers=[
            ("text", text_pipeline, "ticket_text"),
            ("cat", cat_pipeline, CATEGORICAL_FEATURES),
            ("num", num_pipeline, NUMERIC_FEATURES),
        ],
        remainder="drop",
    )


def build_priority_pipeline(
    model_type: str = "xgb",
    n_components: int = 50,
    seed: int = 42,
) -> Pipeline:
    """
    Build complete Priority Prediction Pipeline.
    
    Model A ('xgb' - GPU Recommended):
        Compressed ColumnTransformer (50 SVD + 5 OHE + 4 Metadata = 59 features)
        + PriorityXGBClassifier(device='cuda')
        
    Model B ('rf' - CPU Ensemble Fallback):
        Same compressed ColumnTransformer
        + RandomForestClassifier(n_estimators=200, class_weight='balanced')
    """
    features = build_priority_feature_pipeline(n_components=n_components, seed=seed)

    if model_type.lower() == "xgb":
        if not HAS_XGB:
            raise ImportError("XGBoost is not installed. Please use model_type='rf' or pip install xgboost.")
        device = "cuda" if HAS_CUDA else "cpu"
        classifier = PriorityXGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            device=device,
            random_state=seed,
        )
    elif model_type.lower() == "rf":
        classifier = RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown priority model_type: '{model_type}'. Supported: 'xgb', 'rf'")

    return Pipeline([
        ("features", features),
        ("clf", classifier),
    ])


def train_priority_model(
    df: pd.DataFrame,
    model_type: str = "xgb",
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[Pipeline, dict[str, Any]]:
    """
    Train priority classifier on customer_aware_split and evaluate performance.
    """
    X_train, X_test, y_train, y_test = customer_aware_split(
        df, target_col=TARGET_PRIORITY, test_size=test_size, seed=seed, feature_cols=FEATURE_COLS
    )

    accel = "GPU (CUDA)" if (model_type == "xgb" and HAS_CUDA) else "CPU (Multi-Threaded)"
    model_label = f"Model {'A' if model_type == 'xgb' else 'B'} ({model_type.upper()} [{accel}])"
    print(f"[train_priority_model] Training {model_label} on {len(X_train):,} tickets...")

    pipeline = build_priority_pipeline(model_type=model_type, seed=seed)

    t0 = time.perf_counter()
    pipeline.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    print(f"[train_priority_model] Training completed in {train_time:.3f} seconds.")

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


def evaluate_both_priority_models(df: pd.DataFrame, seed: int = 42) -> tuple[Pipeline, dict[str, Any]]:
    """
    Train both Model A (XGBoost GPU) and Model B (Random Forest CPU).
    Display a side-by-side performance comparison and return the winning pipeline.
    """
    print("\n" + "=" * 85)
    print(" " * 20 + "PRIORITY CLASSIFIER BENCHMARK")
    print("=" * 85)
    print("Partition Strategy: Customer-Aware Split (Zero customer overlap)")
    print("Target Column:      'priority' (HIGH, MEDIUM, LOW)")
    print("Feature Fusion:     50 SVD Components + Product OHE + Operational Metadata (59 dense cols)")
    print("Leakage Safeguard:  'resolution_time' and 'resolved' are STRICTLY EXCLUDED\n")

    # Train Model A (XGBoost GPU)
    if HAS_XGB:
        pipeline_xgb, metrics_xgb = train_priority_model(df, model_type="xgb", seed=seed)
    else:
        pipeline_xgb, metrics_xgb = None, None

    # Train Model B (Random Forest CPU)
    pipeline_rf, metrics_rf = train_priority_model(df, model_type="rf", seed=seed)

    print("\n" + "=" * 85)
    print(" " * 26 + "PRIORITY MODEL COMPARISON")
    print("=" * 85)
    print(f"{'Model Architecture':<32} {'Accuracy':<12} {'Macro F1':<12} {'Train Time':<14} {'Compute Engine':<15}")
    print("-" * 85)

    if metrics_xgb is not None:
        gpu_label = f"GPU ({CUDA_DEVICE_NAME[:16]})" if HAS_CUDA else "CPU Hist"
        print(
            f"{'Model A: XGBoost (CUDA)':<32} "
            f"{metrics_xgb['accuracy']*100:>8.2f}%    "
            f"{metrics_xgb['macro_f1']*100:>8.2f}%    "
            f"{metrics_xgb['train_time_sec']:>8.3f}s     "
            f"{gpu_label:<15}"
        )

    print(
        f"{'Model B: Random Forest (CPU)':<32} "
        f"{metrics_rf['accuracy']*100:>8.2f}%    "
        f"{metrics_rf['macro_f1']*100:>8.2f}%    "
        f"{metrics_rf['train_time_sec']:>8.3f}s     "
        f"{'CPU Multi-Core':<15}"
    )
    print("=" * 85 + "\n")

    # Select best model by Macro F1
    if metrics_xgb is not None and metrics_xgb["macro_f1"] >= metrics_rf["macro_f1"]:
        print("[Selection]: Model A (XGBoost GPU) selected as production priority model.")
        best_pipeline, best_metrics = pipeline_xgb, metrics_xgb
    else:
        print("[Selection]: Model B (Random Forest CPU) selected as production priority model.")
        best_pipeline, best_metrics = pipeline_rf, metrics_rf

    print_evaluation_report(
        best_metrics["y_test"],
        best_metrics["y_pred"],
        y_proba=best_metrics["y_proba"],
        label=f"Priority Model ({best_metrics['model_type'].upper()} Customer-Aware)",
    )

    return best_pipeline, best_metrics


# ---------------------------------------------------------------------------
# 4. PERSISTENCE ROUTINES
# ---------------------------------------------------------------------------

def save_category_model(
    pipeline: Pipeline,
    path: str | Path = "models/category_model.joblib",
) -> Path:
    """Save the fitted category classification pipeline to disk."""
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, save_path)
    print(f"[save_category_model] Saved category pipeline -> {save_path} ({save_path.stat().st_size / 1024:.1f} KB)")
    return save_path


def save_category_tfidf(
    pipeline: Pipeline,
    path: str | Path = "models/category_tfidf.joblib",
) -> Path:
    """Extract and save the fitted TF-IDF vectorizer for category explainability."""
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
    """Load serialized category model pipeline."""
    load_path = Path(path)
    if not load_path.exists():
        raise FileNotFoundError(f"Model artifact not found at: {load_path.resolve()}")
    return joblib.load(load_path)


def save_priority_model(
    pipeline: Pipeline,
    path: str | Path = "models/priority_model.joblib",
) -> Path:
    """Save the fitted priority prediction pipeline to disk."""
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, save_path)
    print(f"[save_priority_model] Saved priority pipeline -> {save_path} ({save_path.stat().st_size / 1024:.1f} KB)")
    return save_path


def load_priority_model(path: str | Path = "models/priority_model.joblib") -> Pipeline:
    """Load serialized priority model pipeline."""
    load_path = Path(path)
    if not load_path.exists():
        raise FileNotFoundError(f"Model artifact not found at: {load_path.resolve()}")
    # Register PriorityXGBClassifier on __main__ for unpickling compatibility
    main_mod = sys.modules.setdefault("__main__", sys.modules[__name__])
    if not hasattr(main_mod, "PriorityXGBClassifier"):
        setattr(main_mod, "PriorityXGBClassifier", PriorityXGBClassifier)
    return joblib.load(load_path)


# ---------------------------------------------------------------------------
# 5. CLI ENTRYPOINT
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Model Training Pipeline: Category & Priority")
    parser.add_argument(
        "--model",
        type=str,
        default="category",
        choices=["category", "priority"],
        help="Target model to train ('category' or 'priority')",
    )
    parser.add_argument(
        "--type",
        type=str,
        default="both",
        help="Model architecture: For category: 'both', 'svm', 'lr'. For priority: 'both', 'xgb', 'rf'.",
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

    # Branch 1: Category Classification
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

    # Branch 2: Priority Prediction
    elif args.model == "priority":
        if args.type == "both":
            best_pipeline, _ = evaluate_both_priority_models(df, seed=42)
        else:
            best_pipeline, best_metrics = train_priority_model(df, model_type=args.type, seed=42)
            print_evaluation_report(
                best_metrics["y_test"],
                best_metrics["y_pred"],
                y_proba=best_metrics["y_proba"],
                label=f"Priority Model ({args.type.upper()})",
            )

        if not args.no_save:
            print("-" * 85)
            save_priority_model(best_pipeline, "models/priority_model.joblib")
            print("-" * 85)


if __name__ == "__main__":
    main()
