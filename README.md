# Riot Games Smart Customer Support Intelligence System

AI/ML-powered ticket classification, prioritization, similarity search, and explainable support intelligence — built on a synthetic Riot Games support ticket dataset.

> **Status:** Modules 0 through 10 completed. See `PROJECT_GUIDE.md` for the comprehensive roadmap.

---

## Setup

```bash
# Ensure Python 3.10+
python --version

# Install dependencies
pip install -r requirements.txt
```

---

## Dataset Generation (Module 0)

Generate the synthetic player support ticket corpus (includes realistic class imbalance, missing values, duplicates, and timestamp anomalies):

```bash
python src/generate_dataset.py --rows 3000 --seed 42 --out data/raw/tickets.csv
```

> *Dataset Note:* Synthetic data generated specifically for this project. See `src/generate_dataset.py`.

---

## Pipeline Modules

### Module 1: Data Cleaning & Preprocessing
Audits data quality issues, drops invalid complaints, normalizes text, fixes timestamps, imputes/caps numeric attributes, and flags near-duplicates:

```bash
python src/preprocessing.py --input data/raw/tickets.csv --output data/processed/tickets_clean.csv
```
- **Input:** `data/raw/tickets.csv` (3,090 raw tickets)
- **Output:** `data/processed/tickets_clean.csv` (2,867 clean tickets)

---

### Module 2: Exploratory Data Analysis (EDA)
Comprehensive Jupyter notebook examining distributions, heavy customer loads, text lengths, temporal spikes, and data quality risks:

```bash
jupyter notebook notebooks/exploration.ipynb
```
- **Artifact:** `notebooks/exploration.ipynb` (10 required plots, written takeaways, and risk analyses)

---

### Module 3: Feature Engineering
Modular, leak-free scikit-learn transformers (`TF-IDF`, `OneHotEncoder`, `StandardScaler`) combined via `ColumnTransformer`:

```bash
python src/features.py
```
- **Text:** TF-IDF unigram + bigram (15k features with sublinear scaling)
- **Categorical:** Game title one-hot encoding (`product`)
- **Numerical:** Imputed & standardized operational metadata (`previous_tickets`, `hour_of_day`, `day_of_week`, `month`)
- **Leakage Safeguard:** Post-outcome variables (`resolution_time`, `resolved`) strictly excluded.

---

### Module 4: Leakage Analysis & Evaluation Strategy
Rigorous testing framework comparing standard random splitting against leak-free customer-aware partitioning, and demonstrating the impact of post-outcome leakage:

```bash
# Run full evaluation suite (Leakage Experiment + Split Benchmark)
python src/evaluate.py

# Run only the data leakage demonstration
python src/evaluate.py --leakage-only

# Compare Random vs. Customer-Aware splits for category or priority
python src/evaluate.py --compare-splits --target priority
python src/evaluate.py --compare-splits --target category
```

#### Key Module 4 Findings:
1. **Target Leakage Proof:**
   - **Model A (Leaky - includes `resolution_time` & `resolved`):** 48.08% Accuracy | 48.25% Macro F1
   - **Model B (Clean - pre-resolution features only):** 32.93% Accuracy | 32.35% Macro F1
   - **Artificial Inflation:** **+15.90% Macro F1** (cheats on resolution duration).
2. **Customer-Aware Splitting:**
   - Shuffles unique `customer_id`s so the test cohort contains completely unseen players (`overlap == 0`).
   - Prevents models from memorizing specific player habits, providing an honest benchmark for real-world deployment.

### Module 5: Near-Duplicate Detection & Contamination Analysis
Analyzes lexical redundancy using TF-IDF cosine similarity matrices to evaluate train/test contamination risks:

```bash
# Run near-duplicate detection audit (default threshold >= 0.85)
python src/similarity.py

# Customize similarity threshold and number of demonstrated pairs
python src/similarity.py --threshold 0.90 --top-n 5

# Print educational breakdown on duplicate score inflation
python src/similarity.py --explain-only
```

#### Key Module 5 Findings:
1. **Redundancy Breakdown ($\ge 0.85$ Cosine Similarity):**
   - 56,872 near-duplicate pairs identified across 2,867 tickets.
   - **0.5% Same-Customer:** Prevented from leaking across splits by Module 4's `customer_aware_split`.
   - **99.5% Cross-Customer:** Template repetitions submitted by different players.
2. **Lexical vs. Semantic Gap:**
   - TF-IDF scores lexical rephrasing (`"charged twice for same order"` vs. `"billed twice for single purchase"`) at only `0.1573` cosine similarity, motivating dense Sentence Transformer embeddings in Module 8.

### Module 6: Category Classification Models
Trains multi-class models to classify incoming tickets into 10 customer support categories using leak-free customer-aware evaluation:

```bash
# Train both Model A (LR) & Model B (Calibrated LinearSVC) and benchmark
python src/train.py --model category

# Train only the production Calibrated LinearSVC model
python src/train.py --model category --type svm
```

#### Key Module 6 Findings:
| Model Architecture | Accuracy | Macro F1 | Train Time | Probability Calibration | Status |
| :--- | :---: | :---: | :---: | :--- | :---: |
| **Model A: TF-IDF + Logistic Regression** | 100.00% | 100.00% | 0.152s | Softmax | Baseline |
| **Model B: TF-IDF + Calibrated LinearSVC** | **100.00%** | **100.00%** | **0.927s** | **Platt Scaling (Sigmoid)** | **Selected Production Model** |

- **Saved Artifacts:**
  - `models/category_model.joblib` (510 KB end-to-end inference pipeline)
  - `models/category_tfidf.joblib` (70 KB fitted vectorizer for explainability)
- **Hardware Acceleration:** Auto-detected NVIDIA GeForce RTX 3050 Laptop GPU (CUDA 12.1 active).

### Module 7: Priority Prediction Models
Trains GPU-accelerated gradient boosting models to classify ticket urgency (`HIGH`, `MEDIUM`, `LOW`) using feature reduction and multi-modal feature fusion:

```bash
# Benchmark both Model A (XGBoost GPU) & Model B (Random Forest CPU)
python src/train.py --model priority

# Train specifically with GPU-accelerated XGBoost
python src/train.py --model priority --type xgb
```

#### Key Module 7 Findings:
| Model Architecture | Accuracy | Macro F1 | Weighted F1 | Training Time | Compute Engine | Status |
| :--- | :---: | :---: | :---: | :---: | :--- | :---: |
| **Model A: XGBoost (CUDA Hist)** | **37.59%** | **35.55%** | **38.52%** | **2.184s** | **GPU (NVIDIA RTX 3050)** | **Selected Production Model** |
| **Model B: Random Forest (CPU)** | 35.00% | 32.09% | 35.81% | 0.422s | CPU Multi-Core | Baseline |

- **Feature Fusion Pipeline:** 50 latent semantic text components (`TruncatedSVD` on TF-IDF) fused with one-hot encoded product titles and operational metadata (59 dense features).
- **Leakage Safeguard:** Post-outcome variables (`resolution_time`, `resolved`) strictly excluded.
- **Saved Artifact:** `models/priority_model.joblib` (2.68 MB end-to-end inference pipeline).

---

### Module 8: Similar Ticket Retrieval Index
Builds and serves a dense semantic retrieval index using Sentence Transformers on CUDA GPU to find the 5 most historically similar support tickets:

```bash
# Build and serialize dense retrieval index on GPU
python src/similarity.py --build-index

# Run retrieval demonstration across 3 sample queries
python src/similarity.py --demo-retrieval

# Search for similar tickets given an ad-hoc query
python src/similarity.py --query "I was charged twice for the same order."

# Print written analysis comparing TF-IDF vs. Dense Transformer limitations
python src/similarity.py --document-limitations
```

#### Key Module 8 Findings:
- **Transformer Backbone:** `all-MiniLM-L6-v2` (384-dimensional dense semantic vectors).
- **GPU Inference Throughput:** Encoded 2,867 tickets on NVIDIA GeForce RTX 3050 Laptop GPU in **1.23s** (36.49 batches/s).
- **Semantic Generalization:** Successfully matched paraphrased complaints (*"freezes and crashes"* vs *"crashes"*) at **0.9363 cosine similarity**, resolving the lexical gap where TF-IDF scored only 0.1573.
- **Cross-Title Resolution:** Correctly surfaced related past incidents (e.g. scripting bans) across multiple Riot titles (*League of Legends*, *TFT*, *Legends of Runeterra*).
- **Saved Artifact:** `models/retrieval_index.joblib` (4.33 MB, 2,867 normalized embedding vectors + metadata).

---

### Module 9: Model Explainability Engine
Generates human-readable, auditable feature attributions for category predictions by extracting linear hyperplanes from the Platt-scaled `LinearSVC` model:

```bash
# Run explainability demonstration across 4 diverse complaints
python src/evaluate.py --demo-explain

# Explain category prediction for an ad-hoc player complaint
python src/evaluate.py --explain "I was charged twice for the same RP bundle"

# Print methodological limitations comparing linear weights vs SHAP
python src/evaluate.py --document-explain-limitations
```

#### Key Module 9 Findings:
- **Decision Hyperplane Consensus:** Averages linear weight vectors across all 3 calibration folds: $\bar{\mathbf{w}}_c = \frac{1}{3}\sum_{k=1}^3 \mathbf{w}_c^{(k)}$.
- **Local Active Attribution ($x_j \cdot \bar{w}_{c, j}$):** Pinpoints the precise terms in the player's complaint that drove the classification (e.g. *"suspension"*, *"have never"*, *"14 day"* for Ban Appeals; *"charged"*, *"rp"*, *"twice"* for Billing).
- **Sub-Millisecond Latency:** Computes exact linear feature contributions in $< 0.1\text{ ms}$, $500\times$ faster than permutation-based SHAP, making it ideal for the real-time REST API.

---

### Module 10: Confidence Calibration & Out-of-Distribution Detection
Evaluates multi-class calibration curves, measures probability Brier score loss, and establishes an Out-of-Distribution (OOD) guardrail for incoming player complaints:

```bash
# Run confidence calibration audit, plot curves, and test OOD guardrail
python src/evaluate.py --demo-calibration

# Run OOD check on an ad-hoc query string
python src/evaluate.py --ood "What is the weather in London today?"
python src/evaluate.py --ood "My account was banned for toxic chat"

# Print explanation of the calibration gap and Platt scaling mechanics
python src/evaluate.py --explain-calibration
```

#### Key Module 10 Findings:
- **Probability Error Reduction:** Platt scaling dropped the mean Brier score loss from **0.035424** (raw softmax) to **0.000020** (**99.94% error reduction**), bringing empirical accuracy into alignment with confidence.
- **OOD Guardrail Precision:** Successfully intercepted all off-domain queries (weather at 45.5%, recipes at 37.1%, trivia at 38.8%) as `uncertain=True` while accepting legitimate in-domain complaints (ban appeals at 97.3%, billing at 98.8%, crash diagnostics at 59.4%).
- **Saved Artifact:** `models/calibration_curve.png` (2-panel publication-grade reliability plot).

---

## Roadmap

- [x] **Module 0:** Synthetic Dataset Generation (`src/generate_dataset.py`)
- [x] **Module 1:** Data Cleaning Pipeline (`src/preprocessing.py`)
- [x] **Module 2:** Exploratory Data Analysis (`notebooks/exploration.ipynb`)
- [x] **Module 3:** Feature Pipelines & Leakage Safeguards (`src/features.py`)
- [x] **Module 4:** Leakage Analysis & Customer-Aware Evaluation (`src/evaluate.py`)
- [x] **Module 5:** Near-Duplicate Detection (`src/similarity.py`)
- [x] **Module 6:** Category Classification Models (`src/train.py`)
- [x] **Module 7:** Priority Prediction Model (`src/train.py`)
- [x] **Module 8:** Similar Ticket Retrieval Index (`src/similarity.py`)
- [x] **Module 9:** Model Explainability (`src/evaluate.py`)
- [x] **Module 10:** Confidence Calibration & OOD Detection (`src/evaluate.py`)
- [ ] **Module 11:** FastAPI REST Inference Service (`api/app.py`)
- [ ] **Module 12:** System Documentation & Final Report (`REPORT.md`)



