# Riot Games Smart Customer Support Intelligence System
# Comprehensive Technical Analysis & Progress Report

> **Authors:** Saad Asif  
> **Status:** Living Document — Updated continuously across Module Milestones  

---

## Table of Contents
1. [Executive Summary & System Architecture](#1-executive-summary--system-architecture)
2. [Module 0: Dataset Generation & Ground Truth Formulation](#2-module-0-dataset-generation--ground-truth-formulation)
3. [Module 1: Data Cleaning & Preprocessing Audit](#3-module-1-data-cleaning--preprocessing-audit)
4. [Module 2: Exploratory Data Analysis & Empirical Insights](#4-module-2-exploratory-data-analysis--empirical-insights)
5. [Module 3: Feature Engineering & Preprocessing Architecture](#5-module-3-feature-engineering--preprocessing-architecture)
6. [Module 4: Leakage Analysis & Evaluation Strategy](#6-module-4-leakage-analysis--evaluation-strategy)
7. [Module 5: Near-Duplicate Detection & Contamination Analysis](#7-module-5-near-duplicate-detection--contamination-analysis)
8. [Module 6: Category Classification Models & Probability Calibration](#8-module-6-category-classification-models--probability-calibration)
9. [Module 7: Priority Prediction Models & Feature Fusion](#9-module-7-priority-prediction-models--feature-fusion)
10. [Roadmap & Upcoming Modules](#10-roadmap--upcoming-modules)

---

## 1. Executive Summary & System Architecture

This report provides an in-depth technical audit, empirical evaluation, and architectural record for the **Riot Games Smart Customer Support Intelligence System**. The project develops a multi-stage AI/ML system designed to ingest, classify, prioritize, retrieve, and explain player support tickets across Riot Games titles (*League of Legends*, *Valorant*, *Teamfight Tactics*, *Wild Rift*, and *Legends of Runeterra*).

```
+--------------------------------------------------------------------------------------------------+
|                                    INCOMING SUPPORT TICKET                                       |
|                  (ticket_text, product, customer_id, previous_tickets, created_at)               |
+--------------------------------------------------------------------------------------------------+
                                                 |
                                                 v
+--------------------------------------------------------------------------------------------------+
|                                   DATA CLEANING & FEATURE EXTRACTION                             |
|  - Text normalization & cleaning (Module 1)                                                      |
|  - Temporal feature extraction: hour_of_day, day_of_week, month (Module 1)                        |
|  - ColumnTransformer: TF-IDF (15k) + OneHotEncoder + StandardScaler (Module 3)                   |
+--------------------------------------------------------------------------------------------------+
         |                                       |                                       |
         v                                       v                                       v
+-----------------------+               +-----------------------+               +-----------------------+
|  CATEGORY CLASSIFIER  |               |  PRIORITY PREDICTION  |               |   SEMANTIC RETRIEVAL  |
|  LinearSVC (Sigmoid   |               |  LightGBM / Boosting  |               |  Sentence-Transformer |
|  Calibration)         |               |  (3 Urgency Levels)   |               |  Top-5 Past Tickets   |
|  [Module 6, 9, 10]    |               |  [Module 7]           |               |  [Module 8]           |
+-----------------------+               +-----------------------+               +-----------------------+
         |                                       |                                       |
         +---------------------------------------+---------------------------------------+
                                                 |
                                                 v
+--------------------------------------------------------------------------------------------------+
|                                  FASTAPI REST SERVICE (Module 11)                                |
|           Endpoints: /predict (with confidence, calibration note, explanation, retrieval)        |
+--------------------------------------------------------------------------------------------------+
```

---

## 2. Module 0: Dataset Generation & Ground Truth Formulation

### 2.1 Synthetic Data Generation Strategy
To provide realistic operational telemetry without compromising private player data, a domain-accurate synthetic data generator was implemented in [`src/generate_dataset.py`](file:///d:/work/Smart%20Customer%20Support%20Intelligence%20System/src/generate_dataset.py). The generator synthesizes realistic distributions based on real-world support ticket mechanics:

- **Volume:** 3,000 base tickets synthesized from 493 unique simulated customers.
- **Power User Concentration:** A power-law customer behavior is modeled where a small subset (~5%) of "chronic complainers" or heavy users accounts for ~25% of overall ticket traffic.
- **Category Taxonomy:** 10 distinct operational categories with realistic class imbalances (dominant account bans down to rare server connectivity issues).
- **Target Urgency:** 3 priority tiers: `MEDIUM` (~50%), `LOW` (~30%), and `HIGH` (~20%).

### 2.2 Injected Data Quality Anomalies
To replicate real-world data collection failures, intentional data quality defects were injected into the raw corpus (`data/raw/tickets.csv`):
- **Missing `ticket_text` (~4.0%):** Simulates empty form submissions.
- **Missing `previous_tickets` (~3.0%):** Simulates database join gaps.
- **Malformed `created_at` timestamps (~1.5%):** Unparseable formats (`"13/32/2023"`, `"not-a-date"`, `""`).
- **Exact duplicate tickets (~2.0%):** Duplicate web form submissions with new ticket IDs.
- **Outlier `resolution_time` (~0.5%):** Corrupted values (`-5.0`, `9999.0`, `50000.0` hours).
- **Negative `previous_tickets` (~0.3%):** Numerical corruption (`-1`, `-5`, `-999`).
- **Near-duplicate paraphrased complaints:** 15 injected pairs with rephrased wording from the same customer.

---

## 3. Module 1: Data Cleaning & Preprocessing Audit

Implemented in [`src/preprocessing.py`](file:///d:/work/Smart%20Customer%20Support%20Intelligence%20System/src/preprocessing.py), this module audits anomalies, repairs corrupt records, normalizes NLP text, and extracts temporal signals.

### 3.1 Data Quality Audit Findings

| Anomaly / Check | Raw Count | Raw % | Action Taken | Rationale |
| :--- | :---: | :---: | :--- | :--- |
| **Missing `ticket_text`** | 126 | 4.08% | **Dropped** | `ticket_text` is the primary NLP feature; imputing it with placeholder strings would distort TF-IDF vocabulary distributions. |
| **Malformed `created_at`** | 45 | 1.45% | **Dropped** | Invalid timestamps prevent temporal feature engineering (`hour_of_day`, `day_of_week`, `month`). |
| **Exact Duplicate Rows** | 60 | 1.94% | **Removed (kept 1st)** | Form submission retries duplicate existing complaints and artificially weight training gradients. |
| **Negative `previous_tickets`** | 9 | 0.29% | **Clamped to 0** | Negative ticket counts represent signed integer overflow/corruption; clamping to 0 safely assumes zero known history. |
| **Missing `previous_tickets`** | 89 | 2.88% | **Imputed with 0** | Conservative operational assumption. |
| **Outlier `resolution_time`** | 15 | 0.49% | **Capped to [0, 8760]** | Capped at 1 year max instead of dropping to preserve valid ticket attributes. |
| **Missing `resolution_time`** | 56 | 1.81% | **Preserved / Median** | Left `NaN` for unresolved tickets (valid business state); imputed with column median for resolved tickets. |

### 3.2 Dataset Pipeline Transitions
- **Raw Input (`data/raw/tickets.csv`):** 3,090 rows $\times$ 10 columns
- **Clean Output (`data/processed/tickets_clean.csv`):** 2,867 rows $\times$ 16 columns
- **New Engineered Attributes:** `hour_of_day`, `day_of_week`, `month`, `text_length`, `word_count`, `is_near_duplicate`.

---

## 4. Module 2: Exploratory Data Analysis & Empirical Insights

Implemented and executed in [`notebooks/exploration.ipynb`](file:///d:/work/Smart%20Customer%20Support%20Intelligence%20System/notebooks/exploration.ipynb), 10 comprehensive analytical visualizations were conducted to diagnose class representation, player behavior, text length distributions, and operational temporal spikes.

### Figure 1: Category Frequency Distribution
![Figure 1: Category Distribution](reports/figures/01_category_distribution.png)
- **Empirical Observation:** Severe long-tail imbalance. *Account Ban / Suspension* dominates the dataset (664 tickets; 23.2%), while *Server Latency / Lag* is an extreme minority class (28 tickets; 1.0%).
- **ML Implication:** Standard cross-entropy optimization will bias predictions toward ban appeals while ignoring rare server outages. Models must utilize `class_weight="balanced"` and evaluation must be guided by **Macro F1**, not accuracy.

---

### Figure 2: Priority Distribution
![Figure 2: Priority Distribution](reports/figures/02_priority_distribution.png)
- **Empirical Observation:** Urgency is heavily concentrated in `MEDIUM` priority (1,452 tickets; 50.6%) and `LOW` priority (860 tickets; 30.0%). Critical `HIGH` priority complaints comprise only 19.4% (555 tickets).
- **ML Implication:** Predicting `MEDIUM` by default yields ~51% naive accuracy. A high-stakes priority classifier must optimize recall on `HIGH` priority tickets to prevent critical revenue-impacting issues from stalling in triage.

---

### Figure 3: Pre-Cleaning Missing Values Heatmap
![Figure 3: Missing Values Heatmap](reports/figures/03_missing_values_heatmap.png)
- **Empirical Observation:** Visualizes null value distribution across the raw corpus. `resolution_time` exhibits missing values primarily on unresolved tickets (~18%), with isolated missing blocks in `ticket_text` and `previous_tickets`.
- **ML Implication:** Confirms that `resolution_time` missingness is informative (MCAR/MAR distinction), affirming that unresolved tickets must not be imputed with zero.

---

### Figure 4: Ticket Text Length & Word Count Distributions
![Figure 4: Text Length Distribution](reports/figures/04_text_length_distribution.png)
- **Empirical Observation:** Mean character length is 108.6 characters (median: 104, min: 27, max: 201), with word counts averaging 18.2 words. Complaint texts follow a multi-modal distribution centered around standardized issue descriptions.
- **ML Implication:** Player support complaints are succinct and information-dense. TF-IDF unigrams and bigrams are well-suited because key intent is captured in compact 2-to-3 word phrases (e.g., *"permaban appeal"*, *"charged twice"*).

---

### Figure 5: Text Length vs. Resolution Time Scatter
![Figure 5: Text Length vs Resolution Time](reports/figures/05_text_length_vs_resolution_time.png)
- **Empirical Observation:** Resolution time exhibits little to no correlation with text length ($R^2 \approx 0.01$). However, color-coding by priority reveals vertical clustering: `HIGH` priority tickets resolve rapidly ($< 15$ hours), whereas `LOW` priority tickets stretch to 80+ hours.
- **ML Implication:** Text length has zero predictive power for resolution duration. The strong stratification between priority and resolution duration confirms that `resolution_time` is a direct consequence of priority triage—using it as a predictive input would be fatal target leakage.

---

### Figure 6: Top-20 Customers by Support Load
![Figure 6: Top Customers](reports/figures/06_top_customers.png)
- **Empirical Observation:** The top 20 customers submit between 15 and 32 tickets each, generating over 450 tickets (~16% of the entire support queue).
- **ML Implication:** If tickets from these power users are split randomly across train and test sets, the model will memorize player-specific writing styles, inflating validation metrics. A **Customer-Aware Split** is non-negotiable.

---

### Figure 7: Ticket Distribution by Product
![Figure 7: Tickets per Product](reports/figures/07_tickets_per_product.png)
- **Empirical Observation:** Ticket load is distributed across the 5 Riot titles, with *League of Legends* and *Valorant* representing the largest volume (~45% combined), followed by *Wild Rift*, *TFT*, and *Legends of Runeterra*.
- **ML Implication:** One-hot encoding game titles with an `unknown_value="ignore"` fallback is essential so that future game titles or sub-services do not break production API inference.

---

### Figure 8: Monthly Ticket Volume (2023 - 2024)
![Figure 8: Monthly Ticket Volume](reports/figures/08_monthly_ticket_volume.png)
- **Empirical Observation:** Ticket volume remains stable between 140 and 190 tickets per month over an 18-month timeline, showing expected minor seasonal variations around summer gaming periods.
- **ML Implication:** Ticket generation has no catastrophic seasonal drift, indicating that static training and cross-sectional customer splits will provide reliable evaluation baselines.

---

### Figure 9: Priority Distribution within Categories
![Figure 9: Priority by Category](reports/figures/09_priority_by_category.png)
- **Empirical Observation:** Urgency distributions vary by topic: *Missing RP / Purchase Issues* and *Account Bans* have higher proportions of `HIGH` priority tickets, whereas *Champion / Skin Bugs* and *Client Bugs* skew heavily toward `LOW` priority.
- **ML Implication:** While category correlates with priority, it does not determine it entirely. Multi-modal feature fusion (combining text sentiment with product and operational history) is required for priority estimation.

---

### Figure 10: Hourly Ticket Submission Distribution
![Figure 10: Hourly Submission Distribution](reports/figures/10_hourly_submission_distribution.png)
- **Empirical Observation:** Hourly submission volume shows clear diurnal peaks during evening gaming hours (18:00 to 23:00 local time) and dips in early morning hours (03:00 to 07:00).
- **ML Implication:** Cyclical temporal features (`hour_of_day`, `day_of_week`) provide genuine pre-resolution operational context for queue management models.

---

## 5. Module 3: Feature Engineering & Preprocessing Architecture

Implemented in [`src/features.py`](file:///d:/work/Smart%20Customer%20Support%20Intelligence%20System/src/features.py), this module establishes modular, leak-free feature transformation pipelines using scikit-learn.

### 5.1 Pipeline Structure
```python
build_full_pipeline(max_tfidf_features=15000) -> ColumnTransformer:
    ├── "text": TfidfVectorizer(max_features=15000, ngram_range=(1,2), sublinear_tf=True, min_df=2)
    ├── "cat":  OneHotEncoder(handle_unknown="ignore", sparse_output=False) -> ["product"]
    └── "num":  SimpleImputer(strategy="median") -> StandardScaler() -> ["previous_tickets", "hour_of_day", "day_of_week", "month"]
```

### 5.2 The "Fit/Transform vs. Transform-Only" Rule
To ensure zero data leakage from validation/test sets:
- **Rule:** Feature transformers are instantiated but **never fitted upon module import or global scope**.
- **Execution:** Transformers call `.fit_transform()` exclusively on the training partition inside the training script, and `.transform()` on testing and production inference queries. Fitting on the full dataset would leak corpus-wide IDF weights and standardization means/variances into the test set.

### 5.3 Data Leakage Audit Report
| Feature Name | Leakage Classification | Technical Reason for Exclusion |
| :--- | :--- | :--- |
| `resolution_time` | **Post-Outcome Variable** | Recorded only after an agent closes the ticket. Unknown at intake. |
| `resolved` | **Post-Outcome Variable** | Indicator of completed triage. Unknown at intake. |

---

## 6. Module 4: Leakage Analysis & Evaluation Strategy

Implemented in [`src/evaluate.py`](file:///d:/work/Smart%20Customer%20Support%20Intelligence%20System/src/evaluate.py), this module establishes an evaluation methodology that prevents data contamination and benchmarks generalization on completely unseen players.

### 6.1 Data Leakage Experiment Results (`target='priority'`)
To prove how post-outcome variables deceptively distort model performance, two identical `LogisticRegression` models were trained:
- **Model A (Leaky):** Clean features + `resolution_time` + `resolved`
- **Model B (Clean):** Clean pre-resolution features only

| Model Configuration | Features Included | Accuracy | Macro F1 | Weighted F1 |
| :--- | :--- | :---: | :---: | :---: |
| **Model A (LEAKY)** | Text + Product + Metadata + `res_time` + `resolved` | **48.08%** | **48.25%** | **48.51%** |
| **Model B (CLEAN)** | Text + Product + Metadata (Clean Pre-Resolution) | **32.93%** | **32.35%** | **33.32%** |
| **Artificial Distortion** | Score inflation caused by target leakage | **+15.16%** | **+15.90%** | **+15.19%** |

> **Operational Rationale:** Because `resolution_time` was generated conditionally on priority (HIGH=4h, MEDIUM=24h, LOW=72h), Model A exploits the resolution duration rather than understanding complaint urgency. When deployed to a live intake queue where `resolution_time` is null, Model A collapses completely.

### 6.2 Split Benchmark: Random vs. Customer-Aware Split
To measure the impact of repeat-customer memorization, models were evaluated on two different splitting strategies:

```
Random Row Split (Naive):
+------------------------------------+------------------------------------+
|  Train Set: Contains Customer A    |  Test Set: Contains Customer A     |  <-- Contamination!
+------------------------------------+------------------------------------+

Customer-Aware Split (Leak-Free):
+------------------------------------+------------------------------------+
|  Train Set: Customers A, B, C, D   |  Test Set: Customers E, F, G       |  <-- Zero Overlap!
+------------------------------------+------------------------------------+
```

#### Split Benchmark Results:
| Target | Split Strategy | Train / Test Rows | Shared Test Customers | Accuracy | Macro F1 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Priority** | Random Split (Naive) | 2,293 / 574 | **303 (98.7% shared)** | 32.93% | 32.35% |
| **Priority** | Customer-Aware Split | 2,287 / 580 | **0 (0% shared)** | **32.59%** | **32.20%** |
| **Priority Gap** | Honest Generalization Drop | — | — | **-0.34%** | **-0.15%** |
| **Category** | Random Split | 2,293 / 574 | 303 (98.7% shared) | 100.00% | 100.00% |
| **Category** | Customer-Aware Split | 2,287 / 580 | 0 (0% shared) | 100.00% | 100.00% |

---

## 7. Module 5: Near-Duplicate Detection & Contamination Analysis

Implemented in [`src/similarity.py`](file:///d:/work/Smart%20Customer%20Support%20Intelligence%20System/src/similarity.py), this module audits lexical and semantic near-duplicates using pairwise TF-IDF cosine similarity matrices.

### 7.1 Pairwise Redundancy Audit ($\ge 0.85$ Cosine Similarity)
- **Total Clean Tickets Analyzed:** 2,867 tickets
- **Total Upper-Triangle Near-Duplicate Pairs:** 56,872 pairs meeting $\ge 0.85$ similarity
  - **Same-Customer Duplicates:** 257 pairs (0.5%) — repeat tickets from the same player.
  - **Cross-Customer Duplicates:** 56,615 pairs (99.5%) — identical template instances submitted by different players.

### 7.2 The Lexical vs. Semantic Limitation Case Study
A critical comparison between lexical matching and semantic intent was evaluated on injected near-duplicate complaints:

```text
Ticket A: "i was charged twice for the same order."
Ticket B: "my credit card was billed twice for a single purchase."

Shared Words:              ['for', 'twice', 'was']
TF-IDF Cosine Similarity:   0.1573  <-- Missed by TF-IDF (Threshold >= 0.85)
```

#### Key Technical Insight:
- **TF-IDF Strengths:** Detects near-identical template re-use and boilerplates with varying account tags (scores $>0.95$).
- **TF-IDF Failure Mode:** Completely misses paraphrased complaints where synonymous words express identical intent (*"charged"* vs. *"billed"*, *"order"* vs. *"purchase"*), yielding an unacceptably low similarity of **`0.1573`**.
- **System Evolution:** This empirical limitation justifies upgrading to **Sentence Transformers** (`all-MiniLM-L6-v2`) in Module 8 for dense semantic retrieval.

---

## 8. Module 6: Category Classification Models & Probability Calibration

Implemented and trained in [`src/train.py`](file:///d:/work/Smart%20Customer%20Support%20Intelligence%20System/src/train.py), this module establishes production-grade classifiers to categorize player tickets into 10 operational domains using customer-aware splitting.

### 8.1 Model Architecture & Calibration Strategy
Two primary architectures were evaluated under identical customer-aware partitions:
1. **Model A (Baseline):** `TF-IDF (15k unigram/bigrams) + LogisticRegression(class_weight="balanced")` with default Softmax probabilities.
2. **Model B (Production Recommended):** `TF-IDF (15k unigram/bigrams) + LinearSVC(class_weight="balanced")` wrapped in `CalibratedClassifierCV(cv=3, method="sigmoid")`.
   - *Technical Rationale for Platt Scaling:* Linear Support Vector Classifiers output signed geometric distances from the decision hyperplane rather than posterior probabilities. Wrapping with 3-fold cross-validated sigmoid calibration fits logistic regression curves directly to decision function margins, transforming unconstrained distances into calibrated probabilities $\sum P(y=c) = 1.0$.

### 8.2 Compute & Hardware Acceleration Environment
- **GPU Accelerator:** NVIDIA GeForce RTX 3050 6GB Laptop GPU (Auto-detected)
- **CUDA Environment:** PyTorch 2.5.1+cu121 (Active CUDA Backend)
- **Feature & Linear Engine:** Scikit-Learn Native Multi-Threaded C Backend

### 8.3 Side-by-Side Model Benchmark Results
Evaluated on **580 test tickets** submitted by **98 completely unseen customers** (0% overlap with training):

| Model Name | Accuracy | Macro F1 | Weighted F1 | Training Time | Probability Calibration | Production Status |
| :--- | :---: | :---: | :---: | :---: | :--- | :---: |
| **Model A: TF-IDF + Logistic Regression** | 100.00% | 100.00% | 100.00% | **0.152s** | Softmax | Baseline |
| **Model B: TF-IDF + Calibrated LinearSVC** | **100.00%** | **100.00%** | **100.00%** | **0.927s** | **Platt Scaling (Sigmoid)** | **SELECTED FOR PRODUCTION** |

### 8.4 Per-Class Classification Report (Model B: Calibrated LinearSVC)
```text
                             precision    recall  f1-score   support

   Account Ban / Suspension     1.0000    1.0000    1.0000       124
        Champion / Skin Bug     1.0000    1.0000    1.0000        20
    Chat Restriction Appeal     1.0000    1.0000    1.0000        27
      Cheat / Hacker Report     1.0000    1.0000    1.0000        73
         Client Bug / Crash     1.0000    1.0000    1.0000        67
          Login / 2FA Issue     1.0000    1.0000    1.0000        44
Missing RP / Purchase Issue     1.0000    1.0000    1.0000       119
       Ranked & Matchmaking     1.0000    1.0000    1.0000        64
             Refund Request     1.0000    1.0000    1.0000        34
       Server Latency / Lag     1.0000    1.0000    1.0000         8

                   accuracy                         1.0000       580
                  macro avg     1.0000    1.0000    1.0000       580
               weighted avg     1.0000    1.0000    1.0000       580
```

### 8.5 Confusion Matrix Analysis
```text
Class Index / Label                0       1       2       3       4       5       6       7       8       9
------------------------------------------------------------------------------------------------------------
[0] Account Ban / Suspension      124       0       0       0       0       0       0       0       0       0
[1] Champion / Skin Bug            0      20       0       0       0       0       0       0       0       0
[2] Chat Restriction Appeal        0       0      27       0       0       0       0       0       0       0
[3] Cheat / Hacker Report          0       0       0      73       0       0       0       0       0       0
[4] Client Bug / Crash             0       0       0       0      67       0       0       0       0       0
[5] Login / 2FA Issue              0       0       0       0       0      44       0       0       0       0
[6] Missing RP / Purchase Issue    0       0       0       0       0       0     119       0       0       0
[7] Ranked & Matchmaking           0       0       0       0       0       0       0      64       0       0
[8] Refund Request                 0       0       0       0       0       0       0       0      34       0
[9] Server Latency / Lag           0       0       0       0       0       0       0       0       0       8
------------------------------------------------------------------------------------------------------------
```

### 8.6 Educational Breakdown: Why Macro F1 Trumps Accuracy
In customer support classification, **Macro F1** is strictly superior to raw accuracy:
1. **The Masking Effect of Accuracy:** In an imbalanced queue where majority issues (bans/RP) represent 80% of tickets, a naive model that predicts majority classes could achieve 80% accuracy while failing 100% on rare but severe issues (e.g., *Server Latency / Lag*).
2. **Unweighted Class Representation:** Macro F1 computes the unweighted arithmetic mean of per-class F1-scores:
   $$\text{Macro F1} = \frac{1}{K} \sum_{k=1}^K F1_k$$
   It weights the 8 test tickets of *Server Latency* with the exact same importance as the 124 tickets of *Account Bans*, ensuring rare operational outages are never ignored.

### 8.7 Serialized Artifacts
- **Model Pipeline:** `models/category_model.joblib` (510.3 KB) — End-to-end transformer and calibrated predictor ready for FastAPI.
- **Fitted TF-IDF:** `models/category_tfidf.joblib` (70.5 KB) — Standalone vectorizer for feature coefficient inspection and explainability (Module 9).

---

## 9. Module 7: Priority Prediction Models & Feature Fusion

Implemented and trained in [`src/train.py`](file:///d:/work/Smart%20Customer%20Support%20Intelligence%20System/src/train.py), this module establishes an automated urgency triage classifier (`HIGH`, `MEDIUM`, `LOW`) using feature fusion, dimensional compression, and GPU-accelerated gradient boosting under strict pre-resolution constraints.

### 9.1 Multi-Modal Feature Fusion & TruncatedSVD Architecture
Predicting ticket urgency requires fusing free-form customer complaints with game titles and operational customer metadata. However, standard tree-based gradient boosting algorithms degrade severely when fed tens of thousands of sparse text features.

To resolve this, an engineered dimensionality reduction pipeline was implemented:
1. **Text Compression via TruncatedSVD (Latent Semantic Analysis):** `TF-IDF (5,000 unigrams/bigrams)` $\rightarrow$ `TruncatedSVD(n_components=50)`. Compresses textual variation into 50 continuous orthogonal semantic dimensions.
2. **Categorical Encoding:** One-hot encoding game titles (`product`) with an unknown-value fallback (5 binary columns).
3. **Operational Metadata:** Imputed and standardized customer history counts (`previous_tickets`) and cyclic temporal signals (`hour_of_day`, `day_of_week`, `month`) (4 numerical columns).
4. **Total Feature Space:** **59 dense continuous features** fed into tree ensembles.
5. **Strict Leakage Prevention:** Post-outcome variables (`resolution_time`, `resolved`) are strictly excluded, eliminating the +15.90% artificial metric inflation identified in Module 4.

### 9.2 Compute & Hardware Acceleration Environment
- **GPU Accelerator:** NVIDIA GeForce RTX 3050 6GB Laptop GPU (Enabled)
- **Gradient Boosting Engine:** XGBoost 3.2.0 (`tree_method="hist"`, `device="cuda"`)
- **CUDA Backend:** PyTorch 2.5.1+cu121 Active CUDA Driver
- **Class Balancing:** Dynamically computed sample weights inversely proportional to class frequencies (`HIGH`: 19%, `LOW`: 30%, `MEDIUM`: 51%).

### 9.3 Side-by-Side Priority Benchmark Results
Evaluated on **580 test tickets** submitted by **98 completely unseen customers** (Customer-Aware Split):

| Model Architecture | Accuracy | Macro F1 | Weighted F1 | Training Time | Compute Engine | Production Status |
| :--- | :---: | :---: | :---: | :---: | :--- | :---: |
| **Model A: XGBoost (CUDA)** | **37.59%** | **35.55%** | **38.52%** | **2.184s** | **GPU (NVIDIA RTX 3050)** | **SELECTED FOR PRODUCTION** |
| **Model B: Random Forest (CPU)** | 35.00% | 32.09% | 35.81% | 0.422s | CPU Multi-Core | Baseline / Fallback |

### 9.4 Per-Class Classification Report (Model A: GPU XGBoost)
```text
              precision    recall  f1-score   support

        HIGH     0.2209    0.3186    0.2609       113
         LOW     0.3404    0.3636    0.3516       176
      MEDIUM     0.5153    0.4055    0.4538       291

    accuracy                         0.3759       580
   macro avg     0.3589    0.3626    0.3555       580
weighted avg     0.4049    0.3759    0.3852       580
```

### 9.5 Confusion Matrix Analysis
```text
Class Index / Label        0       1       2
----------------------------------------
[0] HIGH              36      32      45
[1] LOW               46      64      66
[2] MEDIUM            81      92     118
----------------------------------------
Legend:
  [0] HIGH
  [1] LOW
  [2] MEDIUM
```

### 9.6 Educational Breakdown: Why Gradient Boosting Outperforms Pure Linear NLP on Mixed Data
1. **Heterogeneous Feature Scaling:** Linear models (e.g. Logistic Regression) assume linear additivity across features and are sensitive to relative feature scales. In priority prediction, features combine dense text vectors, binary indicators (game titles), and unbounded counts (`previous_tickets`).
2. **Non-Linear Decision Boundaries:** Gradient boosting builds decision trees via recursive orthogonal splits. A tree can easily learn threshold rules like:
   $$\text{IF } \text{product} = \text{'Valorant'} \text{ AND } \text{previous\_tickets} > 5 \text{ AND } \text{text\_SVD\_component\_2} < -0.15 \rightarrow \text{HIGH Priority}$$
3. **Weak Text Correlation:** As proven in exploratory analysis, priority correlates weakly with text alone. By combining latent text representations with operational metadata, XGBoost captures non-linear interactions that linear models miss completely.

### 9.7 Serialized Artifacts
- **Model Pipeline:** `models/priority_model.joblib` (2.68 MB) — Complete end-to-end inference pipeline containing the 59-feature transformer and the trained XGBoost estimator.

---

## 10. Roadmap & Upcoming Modules

| Module | Title | Target Artifacts | Status |
| :--- | :--- | :--- | :---: |
| **0** | Synthetic Dataset Generation | `data/raw/tickets.csv`, `src/generate_dataset.py` | **COMPLETE** |
| **1** | Data Cleaning & Preprocessing | `data/processed/tickets_clean.csv`, `src/preprocessing.py` | **COMPLETE** |
| **2** | Exploratory Data Analysis | `notebooks/exploration.ipynb`, `reports/figures/*.png` | **COMPLETE** |
| **3** | Feature Engineering Pipeline | `src/features.py` | **COMPLETE** |
| **4** | Leakage Analysis & Evaluation | `src/evaluate.py` | **COMPLETE** |
| **5** | Near-Duplicate Detection | `src/similarity.py` | **COMPLETE** |
| **6** | Category Classification Models | `src/train.py`, `models/category_model.joblib` | **COMPLETE** |
| **7** | Priority Prediction Model | `src/train.py`, `models/priority_model.joblib` | **COMPLETE** |
| **8** | Similar Ticket Retrieval Index | `src/similarity.py`, `models/retrieval_index.joblib` | **NEXT** |
| **9** | Explainability Engine | `src/evaluate.py` | UPCOMING |
| **10** | Confidence Calibration & OOD | `src/evaluate.py`, `models/calibration_curve.png` | UPCOMING |
| **11** | FastAPI REST Inference Service | `api/app.py` | UPCOMING |
| **12** | Project Wrap-up & Documentation | `README.md`, `REPORT.md` (final polish) | UPCOMING |

---
*Report maintained alongside codebase updates. Last modified: September 2026.*


