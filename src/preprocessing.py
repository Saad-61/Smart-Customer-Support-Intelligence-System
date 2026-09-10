"""
src/preprocessing.py
--------------------
Module 1: Data Cleaning & Preparation Pipeline.

Audits raw support ticket data, cleans text, fixes timestamps, handles
missing values and numerical outliers, removes exact duplicate records,
and flags near-duplicate submissions for downstream inspection.

Usage:
    python src/preprocessing.py
    python src/preprocessing.py --input data/raw/tickets.csv --output data/processed/tickets_clean.csv
"""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. LOAD RAW DATA
# ---------------------------------------------------------------------------

def load_raw_data(path: str | Path = "data/raw/tickets.csv") -> pd.DataFrame:
    """Load raw tickets CSV dataset and log dimensions and schema."""
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Raw data file not found at: {csv_path.resolve()}")

    df = pd.read_csv(csv_path)
    print(f"\n[load_raw_data] Successfully loaded {csv_path}")
    print(f"[load_raw_data] Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"[load_raw_data] Data types:\n{df.dtypes.to_string()}\n")
    return df


# ---------------------------------------------------------------------------
# 2. DATA QUALITY AUDIT
# ---------------------------------------------------------------------------

def audit_data_quality(df: pd.DataFrame) -> dict[str, Any]:
    """
    Perform a comprehensive audit of data quality issues.
    
    Identifies:
    - Missing values per column
    - Exact duplicate tickets (ignoring ticket_id)
    - Malformed timestamps that cannot be parsed
    - Negative counts in previous_tickets (corruption)
    - Unrealistic resolution_time outliers (< 0 or > 8760 hours / 1 year)
    """
    total_rows = len(df)
    missing_counts = df.isnull().sum().to_dict()

    # Exact duplicates: excluding ticket_id
    feature_cols = [c for c in df.columns if c != "ticket_id"]
    exact_duplicates = int(df.duplicated(subset=feature_cols).sum())

    # Timestamp audit: attempt parsing
    parsed_dates = pd.to_datetime(df["created_at"], errors="coerce")
    malformed_timestamps = int(parsed_dates.isna().sum())

    # Numeric anomalies
    negative_prev_tickets = int((df["previous_tickets"] < 0).sum()) if "previous_tickets" in df else 0
    res_time = df["resolution_time"].dropna()
    res_outliers = int(((res_time < 0) | (res_time > 8760)).sum())

    audit_report = {
        "total_rows": total_rows,
        "missing_counts": missing_counts,
        "exact_duplicates": exact_duplicates,
        "malformed_timestamps": malformed_timestamps,
        "negative_previous_tickets": negative_prev_tickets,
        "resolution_time_outliers": res_outliers,
    }

    # Print clean summary table
    print("=" * 65)
    print(" " * 20 + "DATA QUALITY AUDIT REPORT")
    print("=" * 65)
    print(f"Total Rows Scanned: {total_rows:,}")
    print("-" * 65)
    print(f"{'Issue / Check':<38} {'Count':<10} {'Percentage':<12}")
    print("-" * 65)

    for col, null_count in missing_counts.items():
        if null_count > 0:
            pct = (null_count / total_rows) * 100
            print(f"Missing in '{col}':{null_count:>22,} ({pct:5.2f}%)")

    print(f"Exact Duplicate Rows:{exact_duplicates:>20,} ({(exact_duplicates/total_rows)*100:5.2f}%)")
    print(f"Malformed Timestamps:{malformed_timestamps:>20,} ({(malformed_timestamps/total_rows)*100:5.2f}%)")
    print(f"Negative previous_tickets:{negative_prev_tickets:>15,} ({(negative_prev_tickets/total_rows)*100:5.2f}%)")
    print(f"Outlier resolution_time (<0 or >1yr):{res_outliers:>4,} ({(res_outliers/total_rows)*100:5.2f}%)")
    print("=" * 65 + "\n")

    return audit_report


# ---------------------------------------------------------------------------
# 3. TEXT CLEANING
# ---------------------------------------------------------------------------

def clean_ticket_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the primary NLP feature: ticket_text.
    
    Decisions & Rationale:
    - Drop rows where ticket_text is null: ticket_text is the primary input
      for category classification, sentiment, similarity retrieval, and
      explainability. An empty complaint has no semantic content to classify
      or retrieve. Imputation (e.g. placeholder string) would distort TF-IDF
      and embedding distributions.
    - Strip leading and trailing whitespace.
    - Normalize repeated spaces/tabs/newlines to a single space.
    - Lowercase all characters for uniform vocabulary matching.
    """
    df = df.copy()
    initial_count = len(df)

    # 1. Drop null ticket_text
    df = df.dropna(subset=["ticket_text"])
    dropped = initial_count - len(df)
    print(f"[clean_ticket_text] Dropped {dropped:,} rows with missing 'ticket_text' "
          f"({(dropped / initial_count) * 100:.2f}%). Rationale: Primary NLP signal cannot be imputed.")

    # 2. String normalization: strip, collapse whitespace, lowercase
    df["ticket_text"] = (
        df["ticket_text"]
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.lower()
    )

    return df


# ---------------------------------------------------------------------------
# 4. TIMESTAMP PARSING & TEMPORAL FEATURE EXTRACTION
# ---------------------------------------------------------------------------

def fix_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse timestamps with error handling and extract temporal features.
    
    Decisions & Rationale:
    - Parse using pd.to_datetime(..., errors='coerce') to safely catch malformed formats.
    - Drop rows where created_at is NaT: Temporal features (hour, day of week,
      month) are required for seasonality and load modeling. Synthetically
      imputing ticket submission times creates temporal leakage or arbitrary
      distributions.
    - Extract hour_of_day (0-23), day_of_week (0-6), and month (1-12) as integers.
    """
    df = df.copy()
    initial_count = len(df)

    # Coerce to datetime
    parsed_dates = pd.to_datetime(df["created_at"], errors="coerce")
    nat_mask = parsed_dates.isna()
    dropped = int(nat_mask.sum())

    if dropped > 0:
        df = df[~nat_mask].copy()
        parsed_dates = parsed_dates[~nat_mask]
        print(f"[fix_timestamps] Dropped {dropped:,} rows with unparseable timestamps "
              f"({(dropped / initial_count) * 100:.2f}%). Rationale: Corrupted timestamps corrupt temporal features.")

    # Assign parsed datetime and derived components
    df["created_at"] = parsed_dates
    df["hour_of_day"] = df["created_at"].dt.hour.astype(int)
    df["day_of_week"] = df["created_at"].dt.dayofweek.astype(int)
    df["month"] = df["created_at"].dt.month.astype(int)

    return df


# ---------------------------------------------------------------------------
# 5. STRUCTURED FEATURE IMPUTATION & OUTLIER HANDLING
# ---------------------------------------------------------------------------

def handle_missing_structured(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute structured missing values and bound numeric outliers.
    
    Decisions & Rationale:
    - previous_tickets missing: Imputed with 0. Conservative business prior:
      if historical ticket count was not logged, assume this is the user's
      first support contact.
    - previous_tickets < 0: Replaced with 0. Negative contact count is physically
      impossible; clipping corrects data entry corruption.
    - resolution_time for unresolved tickets (resolved == False): Left as NaN.
      An unresolved ticket does NOT have a resolution duration yet. Imputing a
      duration would fabricate false resolution signals.
    - resolution_time for resolved tickets (resolved == True) that are missing:
      Imputed with the median resolution_time of resolved tickets. Median is
      robust to heavy right skew in support ticket lifetimes.
    - resolution_time outliers (< 0 or > 8760 hours): Capped to [0, 8760].
      Negative duration is impossible. Durations exceeding 1 year (8,760 hrs)
      reflect stale forgotten tickets or logging bugs. Capping preserves the
      row while preventing extreme values from distorting scaling and variance.
    """
    df = df.copy()

    # 1. previous_tickets
    if "previous_tickets" in df.columns:
        n_missing_prev = int(df["previous_tickets"].isna().sum())
        df["previous_tickets"] = df["previous_tickets"].fillna(0)
        n_neg_prev = int((df["previous_tickets"] < 0).sum())
        df["previous_tickets"] = df["previous_tickets"].clip(lower=0)
        print(f"[handle_missing_structured] 'previous_tickets': Filled {n_missing_prev:,} nulls with 0; "
              f"clipped {n_neg_prev:,} negative values to 0.")

    # 2. resolution_time
    if "resolution_time" in df.columns and "resolved" in df.columns:
        # Check outliers before imputation
        valid_res = df["resolution_time"].dropna()
        outliers_count = int(((valid_res < 0) | (valid_res > 8760)).sum())

        # Cap outliers to [0, 8760] (1 year maximum)
        df["resolution_time"] = df["resolution_time"].clip(lower=0.0, upper=8760.0)
        print(f"[handle_missing_structured] 'resolution_time': Capped {outliers_count:,} outliers to [0.0, 8760.0] hrs.")

        # Impute missing values ONLY for resolved tickets
        resolved_mask = df["resolved"] == True
        resolved_median = df.loc[resolved_mask, "resolution_time"].median()

        missing_resolved_mask = resolved_mask & df["resolution_time"].isna()
        n_imputed_res = int(missing_resolved_mask.sum())
        df.loc[missing_resolved_mask, "resolution_time"] = resolved_median

        unresolved_nulls = int((df["resolved"] == False & df["resolution_time"].isna()).sum())
        print(f"[handle_missing_structured] 'resolution_time': Imputed {n_imputed_res:,} missing resolved tickets "
              f"with median ({resolved_median:.2f} hrs). Left {unresolved_nulls:,} unresolved tickets as NaN.")

    return df


# ---------------------------------------------------------------------------
# 6. DEDUPLICATION
# ---------------------------------------------------------------------------

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify and remove exact duplicate tickets.
    
    Decisions & Rationale:
    - Exclude ticket_id when checking duplicates: Synthetic or upstream system
      glitches often assign new IDs to identical form resubmissions.
    - Keep the first occurrence, drop duplicates.
    - Document count of removed duplicates.
    """
    df = df.copy()
    initial_count = len(df)

    feature_cols = [c for c in df.columns if c not in ("ticket_id", "is_near_duplicate")]
    df = df.drop_duplicates(subset=feature_cols, keep="first")
    removed = initial_count - len(df)

    print(f"[remove_duplicates] Removed {removed:,} exact duplicate rows "
          f"({(removed / initial_count) * 100:.2f}%). Kept first occurrence.")
    return df


# ---------------------------------------------------------------------------
# 7. NEAR-DUPLICATE FLAGGING
# ---------------------------------------------------------------------------

def flag_near_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag potential near-duplicate tickets without deleting them.
    
    Decisions & Rationale:
    - Do NOT remove near-duplicates at this stage: Deleting tickets that share
      similar customer wording risks removing legitimate repetitive complaints
      or valid multiple queries from the same user.
    - Flagging allows downstream modules (Module 4 split design & Module 5 similarity)
      to evaluate the impact of near-duplicate leakage on generalization.
    - Heuristic: Identifies customers who submitted multiple tickets where text
      lengths are within 15 characters of each other.
    """
    df = df.copy()
    df["is_near_duplicate"] = False

    # Heuristic: group by customer_id and check for close text lengths
    customer_counts = df["customer_id"].value_counts()
    multi_customers = customer_counts[customer_counts > 1].index

    flagged_indices = set()
    for cid in multi_customers:
        cust_rows = df[df["customer_id"] == cid]
        texts = cust_rows["ticket_text"].tolist()
        indices = cust_rows.index.tolist()
        lengths = [len(t) for t in texts]

        for i in range(len(lengths)):
            for j in range(i + 1, len(lengths)):
                if abs(lengths[i] - lengths[j]) <= 15:
                    flagged_indices.add(indices[i])
                    flagged_indices.add(indices[j])

    df.loc[list(flagged_indices), "is_near_duplicate"] = True
    print(f"[flag_near_duplicates] Flagged {len(flagged_indices):,} near-duplicate candidates "
          f"({(len(flagged_indices) / len(df)) * 100:.2f}% of cleaned data). Flagged only, not dropped.")
    return df


# ---------------------------------------------------------------------------
# 8. SAVE CLEAN DATA
# ---------------------------------------------------------------------------

def save_clean_data(df: pd.DataFrame, path: str | Path = "data/processed/tickets_clean.csv") -> None:
    """Save cleaned DataFrame to CSV, ensuring target directory exists."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\n[save_clean_data] Cleaned dataset successfully saved to: {out_path.resolve()}")
    print(f"[save_clean_data] Final Clean Shape: {df.shape[0]:,} rows x {df.shape[1]} columns\n")


# ---------------------------------------------------------------------------
# 9. MAIN PIPELINE EXECUTION
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Module 1: Support Ticket Data Cleaning & Preparation")
    parser.add_argument("--input",  type=str, default="data/raw/tickets.csv", help="Input raw CSV path")
    parser.add_argument("--output", type=str, default="data/processed/tickets_clean.csv", help="Output clean CSV path")
    args = parser.parse_args()

    print("=" * 65)
    print(" " * 15 + "MODULE 1: DATA CLEANING PIPELINE")
    print("=" * 65)

    # 1. Load raw data
    raw_df = load_raw_data(args.input)

    # 2. Audit quality
    audit_report = audit_data_quality(raw_df)

    # 3. Clean text (drop nulls, normalize)
    df_step1 = clean_ticket_text(raw_df)

    # 4. Fix timestamps & extract temporal signals
    df_step2 = fix_timestamps(df_step1)

    # 5. Handle missing structured features & cap outliers
    df_step3 = handle_missing_structured(df_step2)

    # 6. Remove exact duplicates
    df_step4 = remove_duplicates(df_step3)

    # 7. Flag near-duplicates (flag only, keep data)
    clean_df = flag_near_duplicates(df_step4)

    # 8. Save clean dataset
    save_clean_data(clean_df, args.output)

    # 9. Pipeline summary before vs after
    print("=" * 65)
    print(" " * 22 + "CLEANING PIPELINE SUMMARY")
    print("=" * 65)
    print(f"{'Metric':<35} {'Before (Raw)':<15} {'After (Cleaned)':<15}")
    print("-" * 65)
    print(f"{'Row Count':<35} {audit_report['total_rows']:<15,} {len(clean_df):<15,}")
    print(f"{'Columns':<35} {raw_df.shape[1]:<15} {clean_df.shape[1]:<15}")
    print(f"{'Missing ticket_text':<35} {audit_report['missing_counts'].get('ticket_text', 0):<15,} {clean_df['ticket_text'].isna().sum():<15}")
    print(f"{'Exact Duplicate Rows':<35} {audit_report['exact_duplicates']:<15,} 0")
    print(f"{'Negative previous_tickets':<35} {audit_report['negative_previous_tickets']:<15,} {(clean_df['previous_tickets'] < 0).sum():<15}")
    print(f"{'Near-duplicate Flagged':<35} {'N/A':<15} {clean_df['is_near_duplicate'].sum():<15,}")
    print("=" * 65)


if __name__ == "__main__":
    main()
