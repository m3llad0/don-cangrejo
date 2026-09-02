"""
pipeline/execution/run_pipeline.py

End-to-end pipeline for the thin-file credit underwriting analysis.

Stages
------
1. INGEST    — load raw Parquet files, run integrity checks.
2. TRANSFORM — join all tables into one application-level base; engineer features
               and segment flags (is_train, is_eval, is_apply).
3. MODEL     — compare logistic regression, random forest, and XGBoost on the
               thin-file pilot-holdout population (the only credible counterfactual).
               Random Forest is saved as the production model and reloaded for scoring.
4. PERSIST   — write application_base, features_clean, model_scores, model_comparison,
               and a JSON manifest with input checksums and output provenance.

Features (feature_cols_v2 from the notebook):
  sessions_30d, avg_session_minutes, deposits_90d_count, balance_avg_90d_mxn,
  payroll_deposit_flag, p2p_inbound_90d_count, bill_payments_90d_count,
  days_since_last_login, secured_card_utilization_pct, declared_income_mxn,
  age, months_with_nu

Usage
-----
    # Inference — load saved RF, score thin-file applicants
    python pipeline/execution/run_pipeline.py

    # Retrain — fit all candidates, overwrite saved RF
    python pipeline/execution/run_pipeline.py --retrain

    # Custom bands (defaults: low=0.40, high=0.70)
    python pipeline/execution/run_pipeline.py --low 0.35 --high 0.65
"""

import argparse
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline.ingestion.loader import IntegrityError, load_raw, run_integrity_checks
from pipeline.persistence.writer import (
    checksum_inputs,
    load_model,
    save_model,
    write_manifest,
    write_table,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_PATH = REPO_ROOT / "models" / "random_forest.joblib"

# 3-band segmentation (from notebook):
#   pd_hat < LOW   → auto_approve
#   pd_hat > HIGH  → auto_reject
#   in between     → manual review
DEFAULT_LOW  = 0.40
DEFAULT_HIGH = 0.70

# feature_cols_v2 from the notebook — drops existing_nu_customer and
# deposits_90d_amount_mxn (high correlation), keeps secured_card_utilization_pct
FEATURES = [
    "sessions_30d",
    "avg_session_minutes",
    "deposits_90d_count",
    "balance_avg_90d_mxn",
    "payroll_deposit_flag",
    "p2p_inbound_90d_count",
    "bill_payments_90d_count",
    "days_since_last_login",
    "secured_card_utilization_pct",
    "declared_income_mxn",
    "age",
    "months_with_nu",
]

CANDIDATES = {
    "logistic_regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2_000, random_state=42),
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    ),
    "xgboost": XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=None,   # set at fit time from label distribution
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
    ),
}


# ---------------------------------------------------------------------------
# Stage 2: transform
# ---------------------------------------------------------------------------

def build_outcome(perf: pd.DataFrame) -> pd.DataFrame:
    """Collapse the monthly performance panel to one row per customer."""
    return (
        perf.groupby("customer_id")
        .agg(
            default_12m=("default_flag_12m", "max"),
            months_observed=("months_on_book", "max"),
            max_dpd=("days_past_due", "max"),
            avg_balance_mxn=("balance_mxn", "mean"),
            credit_limit_mxn=("credit_limit_mxn", "mean"),
        )
        .reset_index()
    )


def join_tables(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Star-join all application-level tables. Fails loudly on fan-out."""
    apps, dec, ops, beh, perf = (
        tables["applications"],
        tables["decisions"],
        tables["ops"],
        tables["behavior"],
        tables["performance"],
    )
    outcome = build_outcome(perf)
    df = (
        apps.merge(dec,     on=["application_id", "customer_id"], how="inner", validate="1:1")
            .merge(ops,     on=["application_id", "customer_id"], how="inner", validate="1:1")
            .merge(beh,     on="customer_id",                     how="inner", validate="1:1")
            .merge(outcome, on="customer_id",                     how="left",  validate="1:1")
    )
    if len(df) != len(apps):
        raise IntegrityError(
            f"Join changed row count: expected {len(apps):,}, got {len(df):,}"
        )
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns and population-segment flags."""
    df = df.copy()
    df["thin_file"] = ~df["bureau_hit"]
    df["approved"]  = df["decision"].eq("approved")
    df["app_month"] = df["application_ts"].dt.to_period("M")

    # secured card: treat null utilisation as "no card"
    df["has_secured_card"] = df["secured_card_utilization_pct"].notna().astype(int)
    df["secured_card_utilization_pct"] = df["secured_card_utilization_pct"].fillna(0.0)

    # numeric label (NaN preserved for unapproved)
    df["default_12m_int"] = (
        df["default_12m"]
        .map({True: 1, False: 0})
        .astype("Int64")
    )

    # is_train: bureau-hit accounts approved by regular policy with observed label
    df["is_train"] = (
        df["bureau_hit"]
        & df["approved"]
        & df["approval_channel"].eq("policy")
        & df["default_12m"].notna()
    ).astype(int)

    # is_eval: pilot holdout with observed label (the counterfactual window)
    df["is_eval"] = (
        df["approval_channel"].eq("pilot_holdout")
        & df["default_12m"].notna()
    ).astype(int)

    # is_apply: thin-file applicants the scorecard will rank
    df["is_apply"] = df["thin_file"].astype(int)

    return df


# ---------------------------------------------------------------------------
# Stage 3: model
# ---------------------------------------------------------------------------

def train(df: pd.DataFrame, model_path: Path = MODEL_PATH) -> tuple[pd.DataFrame, dict]:
    """
    Train Random Forest on is_train (bureau-hit, policy-approved customers with known
    outcomes — ~272k rows) matching the notebook's approach.

    Evaluate all CANDIDATES on is_eval (pilot holdout — the only thin-file population
    with observed outcomes) to report honest OOF AUC on the target population.

    RF is always saved as the production model.
    Returns (df_with_oof_cols, comparison_dict).
    """
    train_pop = df[df["is_train"] == 1].copy()
    eval_pop  = df[df["is_eval"]  == 1].copy()

    X_train = train_pop[FEATURES].astype(float).values
    y_train = train_pop["default_12m_int"].astype(int).values

    X_eval  = eval_pop[FEATURES].astype(float).values
    y_eval  = eval_pop["default_12m_int"].astype(int).values

    neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())
    CANDIDATES["xgboost"].set_params(scale_pos_weight=neg / pos)

    comparison = {}
    df = df.copy()

    print(f"  train pop: {len(train_pop):,} rows (is_train)  |  "
          f"eval pop: {len(eval_pop):,} rows (is_eval, pilot holdout)")
    print("  --- candidate AUC on pilot holdout ---")
    for name, model in CANDIDATES.items():
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_eval)[:, 1]
        auc  = roc_auc_score(y_eval, prob)
        comparison[name] = {"auc_oof": round(auc, 4)}
        print(f"  {name:<22s}  AUC = {auc:.4f}")

        df[f"pd_hat_{name}_oof"] = np.nan
        df.loc[eval_pop.index, f"pd_hat_{name}_oof"] = prob

    comparison["production_model"] = "random_forest"

    # RF was already fitted in the candidate loop — reuse it directly
    save_model(CANDIDATES["random_forest"], model_path)

    return df, comparison


def score_thin_file(df: pd.DataFrame, model_path: Path = MODEL_PATH) -> pd.DataFrame:
    """
    Load the saved Random Forest and score all thin-file applicants.
    Adds column `pd_hat` (NaN for bureau-hit rows).
    """
    rf = load_model(model_path)

    thin_all = df[df["thin_file"]].copy()
    X_apply  = thin_all[FEATURES].astype(float).values

    df = df.copy()
    df["pd_hat"] = np.nan
    df.loc[thin_all.index, "pd_hat"] = rf.predict_proba(X_apply)[:, 1]

    return df


def apply_bands(df: pd.DataFrame, low: float = DEFAULT_LOW, high: float = DEFAULT_HIGH) -> pd.DataFrame:
    """
    3-band segmentation matching the notebook:
      pd_hat < low   → banda = 'auto_approve'
      pd_hat > high  → banda = 'auto_reject'
      in between     → banda = 'manual'

    Only applied to thin-file rows; bureau-hit rows get NaN.
    """
    df = df.copy()
    df["banda"] = None  # object dtype so string values can be assigned
    thin_mask = df["thin_file"]
    scores = df.loc[thin_mask, "pd_hat"]
    df.loc[thin_mask, "banda"] = np.select(
        [scores < low, scores > high],
        ["auto_approve", "auto_reject"],
        default="manual",
    )
    return df


# ---------------------------------------------------------------------------
# Stage 4: build output DataFrames
# ---------------------------------------------------------------------------

def build_application_base(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "application_id", "customer_id", "application_ts", "requested_product",
        "bureau_hit", "bureau_score", "declared_income_mxn", "employment_type",
        "age", "state", "existing_nu_customer", "months_with_nu",
        "decision", "approval_channel", "approved_limit_mxn", "policy_reason",
        "manual_review_flag", "documents_requested", "decision_latency_hours",
        "analyst_review_minutes", "reworked_flag", "abandoned_before_decision",
        "thin_file", "approved", "app_month",
        "is_train", "is_eval", "is_apply",
        "default_12m", "default_12m_int", "months_observed", "max_dpd",
        "avg_balance_mxn", "credit_limit_mxn",
    ]
    return df[[c for c in cols if c in df.columns]].copy()


def build_features_clean(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "application_id", "customer_id",
        "is_train", "is_eval", "is_apply",
        "default_12m_int",
        "has_secured_card",
        *FEATURES,
    ]
    return df[[c for c in cols if c in df.columns]].copy()


def build_model_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Scores for thin-file population: pd_hat, 3-band decision, and per-model holdout probs."""
    thin = df[df["thin_file"]].copy()
    oof_cols = [c for c in thin.columns if c.startswith("pd_hat_") and c.endswith("_oof")]
    cols = [
        "application_id", "customer_id",
        "is_eval", "is_apply",
        "pd_hat",
        "banda",
        *oof_cols,
        # ground-truth label — NaN for rows without observed outcome
        "default_12m_int",
        # pre-decision features kept for segment analysis
        "payroll_deposit_flag",
        "balance_avg_90d_mxn",
        "declared_income_mxn",
    ]
    return thin[[c for c in cols if c in thin.columns]].copy()


def build_model_comparison(comparison: dict) -> pd.DataFrame:
    production = comparison.get("production_model")
    rows = [
        {"model": name, "auc_oof": vals["auc_oof"], "production": name == production}
        for name, vals in comparison.items()
        if isinstance(vals, dict) and "auc_oof" in vals
    ]
    return pd.DataFrame(rows).sort_values("auc_oof", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the thin-file underwriting pipeline.")
    p.add_argument("--data-dir",   type=Path, default=REPO_ROOT / "data",
                   help="Directory containing raw Parquet files (default: data/)")
    p.add_argument("--output-dir", type=Path, default=REPO_ROOT / "output",
                   help="Directory for processed outputs (default: output/)")
    p.add_argument("--model-path", type=Path, default=MODEL_PATH,
                   help=f"Path to saved Random Forest model (default: {MODEL_PATH})")
    p.add_argument("--low",  type=float, default=DEFAULT_LOW,
                   help=f"Auto-approve band: pd_hat < low (default: {DEFAULT_LOW})")
    p.add_argument("--high", type=float, default=DEFAULT_HIGH,
                   help=f"Auto-reject band: pd_hat > high (default: {DEFAULT_HIGH})")
    p.add_argument("--retrain", action="store_true",
                   help="Retrain all candidates and overwrite the saved model. "
                        "Without this flag the pipeline loads the existing model.")
    return p.parse_args()


def run(
    data_dir: Path,
    output_dir: Path,
    model_path: Path,
    low: float,
    high: float,
    retrain: bool,
) -> None:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:6]
    mode = "RETRAIN" if retrain else "INFERENCE"
    print(f"\n{'='*60}")
    print(f"Pipeline run  {run_id}  [{mode}]  bands=(<{low} auto_approve | >{high} auto_reject)")
    print(f"{'='*60}")

    # ---- 1. Ingest --------------------------------------------------------
    print("\n[1/4] INGEST")
    tables = load_raw(data_dir)
    run_integrity_checks(tables)

    input_checksums = checksum_inputs(
        data_dir,
        [
            "credit_applications.parquet",
            "credit_decisions.parquet",
            "underwriting_ops.parquet",
            "credit_performance.parquet",
            "app_behavior_features.parquet",
        ],
    )

    # ---- 2. Transform -----------------------------------------------------
    print("\n[2/4] TRANSFORM")
    df = join_tables(tables)
    df = engineer_features(df)
    print(f"  base table: {len(df):,} applications x {df.shape[1]} columns")
    seg = df.groupby(["is_train", "is_eval", "is_apply"]).size().reset_index(name="n")
    for _, row in seg.iterrows():
        print(f"    is_train={row.is_train} is_eval={row.is_eval} "
              f"is_apply={row.is_apply}  n={row.n:,}")

    # ---- 3. Model ---------------------------------------------------------
    print(f"\n[3/4] MODEL  (features={len(FEATURES)})")
    comparison = None

    if retrain:
        df, comparison = train(df, model_path)
    else:
        print(f"  loading saved model from {model_path} …")

    df = score_thin_file(df, model_path)
    df = apply_bands(df, low, high)

    n_thin         = int(df["thin_file"].sum())
    n_auto_approve = int(df["banda"].eq("auto_approve").sum())
    n_manual       = int(df["banda"].eq("manual").sum())
    n_auto_reject  = int(df["banda"].eq("auto_reject").sum())
    print(f"  auto_approve (<{low}): {n_auto_approve:,} ({n_auto_approve/n_thin:.1%})  |  "
          f"manual: {n_manual:,} ({n_manual/n_thin:.1%})  |  "
          f"auto_reject (>{high}): {n_auto_reject:,} ({n_auto_reject/n_thin:.1%})")

    # ---- 4. Persist -------------------------------------------------------
    print("\n[4/4] PERSIST")
    out = output_dir / run_id
    out.mkdir(parents=True, exist_ok=True)

    records = []
    records.append(write_table(build_application_base(df), out / "application_base.parquet"))
    records.append(write_table(build_features_clean(df),   out / "features_clean.parquet"))
    records.append(write_table(build_model_scores(df),     out / "model_scores.parquet"))
    if comparison is not None:
        records.append(write_table(build_model_comparison(comparison),
                                   out / "model_comparison.parquet"))

    meta = {
        "mode": mode,
        "production_model": "random_forest",
        "model_path": str(model_path),
        "bands": {"auto_approve_below": low, "auto_reject_above": high},
        "features": FEATURES,
    }
    if comparison is not None:
        meta["auc_oof_by_model"] = {
            k: v["auc_oof"] for k, v in comparison.items()
            if isinstance(v, dict) and "auc_oof" in v
        }

    manifest_path = write_manifest(output_dir, run_id, input_checksums, records, meta=meta)

    print(f"\nDone.  Outputs -> {out}")
    print(f"       Manifest -> {manifest_path}")


if __name__ == "__main__":
    args = parse_args()
    run(args.data_dir, args.output_dir, args.model_path, args.low, args.high, args.retrain)
