"""
pipeline/ingestion/loader.py

Loads the five raw Parquet files and runs referential-integrity checks before
any transformation happens.  Any failed check raises IntegrityError so the
pipeline halts regardless of the Python optimization level (-O).
"""

from pathlib import Path
import pandas as pd


class IntegrityError(ValueError):
    """Raised when a referential-integrity check fails."""


RAW_FILES = {
    "applications":  "credit_applications.parquet",
    "decisions":     "credit_decisions.parquet",
    "ops":           "underwriting_ops.parquet",
    "performance":   "credit_performance.parquet",
    "behavior":      "app_behavior_features.parquet",
}


def load_raw(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Read every raw table and return a dict keyed by logical name."""
    tables = {}
    for name, filename in RAW_FILES.items():
        path = data_dir / filename
        tables[name] = pd.read_parquet(path)
        print(f"  loaded {name:12s}  {len(tables[name]):>10,} rows  ({path.name})")
    return tables


def run_integrity_checks(tables: dict[str, pd.DataFrame]) -> None:
    """Check key uniqueness and referential integrity across all tables.

    Raises IntegrityError on the first failed check so the pipeline never
    proceeds on invalid input, even when running under python -O.
    """
    apps  = tables["applications"]
    dec   = tables["decisions"]
    ops   = tables["ops"]
    beh   = tables["behavior"]
    perf  = tables["performance"]

    checks = [
        (
            "applications.application_id unique",
            apps.application_id.is_unique,
        ),
        (
            "one application per customer in applications",
            apps.customer_id.is_unique,
        ),
        (
            "decisions 1:1 with applications",
            dec.application_id.is_unique
            and set(dec.application_id) == set(apps.application_id),
        ),
        (
            "underwriting_ops 1:1 with applications",
            ops.application_id.is_unique
            and set(ops.application_id) == set(apps.application_id),
        ),
        (
            "behavior features 1:1 with customers",
            beh.customer_id.is_unique
            and set(beh.customer_id) == set(apps.customer_id),
        ),
        (
            "performance keyed (account_id, observation_month)",
            not perf.duplicated(["account_id", "observation_month"]).any(),
        ),
        (
            "account_id <-> customer_id is 1:1 in performance",
            perf.groupby("customer_id").account_id.nunique().eq(1).all(),
        ),
        (
            "performance covers exactly the approved population",
            set(perf.customer_id)
            == set(dec.loc[dec.decision == "approved", "customer_id"]),
        ),
    ]

    results = []
    for description, passed in checks:
        status = "PASS" if passed else "FAIL"
        results.append({"check": description, "result": status})
        print(f"  [{status}] {description}")
        if not passed:
            raise IntegrityError(f"Integrity check failed: {description}")

    print(f"  integrity: {sum(r['result'] == 'PASS' for r in results)}/{len(results)} checks passed")
