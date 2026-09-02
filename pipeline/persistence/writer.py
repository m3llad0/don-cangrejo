"""
pipeline/persistence/writer.py

Writes processed DataFrames as Parquet files under output_dir and records a
JSON run manifest so every execution is auditable:
  - what ran, when, with which input checksums
  - which files were written and how many rows each contains
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def write_table(df: pd.DataFrame, path: Path) -> dict:
    """Persist one DataFrame as Parquet and return a provenance record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    sha = _sha256(path)
    record = {
        "file": str(path),
        "rows": len(df),
        "columns": list(df.columns),
        "sha256": sha,
    }
    print(f"  wrote {path.name:40s}  {len(df):>10,} rows  sha256={sha[:12]}…")
    return record


def write_manifest(
    output_dir: Path,
    run_id: str,
    input_checksums: dict[str, str],
    output_records: list[dict],
    meta: dict | None = None,
) -> Path:
    """Write a JSON manifest capturing the full provenance of one pipeline run.

    File paths inside the manifest are stored relative to output_dir so the
    manifest remains valid after the repo is cloned to a different machine.
    """
    portable_records = []
    for rec in output_records:
        r = dict(rec)
        try:
            r["file"] = str(Path(rec["file"]).relative_to(output_dir))
        except ValueError:
            pass
        portable_records.append(r)

    manifest = {
        "run_id": run_id,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": input_checksums,
        "outputs": portable_records,
        "meta": meta or {},
    }
    path = output_dir / f"manifest_{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))
    print(f"  manifest -> {path}")
    return path


def checksum_inputs(data_dir: Path, filenames: list[str]) -> dict[str, str]:
    """SHA-256 every raw input file so the manifest records what data was used."""
    return {name: _sha256(data_dir / name) for name in filenames}


def save_model(model, path: Path) -> str:
    """Persist a fitted sklearn-compatible model with joblib. Returns SHA-256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    sha = _sha256(path)
    print(f"  saved  {path.name:40s}  sha256={sha[:12]}…")
    return sha


def load_model(path: Path):
    """Load a joblib model from disk. Raises FileNotFoundError if absent."""
    if not path.exists():
        raise FileNotFoundError(f"No saved model at {path}. Run with --retrain to train one.")
    model = joblib.load(path)
    print(f"  loaded {path.name:40s}  (skip retraining)")
    return model
