"""Import the synthetic CSV tables into the database (raw CSVs are never modified).

Usage: python -m backend.data_import [--drop]
Validates each table first (backend.data_validation); refuses to import a table that fails.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import delete, insert

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app import db  # noqa: E402
from backend.data_validation import validate_frame  # noqa: E402
from backend.database_setup import create_all  # noqa: E402
from ml.common.config import SYN  # noqa: E402

DATE_COLS = {"inventory": ["week_start"], "procurement": ["order_date", "promised_date"],
             "shipments": ["ship_date", "promised_date", "actual_delivery_date"],
             "disruptions": ["start_date", "end_date", "reported_date"], "external_risks": ["month"], "production": ["week_start"]}


def import_all(drop=False, syn_dir=SYN, verbose=True):
    eng = create_all(drop=drop)
    frames = {t: pd.read_csv(syn_dir / f"{t}.csv") for t in db.DATA_TABLES}
    counts = {}
    with eng.begin() as conn:
        for t in reversed(db.DATA_TABLES):
            conn.execute(delete(db.metadata.tables[t]))
        for t in db.DATA_TABLES:
            df = frames[t]
            rep = validate_frame(df, t, frames)
            if rep["status"] != "pass":
                raise ValueError(f"validation failed for {t}: {rep['errors']}")
            df = df.copy()
            for c in DATE_COLS.get(t, []):
                df[c] = pd.to_datetime(df[c]).dt.date
            cols = [c.name for c in db.metadata.tables[t].columns]
            df = df[[c for c in df.columns if c in cols]]
            df = df.astype(object).where(pd.notna(df), None)
            recs = df.to_dict(orient="records")
            for i in range(0, len(recs), 2000):
                conn.execute(insert(db.metadata.tables[t]), recs[i:i + 2000])
            counts[t] = len(recs)
            if verbose:
                print(f"{t:24s} {len(recs):6d} rows")
    return counts


if __name__ == "__main__":
    import_all(drop="--drop" in sys.argv)
