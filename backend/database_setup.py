"""Create the database schema and write database_schema.sql (MySQL dialect).

Usage:
    python -m backend.database_setup            # create tables on DATABASE_URL (default SQLite)
    python -m backend.database_setup --ddl-only # only regenerate backend/database_schema.sql
"""
from __future__ import annotations

import sys
from pathlib import Path

from sqlalchemy.dialects import mysql
from sqlalchemy.schema import CreateIndex, CreateTable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.db import database_url, get_engine, metadata  # noqa: E402


def write_ddl(path: Path = ROOT / "backend" / "database_schema.sql"):
    lines = ["-- Supply Chain Black Box - MySQL 8 schema (generated from backend/app/db.py)",
             "CREATE DATABASE IF NOT EXISTS supply_chain_black_box CHARACTER SET utf8mb4;", "USE supply_chain_black_box;", ""]
    for t in metadata.sorted_tables:
        lines.append(str(CreateTable(t).compile(dialect=mysql.dialect())).strip() + " ENGINE=InnoDB;\n")
        for ix in t.indexes:
            lines.append(str(CreateIndex(ix).compile(dialect=mysql.dialect())).strip() + ";")
        lines.append("")
    path.write_text("\n".join(lines))
    return path


def create_all(drop=False):
    eng = get_engine()
    if drop:
        metadata.drop_all(eng)
    metadata.create_all(eng)
    return eng


if __name__ == "__main__":
    p = write_ddl()
    print("DDL written to", p)
    if "--ddl-only" not in sys.argv:
        create_all(drop="--drop" in sys.argv)
        print("tables created on", database_url().split("@")[-1])
