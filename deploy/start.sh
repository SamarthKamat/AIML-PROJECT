#!/usr/bin/env bash
# Container entrypoint: prepare the database, warm caches, start gunicorn.
# - No DATABASE_URL: bundled SQLite is rebuilt from the versioned CSVs on every start (ephemeral).
# - DATABASE_URL set (e.g. MySQL): tables are created and seeded only if empty; data persists.
set -euo pipefail
cd /app
if [ -z "${DATABASE_URL:-}" ]; then
  python -m backend.database_setup --drop
  python -m backend.data_import
else
  python -m backend.database_setup
  python -c "
from sqlalchemy import text
from backend.app.db import get_engine
with get_engine().connect() as c:
    n = c.execute(text('SELECT COUNT(*) FROM suppliers')).scalar()
raise SystemExit(0 if n else 3)" || python -m backend.data_import
fi
python -m deploy.warmup || echo "warm-up skipped"
exec gunicorn "backend.run:app" --bind "0.0.0.0:${PORT:-10000}" --workers "${WEB_CONCURRENCY:-1}" --threads 4 --timeout 300 --access-logfile -
