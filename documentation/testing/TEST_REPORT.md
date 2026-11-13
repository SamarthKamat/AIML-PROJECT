# Test Report (executed, not just defined)

Date of execution: 2026-11-12, Python 3.13.18, Linux container.

## Automated tests (`python -m pytest tests -q`)

| Suite | Tests | Result | What it checks |
|---|---:|---|---|
| `test_leakage.py` | 4 | pass | changing future disruptions/shipments or removing hidden edges leaves features at t unchanged; labels use only the (t, t+30] window; splits are embargoed |
| `test_metrics.py` | 2 | pass | P/R/specificity/F1 equal formulas from TP/FP/TN/FN; ECE on known cases |
| `test_policy.py` | 3 | pass | zero exposure → monitoring; actions monotone in probability; break-even consistent with decision |
| `test_simulation_optimisation.py` | 4 | pass | no shock → no loss; unmet demand monotone in severity; CRN reproducibility; MILP respects capacity, risk limit, exclusions |
| `test_validation.py` | 2 | pass | injected duplicates, bad dates, negatives, quantity violations detected; all synthetic tables pass |
| `test_api.py` | 20 | pass | auth (401/400/409), dashboard totals, suppliers, graph build/network, risk + SHAP, link prediction, delivery delay, simulation (incl. validation errors), nine scenarios, recovery MILP vs rule, decision engine + audit trail, 7 assistant intents, upload validation and history |
| **Total (SQLite)** | **35** | **35 passed** | |
| `test_api.py` on **MySQL** (MariaDB 10.11) | 20 | **20 passed** | same API suite with `DATABASE_URL=mysql+pymysql://…` (this run found and fixed a reserved-word bug: `rows`/`columns` → `row_count`/`column_count`) |

## Result verification (`python -m ml.experiments.verify_results`)
61/61 checks passed. See `experiments/VERIFICATION_REPORT.md`.

## Browser test (Playwright, Chromium)
All 12 pages were loaded after registration; the simulator, recovery optimiser, decision engine and two assistant queries were executed.
**Zero page errors and zero console errors.** Screenshots are in `documentation/testing/screenshots/`. Defects found and fixed
during browser testing: SPA deep links returned 404 (Flask static route), and `NaN` in the materials JSON broke parsing (fixed, plus an
app-wide NaN-safe JSON provider).

## Data validation (`python -m backend.data_validation`)
14 tables, 178 checks (176 per-table + 2 cross-table), all pass; report in `documentation/testing/data_quality_report.json`.

## Gemini (executed)
With a real key, `gemini-flash-latest` answered questions by calling the `supplier_failure_impact` and `explain_supplier_risk` tools; numbers matched the tool outputs. Free-tier quota (5 requests/min) and occasional 503s trigger model rotation, then the deterministic router (30 s budget).

## Not tested
* Load/performance testing; no real-time claims are made.
