# Project Status

Legend: **P** planned · **I** implemented · **T** tested · **V** experimentally validated · **D** demonstrated (in running app / screenshots)

| Phase | Item | Status |
|---|---|---|
| 1 | Document review and requirements extraction | I — `documentation/research/REQUIREMENTS_AND_FEEDBACK_MATRIX.md` |
| 2 | Proposal, literature review, dataset verification | I — `PROPOSAL_AND_LITERATURE_REVIEW.md`, `DATASET_CATALOG.md` |
| 3 | Data acquisition and engineering (synthetic generator, real processing, validation, MySQL) | I T — 196 data checks pass; MySQL import tested |
| 4 | Exploratory analysis | I — profiles in the data dictionary, dashboard, notebook |
| 5 | Supervised ML (Exp 1, Exp 3) | I T V |
| 6 | Calibration and decision engine | I T V D |
| 7 | GNN and link prediction (Exp 2A/2B) | I V (negative result reported) |
| 8 | Cascading simulation (Exp 4) | I T V D |
| 9 | Nine-scenario evaluation | I T V D — 500 CRN replications, 61/61 verification checks |
| 10 | Alternative-supplier optimisation | I T V D |
| 11 | Flask backend (31 routes) | I T D |
| 12 | React frontend (12 pages) | I T D — Playwright, zero console errors |
| 13 | AI assistant | I T D (rule-based router); Gemini path **I only — not executed** (no key or network) |
| 14 | Testing and validation | T — 35/35 pytest on SQLite; 20/20 API tests on MySQL |
| 15 | IEEE report and documentation | I — 5-page PDF, Overleaf package |
| 16 | Presentation and viva | I — 22-slide PPTX, 35 viva Q&As |
| 17 | Rubric audit and packaging | I — `RUBRIC_AUDIT.md`, deliverables ZIP |

## Dataset status
* DataCo (real): downloaded, hash-verified, processed (172,765 rows). Raw file not committed (size/licence caution); `download_real.sh` restores it.
* SupplyGraph (real): committed under LGPL-2.1 with its licence; one duplicated node removed and documented.
* Synthetic: 14 tables, seed 42, regenerable.

## Model status
`models/`: four Exp 1 classifiers (+ calibrators), five Exp 2A link models (`.pt`), Exp 3 XGBoost classifier and regressor.

## Errors encountered and resolved
PyPI CPU torch index blocked → CUDA wheels from PyPI; Debian `blinker` conflict; PuLP 4.x API break → pinned 2.9.0; pickled class in `__main__`;
SQLite datetime binding; MySQL reserved words; SPA deep links; NaN in JSON; simulator inventory cap and hard exclusion (modelling defects);
recovery-time censoring (horizon 120 → 180 days).

## Known limitations
Supplier-level results use synthetic data; economics are illustrative; GNNs did not help; the static MILP over-hedges for mild shocks; the Gemini path is
untested; author details are placeholders; AI-generated content exceeds the course's 20 % guideline until the team rewrites it.

## Deadlines (template)
Second review: 20 Nov 2026 · Final submission: 4 Dec 2026 (11:59 PM AoE).
