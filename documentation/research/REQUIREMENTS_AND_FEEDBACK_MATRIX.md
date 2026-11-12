# Requirements and Feedback Matrix

Sources: the official AMLBT template (primary authority), the AgriGuard senior report (style reference only), and two first-review
feedback reports. **Note:** the file provided as "39B" is actually the **26B-V1** review (project "Universal Dataset Intelligence", 20.2/30).
The second report is **24B-V1** (hotel overbooking, 11.8/30).

## 1. Official template requirements

| Requirement | Evidence |
|---|---|
| IEEE conference format (Overleaf IEEEtran) | `documentation/final_report/main.tex` (IEEEtran conference class) |
| ≤ 6 pages including references | compiled PDF = **5 pages** (checked with `pdfinfo`) |
| Sections: introduction, scope/objectives, related work, methodology (data, models, metrics), results, discussion, reproducibility, AI declaration, conclusion | all present; template instructions, prompts, guidelines and example tables removed |
| IEEE numbered citations; 5–15 relevant references | 17 references; DOIs verified by web search where available |
| GitHub link plus reproduction steps | Section VI + `README.md` + `run_all.sh` |
| AI-use declaration, < 20 % AI content | Section VI + `AI_USE_DECLARATION.md`; non-compliance of the current draft stated explicitly, and the team must act on it |
| Deadlines: 2nd review 20 Nov 2026, final 4 Dec 2026 (AoE) | noted in `PROJECT_STATUS.md` |
| Private copy of Overleaf template | `documentation/final_report/overleaf_package.zip` is ready to upload into a *private copy* |

## 2. Lessons from 26B-V1

| Weakness | Response |
|---|---|
| Seven pages exceeded the six-page limit | 5 pages |
| Conditional AI declaration ("may have been used") | precise tool, tasks and responsibility statement |
| Defined tests ≠ executed tests | `documentation/testing/TEST_REPORT.md` lists executed results (35 passed, plus a MySQL run) |
| Small generated datasets; real data needed | two real datasets (DataCo 172,765 rows; SupplyGraph) in dedicated experiments |
| Held-out evaluation of routing / numerical correctness | 7 parametrised assistant-routing tests; verification script recomputes metrics |

## 3. Lessons from 24B-V1 (all 15 points)

| # | Lesson | Evidence |
|---:|---|---|
| 1 | Relevance alone is not achievement | trained models, logged runs, working application, screenshots |
| 2 | Link predictions to an executable decision | `ml/decision/policy.py` and `engine.py`, `/api/decision/evaluate` |
| 3 | Quantitative model and calibration comparisons | Tables I–IV; reliability figure |
| 4 | Exact prediction timestamp | 00:00 on the first of each month; information cut-off stated |
| 5 | Dataset source and target documented | `DATASET_CATALOG.md`, `DATA_PROVENANCE.md`, Eq. (1) |
| 6 | Leakage-safe splits | chronological split with embargo; `tests/test_leakage.py` |
| 7 | Calibration method explained | Platt/isotonic fitted on the calibration split; CV selection |
| 8 | Executable and reproducible decision rule | argmin expected cost; cost-derived thresholds |
| 9 | All predefined policy scenarios evaluated | all nine scenarios × 13 metrics × 500 replications |
| 10 | Explicit financial assumptions | `ECONOMIC_ASSUMPTIONS.md`, `scenario_assumptions.json` |
| 11 | Measurable business outcomes | realised-cost comparison (Exp 1); paired CRN savings (scenarios) |
| 12 | No placeholder figures or cartoons | every figure is generated from data or code |
| 13 | No template instruction sections | removed |
| 14 | A URL alone is not reproducibility | `run_all.sh`, pinned hashes, seeds, `verify_results.py` |
| 15 | Dataset, parameter, code and execution documentation | data dictionary, assumptions, `run.json` per experiment |

## 4. Additional review checks (dataset provenance, real vs generated data, consistent counts, metrics)

| Check | Evidence |
|---|---|
| Real vs generated data separated | labels in data, UI and paper |
| Conflicting dataset sizes / split counts | verification checks 1–4, 23–25 |
| Confusion matrices reconcile | verification checks (every model × calibration) |
| Unsupported real-time claims | none made |
| Irrelevant references | all 17 references are cited in the text |
