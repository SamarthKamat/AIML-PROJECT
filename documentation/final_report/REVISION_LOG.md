# Final Revision Log: Supply Chain Black Box IEEE Report

Final PDF: `Supply_Chain_Black_Box_IEEE.pdf` (6 pages, IEEE two-column). It compiles with no errors, no overfull boxes and no undefined
references. All six pages were inspected visually.

## 1. Changes implemented

| # | Area | Change |
|---|---|---|
| 1 | Figure 4 | The reviewed PDF had been built with an older `fig_scenarios.pdf`, so its charts did not match the caption. Figure 4 now shows exactly the requested panels from the same 500-replication data: (a) mean financial impact, (b) unmet demand, (c) minimum service level, (d) cheapest policy. 95% CI bars are on (a) and (b). It uses the requested caption, plus one sentence explaining the error bars. Ablation and sensitivity results remain in Tables VII and VIII. |
| 2 | Cross-references | Section III-G now cites Table VIII instead of "Fig. 4d". Figs. 1, 2 and 4 and Table VI were never cited in the text; citations were added. "Eq. 2" was hard-coded and is now `\eqref`. All 8 tables, 4 figures and labelled equations resolve. |
| 3 | AI disclosure | Rewritten to be concise and accurate (see section 3 below). No percentage is claimed. |
| 4 | Numerical error | "196 validation checks" was wrong; the validation report shows 178 (176 per-table checks across 14 tables + 2 cross-table checks). Corrected in the text, Fig. 1, the slides, `TEST_REPORT.md` and the pipeline diagram. |
| 5 | Numerical clarification | Link-prediction AP 0.227 / P@100 0.32 are 5-seed means. The single list used downstream (seed 0) has AP 0.220 / P@100 0.29. Both are now reported (Table V caption and text), and the A2 AP shows ± sd. |
| 6 | Deployment claim | "MySQL" deployment corrected: the Render deployment uses SQLite; MySQL is supported. |
| 7 | Reproducibility claim | "Both datasets with hash checks" corrected to: both downloaded at pinned commits, DataCo SHA-256 checked (SupplyGraph has no hash check). The report now states that DataCo comes from a pinned public mirror of the Mendeley V5 file. |
| 8 | Abbreviations | Expanded at first use: FMCG, INR, MILP, ML, GNN, OR/MS, AP, ECE, LR, ROC-AUC, PR-AUC, MLP, CI, CM, SHAP. |
| 9 | References | • Platt: book editors added.<br>• Hamilton: unverifiable page range removed; venue given as *Advances in Neural Information Processing Systems 30 (NIPS)*.<br>• Brintrup 2020: DOI added.<br>• Wasi: workshop venue added.<br>• ICLR venue names expanded.<br>• Title capitalisation fixed. |
| 10 | Wording | Shortened without removing any limitation, disclosure or result. |

## 2. Numerical checks completed

- `experiments/REPORT_NUMBER_AUDIT.md`: **87/87 pass.** Every value is recomputed from raw data, saved models or per-replication results,
  not from the paper's macros. Covered:
  - supplier and link counts: 500 suppliers; 120/200/180 by tier; 1,500 links; 461 observed; 1,039 hidden; 69.3% hidden; 501 graph nodes;
  - disruptions: 1,048, of which 782 qualifying;
  - E1 test set: 3,000 supplier-months, 223 positives, prevalence 7.43%;
  - E1 models: LR ROC-AUC 0.666; XGBoost ROC-AUC 0.626 and PR-AUC 0.122; ECE 0.020→0.010; tree ECE 0.037→0.011;
  - link prediction: A2 candidates 8,863, base rate 0.117, AP (5-seed mean and seed-0);
  - exposure gap: 2.1× (Tier-2) and 10.8× (Tier-3);
  - severe shock: 4.72 / 3.98 / 1.16 crore for A / B / C; savings of 3.56 and 2.82 crore; 75% reduction;
  - moderate shock: mean +16.7 L, median −11.8 L; mild shock: +0.82 L extra cost in 100% of replications;
  - ablation: 3.64 / 1.21 / 4.73 crore; 2.48 crore from predicted links; 88% share; 0.34 crore from the MILP alone;
  - every p and Holm p in Table IV, every severe sensitivity CI in Table VIII, and every printed Table VII contrast;
  - macros vs recomputation.
- `experiments/VERIFICATION_REPORT.md`: **73/73 pass.** Covers splits, prevalence, AUCs, confusion matrices, CRN integrity, impact components,
  and confirms the ablation arms reproduce A/B/C exactly.
- `pytest tests`: **35 passed.**
- Counts in Fig. 1 verified against the code:
  - 22 database tables (`backend/database_schema.sql`);
  - 12 frontend pages;
  - 31 API routes (29 blueprint routes + `/api/health` + `/api/docs`);
  - 9 assistant tools;
  - 35 tests.
- `download_real.sh` was re-run on 13 Nov 2026: both repositories cloned at their pinned commits, and the DataCo SHA-256 check returned OK.
- Repository: public, default branch `claude/exciting-euler-5v1r2l` (checked through the GitHub API).

## 3. AI-use declaration and evidence

See `AI_USE_DECLARATION.md`. In summary, all 12 git commits were created in Claude Code sessions (each has a `Co-Authored-By: Claude`
trailer), so the evidence does **not** support "limited assistance" or "below 20%". The declaration states the actual extent and
separates out the Gemini API as a runtime component. It also explains how the authors can genuinely reduce AI dependence and then update
the declaration.

## 4. Items not independently verifiable from this environment

- **Live site.** This session's network blocks `onrender.com`, so the page could not be opened here. The Render API reports the deploy of
  commit `d9a7f5e` as **live**, and the same Docker build was tested locally.
- **Full `run_all.sh`.** A complete end-to-end re-run (including GNN training) was not repeated for this revision. Individual stages and
  all reported numbers were re-checked as listed above.
- **References.**
  - Hendricks & Singhal (2005) DOI: not confirmed, so no DOI is printed.
  - Platt (1999): the book is dated 1999 by some sources and 2000 by others; 1999 and pp. 61–74 (the conventional citation) are kept.
  - XGBoost: the end page (794) was not shown by the search results; it is the standard citation.
  - Crossref was unreachable from this environment; references were checked against publisher, RePEc/IDEAS, arXiv, DBLP and
    Mendeley records.
- **Institutional AI policy.** This environment cannot see the institution's policy. The authors must confirm that the declaration
  complies with it.
- **Provenance.** "No earlier project of the authors was reused" is an author declaration; only the authors can confirm it.
