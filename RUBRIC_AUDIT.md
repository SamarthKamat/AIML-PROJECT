# Rubric Audit (self-assessment — final phase)

This audit was written by the AI assistant and must be re-checked by the team. Marks are estimates of what the *evidence* supports, not a
prediction of the faculty's grade.

| Criterion | Max | Evidence available | Missing / weak evidence | Current assessment | Corrective action |
|---|---:|---|---|---:|---|
| Relevance | 4 | Concrete FMCG decision (monthly action + recovery allocation); executable decision engine; FMCG-specific BOMs, materials and regions | Company is fictional; no practitioner validation | 3.4 | Add a short interview or quote from an FMCG procurement professional if available |
| Significance | 4 | Nine-scenario comparison with 13 metrics, 500 CRN replications, paired CIs (severe: ₹3.56 Cr saved, CI 3.25–3.87); explicit economics; black-box gap 2.1× / 10.8× | INR values are illustrative; supplier-level policy shows little value | 3.2 | Calibrate the economics with any real cost figures the team can source; keep the honesty about mild shocks |
| Originality | 6 | Leakage-controlled integration of discovery, calibration, exposure, decision and MILP; quantified observed-vs-true exposure gap; derated-availability MILP; negative GNN findings with an oracle ablation | Components individually known; validated on synthetic supplier data only | 4.3 | Emphasise the integration and the oracle-graph ablation in the viva; do not claim algorithmic novelty |
| Achievement | 4 | 5 experiment families run and logged; 9 trained model artefacts; working 12-page app; 35 tests + MySQL run; screenshots; 61 verification checks | Gemini path not executed; no deployment | 3.5 | Run the assistant once with a Gemini key and add a screenshot |
| Writing | 4 | IEEE format, 5 pages, no template residue, 5 tables + 3 figures from data, 17 verified references | Placeholder author block; dense prose; AI-drafted text must be rewritten for the 20 % rule | 3.0 | Team rewrites the text in their own words; fill in the author block |
| Reproducibility | 4 | `run_all.sh`; pinned, SHA-256-verified downloads; seeds; `run.json` registry; generated tables and number macros; data dictionary; requirements | DataCo raw file not redistributed (licence caution); GPU-free but ~30 min runtime | 3.6 | Push the repository (done on the development branch); optionally attach a release archive |
| Technical quality | 4 | Correct target and timestamp; embargoed chronological split; calibration on a separate split; cost-derived thresholds; test edges removed from message passing; validation-only model selection; CRN; verification of metrics | Modest predictive performance; synthetic labels for supplier tasks; single-period MILP | 3.3 | Discuss limitations openly; propose a stochastic/rolling-horizon MILP as future work |
| **Total** | **30** | | | **≈24.3** | |

## Audit history
* Phase 1–2 (requirements, data): evaluation matrix built (`documentation/research/REQUIREMENTS_AND_FEEDBACK_MATRIX.md`); real datasets verified.
* Phase 5–7 (experiments): GNNs did not beat feature models; this was reported, and the link model was chosen by validation AP.
* Phase 9–10 (scenarios): two simulator defects were found by checking results (an inventory cap that discarded deliveries; hard exclusion of
  alternates) and fixed. The horizon was extended to 180 days because recovery time was censored in 52 % of replications (now 3 %). All reruns were logged.
* Phase 14 (testing): a MySQL reserved-word bug and two UI defects (SPA routing, NaN JSON) were found and fixed.
