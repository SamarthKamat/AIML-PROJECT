#!/usr/bin/env bash
# Reproduce every dataset, model, experiment, figure and verification in order.
# Usage: bash run_all.sh            (~25-40 min on 4 CPU cores; no GPU needed)
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONWARNINGS=ignore
step() { echo; echo "==> $*"; }
if [ ! -f datasets/raw/dataco/DataCoSupplyChainDataset.csv ]; then step "Download real datasets"; bash ml/data_engineering/download_real.sh; fi
step "1. Synthetic FMCG ecosystem (seed 42)";       python -m ml.data_generation.generate_synthetic
step "2. Process real datasets";                    python -I ml/data_engineering/process_real.py
step "3. Validate data";                            python -m backend.data_validation
step "4. Point-in-time features (Exp 1)";           python -m ml.data_engineering.features_supplier
step "5. Exp 1 supplier risk + calibration";        python -m ml.experiments.exp1_supplier_risk
step "6. Exp 3 DataCo delay (real)";                python -m ml.experiments.exp3_delivery_delay
step "7. Exp 2A link prediction";                   python -m ml.experiments.exp2a_link_prediction
step "8. Exp 2B graph risk classification";         python -m ml.experiments.exp2b_graph_risk
step "9. Exp 4/5 nine scenarios + disruption types"; python -m ml.simulation.scenarios
step "10a. Robustness: link ablation + sensitivity"; python -m ml.experiments.robustness all
step "10b. Statistical tests (bootstrap, Wilcoxon-Holm)"; python -m ml.experiments.statistics
step "10c. Sensitivity: LR risk in scenarios";       python -m ml.experiments.sensitivity_lr_risk
step "10d. Independent verification";                python -m ml.experiments.verify_results
step "11. Report figures, tables, diagrams, dictionary"; python -m ml.experiments.report_figures && python -m ml.experiments.paper_figures && python -m ml.experiments.paper_tables && python documentation/diagrams/make_diagrams.py && python -m ml.data_engineering.data_dictionary
step "11b. Report-number audit";                  python -m ml.experiments.audit_report_numbers
step "12. Database (DATABASE_URL or SQLite)";       python -m backend.database_setup --drop && python -m backend.data_import
step "13. Tests";                                   python -m pytest tests -q
echo; echo "All steps completed."
