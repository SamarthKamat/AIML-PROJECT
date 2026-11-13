"""Sensitivity analysis: re-run the nine scenarios with the logistic-regression risk model in place of the
pre-specified XGBoost model (risk feeds policy B/C decisions and the simulated failure chance of alternates).
Writes experiments/sensitivity_lr_risk/.  Run: python -m ml.experiments.sensitivity_lr_risk
"""
import ml.simulation.scenarios as S
from ml.common.config import EXPERIMENTS


def main():
    S.RISK_MODEL = "exp1_logistic_regression.joblib"
    res, _ = S.run_nine(verbose=False)
    summ = S.summarise(res)
    d = EXPERIMENTS / "sensitivity_lr_risk"
    d.mkdir(exist_ok=True)
    summ.to_csv(d / "scenario_results_lr_risk.csv", index=False)
    comp = S.paired_comparison(res)
    comp.to_csv(d / "scenario_comparison_lr_risk.csv", index=False)
    print(summ[["scenario", "financial_impact_inr", "unmet_demand_units"]].round(0).to_string())
    print(comp[["severity", "comparison", "mean_saving_inr"]].round(0).to_string())


if __name__ == "__main__":
    main()
