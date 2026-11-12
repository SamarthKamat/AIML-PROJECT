# Final Proposal and Literature Review

## 1. Proposal
**Title.** Supply Chain Black Box: AI-powered hidden supplier dependency discovery, cascading risk prediction and intelligent supply chain recovery.

**Problem.** An FMCG chocolate and malt-beverage manufacturer knows its Tier-1 suppliers but not the Tier-2/3 suppliers they share. Disruptions at hidden
hubs propagate to several Tier-1 suppliers at once, and procurement reacts only when inventory runs low.

**Objectives.** (1) Discover candidate hidden relationships. (2) Predict calibrated 30-day supplier disruption risk. (3) Identify structurally critical
suppliers. (4) Simulate cascades and quantify inventory, production and financial exposure. (5) Identify alternative suppliers and optimise recovery.
(6) Explain recommendations and make them executable in an application.

**Hypotheses tested** (outcomes are in brackets).
* H1: Calibration improves decision quality. [Supported for probability quality (ECE halved). The realised-cost benefit is small because few actions are triggered.]
* H2: Message-passing GNNs outperform feature-based models for hidden-link discovery and risk classification. [**Not supported** on either the synthetic or the real data.]
* H3: Network-aware optimisation reduces disruption impact compared with rule-based recovery. [Supported for moderate and severe shocks; not supported for mild shocks, where it over-hedges.]

**Originality claim.** The defensible, leakage-controlled *integration* — discovery → calibrated prediction → exposure → cost-optimal action → MILP —
together with the quantified observed-vs-true exposure gap. We do not claim new learning algorithms.

## 2. Literature review (themes)

### 2.1 Supply chain risk prediction
Baryannis *et al.* (2019, IJPR 57(7):2179–2202) review AI in supply chain risk management (SCRM) and find most work on risk identification and assessment,
with less on response; our decision engine targets that gap. Hendricks and Singhal (2005, POM 14(1):35–52) document long-run stock-price and
equity-risk effects of disruptions, motivating the business significance. Chen and Guestrin (2016) provide XGBoost, our primary tabular learner.

### 2.2 Probability calibration
Platt (1999) introduced sigmoid calibration, and Zadrozny and Elkan (2002) isotonic calibration. Niculescu-Mizil and Caruana (2005) show that boosted
trees and maximum-margin methods are systematically mis-calibrated and benefit from post-hoc calibration. We use these methods to turn scores into
decision-grade probabilities and report Brier and ECE.

### 2.3 Graph-based supply chain intelligence
Brintrup *et al.* (2018, Complexity, doi:10.1155/2018/9104387) frame hidden-dependency discovery as link prediction on a real automotive supply
network. Kosasih and Brintrup (2022, IJPR 60(17):5380–5393) apply GNNs to the same problem and report improvements over heuristics. Classical
heuristics (Liben-Nowell and Kleinberg 2007), node2vec (Grover and Leskovec 2016), GCN (Kipf and Welling 2017), GraphSAGE (Hamilton *et al.* 2017)
and GAT (Veličković *et al.* 2018) are our comparators. SupplyGraph (Wasi *et al.* 2024) is a real FMCG graph benchmark on which GNNs were
reported to outperform non-graph baselines. Our replication on a different target (7-day service shortfall), with chronological splits and
calibration, did **not** find that advantage.

### 2.4 Cascading disruption and recovery
Snyder *et al.* (2016, IIE Transactions 48(2):89–109) review OR/MS disruption models (inventory, sourcing, contracts). Ivanov, Dolgui and
Sokolov (2019, IJPR 57(3):829–846) discuss the ripple effect and digital risk analytics. Our simulator is a deliberately simple, transparent,
daily ripple model with buffers and lead-time lags. Our MILP is a single-period sourcing model with expedited-mode and onboarding decisions.

### 2.5 Gap and positioning
Existing studies address discovery, prediction or recovery in isolation and rarely report calibrated, leakage-controlled, policy-level business
outcomes with explicit economics. We evaluate the combination and publish negative results where they occur.

Full bibliographic entries are in `documentation/final_report/main.tex`.
