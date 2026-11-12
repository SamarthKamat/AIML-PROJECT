# Mathematical Modelling (with links to the implementation)

| # | Model | Formula | Variables | Assumptions | Business relevance | Code |
|---|---|---|---|---|---|---|
| 1 | Disruption target | y_i(t) = 1[∃e: s_e ∈ (t, t+30d], ℓ_e ≥ 0.25, d_e ≥ 3] | s start, ℓ capacity loss, d duration | only *qualifying* events matter | defines the decision horizon | `features_supplier.py` |
| 2 | Disruption probability | p̂ = f(x_i(t)), f ∈ {LR, DT, RF, XGBoost} | x: 25 point-in-time features | information available before t only | ranking of suppliers | `exp1_supplier_risk.py` |
| 3 | Platt calibration | q = σ(a·logit(p̂) + b) | a, b fitted by maximum likelihood on the calibration split | monotone sigmoid distortion | decision-grade probabilities | `ml/decision/calibration.py` |
| 4 | Isotonic calibration | q = g(p̂), g non-decreasing (pool-adjacent-violators) | — | monotone distortion | alternative calibrator | same |
| 5 | Brier, ECE | B = (1/N)Σ(q−y)²; ECE = Σ_b (n_b/N)\|ȳ_b − q̄_b\| | 10 equal-width bins | — | probability quality | `ml/common/tracking.py` |
| 6 | Dependency concentration | HHI_m = Σ_s w_sm² | w = share of material m from supplier s | observed edges | critical-dependency KPI | `services.dashboard_summary` |
| 7 | Graph centrality | degree, betweenness, PageRank on the reversed graph (α = 0.85) | — | observed network | structural criticality | `services.compute_graph_metrics` |
| 8 | GraphSAGE | h_v ← ReLU(W₁h_v + W₂·mean_{u∈N(v)} h_u) | 2 layers, 64 units | homophily of attributes | candidate-link scoring | `ml/graph/models.py` |
| 9 | Link score | P(u~v) = σ(MLP([z_u ⊙ z_v, \|z_u − z_v\|])) | z: node embeddings | symmetric pairs | ranking of candidate links | same |
| 10 | Reach | r_im = Σ_paths Π (w_e / n_inputs(buyer)) | n_inputs: number of distinct inputs of the buyer | bottleneck rule | propagation of a supplier's share | `ml/decision/exposure.py` |
| 11 | Exposure | E_i = Σ_m r_im·CM_m·E[S·max(0, D − c_m − 10(τ_i − 1))] | CM daily margin, c cover days, τ tier | empirical (S, D) from training period | ₹ at risk per disruption | same |
| 12 | Inventory coverage | c_m = I_m / R_m (days) | I stock, R daily requirement | — | buffer against shocks | inventory table, simulator |
| 13 | Cascade | a_b(t) = min(1 − ℓ_b(t), min_i min(1, I_bi(t) + Σ_u w_ub a_u(t − L_ub))) | 10-day slot buffers, lead time L | Leontief bottleneck, 15 % surge when refilling | ripple propagation | `ml/simulation/cascade.py` |
| 14 | Shortage / unmet demand | short_m = max(0, need_m − (I_m + arrivals_m)); unmet_p = D_p(1 − min_{m∈BOM(p)} fill_m) | — | no finished-goods buffer | lost sales | same |
| 15 | Decision rule | a* = argmin_a C_a + q(1 − ρ_a)E_i; threshold q* = ΔC / (E·Δρ) | C action cost, ρ mitigation | calibrated q | executable action | `ml/decision/policy.py` |
| 16 | Recovery MILP | see the paper, Eq. (5) | x quantity, z onboarding, s shortage | expected-yield cover, deadline, share and risk limits | allocation plan | `ml/optimisation/recovery.py` |
| 17 | Financial impact | lost CM + penalty + premium + transport + holding + onboarding + interruption + management | see `ECONOMIC_ASSUMPTIONS.md` | illustrative INR | policy comparison | `ml/simulation/scenarios.metrics` |
| 18 | Paired CRN saving | Δ = mean_r(Impact_A,r − Impact_C,r), CI = Δ ± 1.96·sd/√R | R = 500 | same seed per replication | statistical comparison of policies | `paired_comparison` |
