# Viva Questions and Answers (reflecting the actual implementation)

Numbers refer to the logged runs in `experiments/` (see `experiments/VERIFICATION_REPORT.md`).

## A. Problem, data and business framing

**1. What exact business decision does the system support?**
Two decisions. (i) Monthly, per supplier: which of four actions to take (routine monitoring, enhanced monitoring, contingency review, escalate & run recovery), chosen by `argmin_a C_a + q(1−ρ_a)E` in `ml/decision/policy.py`. (ii) After a reported shock: which alternative suppliers receive which quantities by which transport mode, from the MILP in `ml/optimisation/recovery.py`.

**2. Why is the supplier data synthetic, and how do you avoid overclaiming?**
No verifiable public dataset contains multi-tier supplier relationships, dependency shares, disruptions, BOMs and inventory for one FMCG firm (AnonNet/SynDelay could not be verified; UN Comtrade and NOAA were unreachable). Every synthetic row is labelled, the UI shows a "SYNTHETIC DEMO DATA" badge, money is called "simulated INR", and the paper states that supplier-level performance on real networks is not established. Real data (DataCo, SupplyGraph) are used only where they fit.

**3. How were the synthetic relationships generated, and why does that matter for link prediction?**
A buyer needing input material m picks 2–6 sources among producers of m with probability ∝ size^1.6·exp(−distance/2,500 km). Tier-2 edges are observed with p = 0.45, Tier-3 with p = 0.15 (461 of 1,500 observed). Because formation depends on node attributes (material, location, size), a feature-based pair model can learn it — which explains why the feature-only MLP beat GNNs.

**4. What is the "black-box gap"?**
Mean network exposure computed on the observed graph versus the true graph: Tier-2 exposure is understated 2.1× and Tier-3 10.8×. This is what the manufacturer does not know it is exposed to.

**5. Why did you choose cocoa-bean traders for the nine scenarios?**
They are concentrated Tier-2 hubs: the five largest have 19 observed and 25 hidden outgoing links to Tier-1 cocoa processors. A single-supplier outage was fully absorbed by multi-sourcing and ~30 days of cover even for 90 days, so a hub-level shock is the business-relevant stress case (and mirrors real cocoa-origin concentration).

## B. Supervised learning, leakage and timestamps

**6. Define the target precisely.**
`disruption_next_30_days = 1` if a qualifying disruption (capacity loss ≥ 25 %, duration ≥ 3 days) **starts** in (t, t + 30 days], with t = 00:00 on the first of the month.

**7. How do you prevent leakage?**
Features use disruptions with `reported_date < t` (reporting lag of Poisson(1.5) days is modelled), shipments with `actual_delivery_date < t`, the monthly risk forecast for t, and only observed edges. `tests/test_leakage.py` tampers with future records and hidden edges and asserts the features are unchanged.

**8. Describe the split.**
Chronological by snapshot: train 2024-04…2025-02 (5,500 rows), embargo 2025-03 (its label window overlaps), calibration 2025-04…06 (1,500), test 2025-07…12 (3,000; prevalence 7.4 %). Exp 3 uses a chronological 60/20/20 split by order time.

**9. Why is accuracy not reported as the headline metric?**
At 7.4 % prevalence, predicting "no disruption" gives 92.6 % accuracy. We report ROC-AUC, PR-AUC (Saito & Rehmsmeier's argument for imbalanced data), Brier score, ECE, and full confusion matrices.

**10. Your ROC-AUC is only ~0.63–0.67. Is the model useful?**
Discrimination is modest because the generator includes an unobserved frailty and contagion through hidden edges — realistic for rare events. The models still beat chance (PR-AUC 0.12–0.15 vs 0.074 base rate) and, more importantly, produce calibrated probabilities that feed a cost-based decision. We report this honestly instead of tuning the generator.

**11. Why is logistic regression better than XGBoost here?**
The signal is mostly additive in a few drivers (utilisation, seasonal weather, financial risk); with 5,500 training rows the tree ensemble overfits slightly. We kept XGBoost as the deployed model because it was pre-specified and supports TreeSHAP, but we report LR's better ranking.

**12. What does SHAP tell you?**
Mean |SHAP| ranks capacity utilisation, weather-risk forecast and financial-risk score highest — consistent with the generator's latent drivers, which is a sanity check rather than a discovery.

## C. Calibration and thresholds

**13. Why calibrate?**
The action rule multiplies probability by exposure; a mis-calibrated probability mis-prices every action. Platt reduced XGBoost ECE from 0.020 to 0.010 and the decision tree's from 0.037 to 0.011.

**14. How were calibrators fitted without leakage?**
On the calibration split only; the method (none/Platt/isotonic) was chosen by 5-fold CV Brier **within** that split. Test labels were used only for the final evaluation.

**15. What is the Brier score, and how is ECE computed?**
Brier = mean squared difference between predicted probability and outcome. ECE = Σ_b (n_b/N)|ȳ_b − q̄_b| with 10 equal-width bins (equal-mass ECE is also logged).

**16. How are thresholds derived?**
From costs, not chosen arbitrarily: action b replaces a when q > (C_b − C_a)/(E(ρ_b − ρ_a)). The rule is Bayes-optimal if q is calibrated.

**17. The decision policy issued only 5 alerts and 0 true positives. Isn't that a failure?**
It is an honest finding: with 3–4 weeks of inventory, most single-supplier events cause little expected loss, so the cost-optimal action is usually monitoring. Realised cost was ₹118.3 L vs ₹116.9 L for monitor-only and ₹664.3 L for reviewing everyone; the uncalibrated version cost ₹120.4 L. Value concentrates in network-level shocks, which is why recovery optimisation matters.

## D. Graph learning

**18. Explain GraphSAGE.**
Each layer updates h_v = ReLU(W1·h_v + W2·mean of neighbours' h_u); two layers of 64 units, dropout 0.3, then a pair decoder MLP([z_u ⊙ z_v, |z_u − z_v|]). It is inductive: it learns aggregation functions, not node-specific embeddings.

**19. How did you split edges and avoid test-edge leakage?**
Observed edges were split 70/10/20; test edges were removed from the message-passing graph; validation edges were used only for early stopping. Hidden edges were never used for training.

**20. Why use material-compatible negatives?**
Random negatives are trivially easy (a cocoa trader never supplies a carton maker). Compatible negatives — the supplier makes an input the buyer needs — are the real business question.

**21. Why do Common Neighbours and Adamic–Adar score 0.5?**
The network is layered (Tier-3 → Tier-2 → Tier-1 → manufacturer), so a Tier-2 supplier and a Tier-1 buyer never share a neighbour. Their scores are all zero.

**22. GNNs did not win. How do you explain that?**
The observed graph is sparse (461 edges; most Tier-3 nodes are isolated), and edge formation is driven by node attributes, so message passing mostly adds noise. The validation-selected pair MLP reached AP 0.227 against a base rate of 0.117, versus 0.168 for GraphSAGE. In Exp 2B, GNNs did not beat tabular models even with the oracle graph, and on real SupplyGraph logistic regression was best (PR-AUC 0.488). We report this negative result rather than hiding it.

**23. How did you choose the link model used downstream without peeking at the test set?**
By mean validation AP on held-out observed edges in A2 (MLP 0.911 vs GNNs 0.826–0.863). The test (hidden-edge) metrics were reported for all models but not used for selection.

**24. Are predicted links treated as real?**
No. They are labelled "candidate (predicted, unverified)" in the API and UI and used only as an *estimate* of exposure in Policy C. The simulator always uses the true network.

## E. Simulation, Monte Carlo and optimisation

**25. How does the cascade simulator work?**
Daily: each buyer-input slot holds a 10-day buffer; arrivals equal Σ share × upstream availability lagged by lead time; output = min(1 − own loss, min over inputs of available input). At the plants, BOM-bottleneck production gives unmet demand per product.

**26. Why Monte Carlo, and why 500 replications?**
Duration (log-normal), demand noise, alternate lead times, yields and independent alternate failures are uncertain. With 500 replications the relative standard error of the severe no-intervention mean is 3.9 %, and the convergence plot is flat.

**27. What are common random numbers, and why use them?**
Replication r uses seed base + r for every policy, so all three policies face identical durations, demand and supplier draws. Paired differences then have much lower variance, e.g. the severe A−C saving is ₹3.56 Cr with a 95 % CI of ₹3.25–3.87 Cr.

**28. Formulate the MILP.**
It minimises (price × (1 + freight + holding)) × quantity + onboarding × z + shortage penalty × s. Constraints: expected-yield cover (reliability × (1 − risk) × compatibility × availability × x + s ≥ deficit), capacity before the deadline, a 60 % allocation-share cap, a calibrated-risk limit of 0.35, and onboarding linking x ≤ M·z. It is solved with PuLP/CBC (17 variables / 21 constraints in the example run).

**29. Why is Policy C sometimes worse than doing nothing?**
Under a mild shock, buffers cover the loss, so proactive buying is wasted (+₹0.82 L). Under a moderate shock, C has the lowest mean impact (₹16.7 L saved), but no intervention is cheapest in 82 % of replications; C pays off in the tail. Under a severe shock, C is cheapest in 73 % of replications and cuts mean impact by 75 %. This argues for a tiered policy.

**30. What fairness controls apply to the policy comparison?**
The same network, demand plan, initial inventory, assumptions file, horizon (180 days) and random draws apply to every policy. Policy A places no orders, which is verified. The impact components are verified to sum to the reported impact.

**31. You changed the simulator during development. Is that result manipulation?**
Two modelling defects were fixed and are documented: (a) an inventory cap that silently discarded early deliveries, replaced by a 60-day storage cap; (b) hard exclusion of every alternate linked to the hubs, replaced by quantitative availability derating. We also extended the horizon from 120 to 180 days because recovery time was right-censored in 52 % of replications (now 3 %). The changes apply equally to all policies, and every rerun was reported.

## F. Financials, explainability, system and reproducibility

**32. Where do the INR values come from?**
They are illustrative assumptions, listed in `documentation/research/ECONOMIC_ASSUMPTIONS.md` and `experiments/scenario_assumptions.json` (e.g., 6 % alternate premium, 3× expedited freight, 24 %/yr holding, 15 % shortage penalty). They are not company figures; conclusions concern relative policy performance.

**33. How does the AI assistant avoid hallucination?**
The LLM (Gemini, when a key is configured) can only call nine tools: database queries, model inference, simulation and MILP. The answer is returned together with the raw tool outputs ("Show evidence"). Without a key, a deterministic intent router calls the same tools. The Gemini path was not executed in our environment; the router is covered by 7 parametrised tests.

**34. How can someone reproduce your results?**
Run `bash run_all.sh`. It performs pinned, hash-verified downloads, regenerates data (seed 42), runs all experiments, executes 61 verification checks, rebuilds every paper table and the number macros from result files, loads the database, and runs 35 tests (also passing on MySQL). A repository URL alone is not the claim; the executable pipeline is.

**35. What would you do with more time or real data?**
Label hidden links with bills of lading or supplier surveys; validate on a real supplier network; replace the static MILP with a stochastic/rolling-horizon policy to avoid over-hedging; and calibrate the economics with company data.
