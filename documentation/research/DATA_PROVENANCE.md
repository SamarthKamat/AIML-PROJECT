# Data Provenance

## Directory contract

| Folder | Content | Mutable? |
|---|---|---|
| `datasets/raw/` | Files exactly as downloaded (DataCo mirror, SupplyGraph clone) | **Never modified** |
| `datasets/synthetic/` | Generator output (seed 42) + `ground_truth/` latent variables | Regenerated only by the generator |
| `datasets/processed/` | Cleaned real data (`process_real.py`) | Regenerated from raw |
| `datasets/experiment/` | Point-in-time feature tables (Exp 1) | Regenerated from synthetic |
| `datasets/data_dictionary/` | Generated data dictionary | Regenerated |

## Lineage

```
DataCo (Mendeley DOI 10.17632/8gx2fvg2k6.5, via GitHub mirror @7a466e7)
  └─ ml/data_engineering/process_real.py ─▶ datasets/processed/dataco_shipments.csv ─▶ Exp 3
SupplyGraph (CIOL-SUST/SupplyGraph @0e6d010)
  └─ ml/data_engineering/process_real.py ─▶ supplygraph_nodes_daily.csv, supplygraph_edges.csv ─▶ Exp 2B (part 2)
ml/data_generation/generate_synthetic.py (seed 42)
  └─ datasets/synthetic/*.csv ─▶ ml/data_engineering/features_supplier.py ─▶ exp1_supplier_snapshots.csv ─▶ Exp 1, 2B (part 1)
                              ─▶ ml/graph/* ─▶ Exp 2A ─▶ candidate_links.csv ─▶ Exp 2B, scenarios, app
                              ─▶ ml/simulation/* + ml/optimisation/* ─▶ Exp 4, nine scenarios
                              ─▶ backend/data_import.py ─▶ MySQL / SQLite ─▶ web application
```

## How the synthetic data are generated (summary; code is authoritative)

* **Suppliers.** Created per material and tier with counts in `catalog.py`; region drawn from the material's plausible regions;
  latent financial risk ~ Beta(2, 5); utilisation ~ Beta(6, 3) clipped to [0.45, 0.97]; reliability decreases with latent financial risk;
  the *observed* financial-risk score adds N(0, 0.08) noise; an unobserved frailty ~ N(0, 0.45) is stored only in
  `ground_truth/supplier_latent.csv`.
* **Relationships.** Tier-1 → manufacturer for every Tier-1 supplier, shares ~ Dirichlet. A buyer that needs input material *m*
  selects k ∈ {2,…,6} sources among the suppliers of *m* with probability ∝ size^1.6 · exp(−distance / 2,500 km); shares ~ Dirichlet(1.5).
  Visibility: every Tier-1 edge observed; Tier-2 → Tier-1 edges observed with p = 0.45; Tier-3 → Tier-2 with p = 0.15.
* **Disruptions.** Daily hazard h_i(t) = b · exp(1.8 f_i + 3(u_i − 0.75) + 1.5 g_i + ε_i + 1.4 s_i(t)) + 0.035 · Σ_u W_iu · sev_u(t),
  with latent financial risk f, utilisation u, geo-risk g, frailty ε, regional seasonality s (monsoon, cyclone, West-African lean season) and
  contagion over the TRUE network (W = dependency shares divided by number of input materials). b is calibrated so the two-year total is
  ≈1,000 events. Type, severity ~ Beta(2, 2.2) (contagion events: severity ∝ upstream loss) and duration ~ log-normal by type.
  Reported date = start + Poisson(1.5) days.
* **Class label.** `disruption_next_30_days` = a *qualifying* disruption (capacity loss ≥ 25 % and duration ≥ 3 days) starts in (t, t + 30 days].
* **Shipments.** For each Tier-1 → manufacturer edge and month: 1–3 POs for 105 % of the planned need × share (5 % replenishment buffer);
  each PO split into 1–3 shipments; delay ~ N(μ, 1.3) rounded at ≥ 0 with μ = −0.6 + 6(1 − reliability) + 2.2·seasonality + (4 + 10·severity if
  disrupted) + sea-freight noise; delivered quantity reduced while disrupted; `is_delayed = 1[delay ≥ 2 days]`.
* **Material compatibility.** Same material = 1.0; ten explicit substitution pairs with compatibility 0.5–0.9 (`material_substitutes.csv`).
* **Inventory.** Weekly recursion per plant-material: closing = opening + receipts − consumption, consumption limited by the scarcest
  BOM input of each product; initial stock = 3 weeks of requirement; storage cap 6 weeks.
* **Dependency percentages.** Dirichlet shares, normalised per buyer-material.
* **Costs.** Material prices and product prices are illustrative INR values (see `ECONOMIC_ASSUMPTIONS.md`).

## Labelling rules used everywhere

* Synthetic records: `data_label = synthetic`; UI shows a "SYNTHETIC DEMO DATA" badge on every page.
* `visibility = hidden` edges are **ground truth only**. They are never model inputs; they are used to (a) drive the simulator (the
  physical world) and (b) evaluate discovery and the hidden-dependency gap.
* Predicted links are labelled "candidate (predicted, unverified)".
* Simulated money is labelled "simulated INR".
