# Economic Assumptions (INR)

**Status: ILLUSTRATIVE ASSUMPTIONS for a synthetic demonstration.** None of these values is a verified company figure or a
sourced market quote; live market prices could not be retrieved from this environment. They were chosen to be of a plausible
order of magnitude for an Indian FMCG manufacturer so that the *relative* comparison of response policies is meaningful. The
machine-readable copy used by the code is `experiments/scenario_assumptions.json`; every scenario uses the same file.

| Parameter | Value | Where used | Rationale |
|---|---|---|---|
| Raw-material unit prices | `materials.csv` (e.g. cocoa powder ₹520/kg, cocoa butter ₹1,450/kg, malt extract ₹140/kg, sugar ₹42/kg) | procurement cost, holding cost, shortage value | illustrative order-of-magnitude values |
| Product prices & contribution margins | `products.csv` (₹25–₹480 per pack; margin 24–40 %) | lost contribution margin | typical FMCG margin range used for demonstration |
| Alternative-supplier premium (approved supplier, incremental volume) | +6 % | MILP, rule-based, cost accounting | spot/rush premium |
| Substitute-material supplier premium | +10 % | MILP | qualification & reformulation |
| Onboarding / qualification fixed cost | ₹1,50,000 per substitute supplier; +7 days lead time | MILP binary *z* | trial batches, audits |
| Freight (standard) | 2.5 % of material value (road), 4 % (sea) | shipments, recovery transport | |
| Expedited transport | 3 × standard freight rate; 50 % of standard lead time (min 2 days) | MILP option | air / dedicated truck |
| Inventory holding cost | 24 % per annum of value (0.066 % per day) | holding cost of excess stock | capital + storage + obsolescence |
| Production interruption cost | ₹75,000 per plant-day with service level < 90 % | scenario impact | line changeovers, idle labour |
| Shortage penalty | 15 % of selling price per unmet unit, in addition to lost contribution margin | scenario impact, MILP π | retailer OTIF penalties, goodwill |
| Recovery-time (management) cost | ₹25,000 per day until supply and production return to ≥ 99 % of plan | scenario impact | war-room / expediting overhead |
| Risk limit for alternates | calibrated 30-day disruption probability ≤ 0.35 | MILP | risk appetite |
| Maximum allocation share | 60 % of a deficit per supplier (when > 1 candidate) | MILP | diversification policy |
| Decision-policy action costs | monitoring ₹0 · enhanced ₹5,000 · contingency review ₹20,000 · escalate & recover ₹60,000 | Exp 1 decision policy | analyst/time costs |
| Decision-policy mitigation | 0 · 15 % · 45 % · 70 % of expected loss avoided | Exp 1 decision policy | assumption |

## Financial impact definition (per scenario replication)

```
impact = lost contribution margin + shortage penalty + procurement premium + recovery transport
       + holding cost of excess inventory + onboarding cost + production-interruption cost + recovery-management cost
```

Base material spend that would have been incurred anyway is *not* an impact; only the premium over base price counts. Total
recovery procurement spend is reported separately. The verification script checks that the components sum to the reported impact.
