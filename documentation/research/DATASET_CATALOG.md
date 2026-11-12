# Dataset Catalog and Selection Matrix

Download date for all files: **2026-11-12** (this project session). Counts below were measured on the downloaded files,
not copied from publishers' pages.

## 1. Candidates investigated

| Candidate | Publisher / source | Access in this environment | Verified? | Decision |
|---|---|---|---|---|
| **DataCo Smart Supply Chain for Big Data Analysis** | Constante, Silva & Pereira (2019), Mendeley Data V5, doi:10.17632/8gx2fvg2k6.5, licence CC BY 4.0 | Mendeley blocked by network policy (HTTP 403 at proxy); obtained from GitHub mirror `ashishpatel26/DataCo-SMART-SUPPLY-CHAIN-FOR-BIG-DATA-ANALYSIS` commit `7a466e7` (2021-11-17), which cites the Mendeley DOI | Files downloaded and inspected; SHA-256 recorded; byte-identity with the Mendeley original **could not be verified** (Mendeley unreachable) | **Selected** for Experiment 3 (shipment delay) – real shipment-level outcomes |
| **SupplyGraph** | Wasi, Islam & Akib (2024), arXiv:2401.15299; GitHub `CIOL-SUST/SupplyGraph` commit `0e6d010` (2025-01-15); repository licence LGPL-2.1 | Cloned via git | Files downloaded and inspected | **Selected** for Experiment 2B (real FMCG graph learning) |
| AnonNet | Named in the brief | No verifiable data files, documentation or licence located from this environment | Not verified | Not used |
| SynDelay | Named in the brief | No verifiable files or provenance located | Not verified | Not used |
| UN Comtrade | United Nations | API host blocked by network policy (HTTP 403) | Not accessed | Not used; listed as future work (cocoa HS 1801 import flows) |
| NOAA climate data | NOAA | Not reachable | Not accessed | Not used; seasonal weather risk is synthetic |
| Kaggle mirrors | Kaggle | Host not reachable | — | Not used (GitHub mirrors used instead, with licence noted) |

## 2. Selected real datasets

### 2.1 DataCo (REAL)
* Raw file: `datasets/raw/dataco/DataCoSupplyChainDataset.csv`, SHA-256 `994b3c8d24049cc46bf20e8161fedece9a9b95376cc0dc929921d76cf6b9bc0d`
* Original size: **180,519 rows × 53 columns**; order dates 2015-01-01 → 2018-01-31.
* Targets present: `Late_delivery_risk` (binary), `Days for shipping (real)` vs `Days for shipment (scheduled)`.
  We verified `Late_delivery_risk == 1[(real − scheduled) > 0]` for every non-cancelled row (assertion in `process_real.py`).
* Missing values: `Order Zipcode` 155,679; `Product Description` 180,519 (entirely empty); `Customer Lname` 8; `Customer Zipcode` 3.
* Removed: 7,754 cancelled orders (no delivery outcome) → **172,765 processed rows**.
* PII-like columns (names, e-mail, password, street, zip) are dropped in processing.
* Limitations: retail/sports/electronics goods (not FMCG food); US-centric company; no supplier identity, so it supports
  *shipment-delay* modelling but not supplier-network modelling.
* Suitability: tabular ML ✔; graph learning ✘.

### 2.2 SupplyGraph (REAL, FMCG)
* Raw files: `datasets/raw/supplygraph/Raw Dataset/Homogenoeus/…` (nodes, 4 edge files, 8 temporal files).
* 41 node rows but **40 unique products** — `POP001L12P` is listed twice (both with index 12) and the temporal files contain a
  duplicate column `POP001L12P.1`; we keep the first occurrence (documented data-quality issue).
* 221 daily observations, 2023-01-01 → 2023-08-09; series: production, sales order, delivery to distributor, factory issue (units and weight).
* Edges (after de-duplication): 684 unique product pairs from same-plant (360), same-storage (665), same-group (179), same-sub-group (48) relations.
* Target we define (not supplied by the publisher): 7-day service shortfall = deliveries in (t, t+7] < 80 % of sales orders in (t, t+7];
  5,466 product-days with positive orders; prevalence 28.2 %.
* Limitations: products (not suppliers) as nodes; short horizon; anonymised codes; one company.
* Suitability: graph learning ✔ (homogeneous product graph); supplier-network discovery ✘.

## 3. Synthetic FMCG demonstration dataset

No verifiable public dataset contains multi-tier supplier relationships, dependency shares, disruptions, inventory and BOMs for one
FMCG manufacturer. We therefore generate a **fictional** chocolate & malt-beverage ecosystem (`datasets/synthetic/`,
`ml/data_generation/generate_synthetic.py`, seed 42). Every row carries `data_label = synthetic`.

| Table | Rows | Target in brief |
|---|---:|---:|
| companies | 1 | 1 |
| suppliers | 500 (120 T1 / 200 T2 / 180 T3) | 500 |
| supplier_relationships | 1,500 (461 observed / 1,039 hidden) | 1,500 |
| materials | 50 (28 / 14 / 8 by BOM level) | 50 |
| products | 100 | 100 |
| inventory | 5,824 | 5,000 |
| shipments | 9,579 | 10,000 |
| disruptions | 1,048 (782 qualifying) | 1,000 |
| procurement | 4,929 | 5,000 |
| external_risks | 504 | — |
| warehouses | 4 | — |
| production | 10,400 | — |
| bill_of_materials | 960 | — |
| material_substitutes | 10 | — |

## 4. Selection matrix

| Requirement | DataCo | SupplyGraph | Synthetic |
|---|:-:|:-:|:-:|
| Real observations | ✔ | ✔ | ✘ |
| FMCG | ✘ | ✔ | ✔ (fictional) |
| Timestamped outcomes | ✔ | ✔ | ✔ |
| Supplier identities & tiers | ✘ | ✘ | ✔ |
| Supplier relationships / hidden links | ✘ | ✘ (product graph) | ✔ |
| Disruption events | ✘ | ✘ | ✔ |
| Inventory & BOM | ✘ | partial | ✔ |
| Used in | Exp 3 | Exp 2B (real) | Exp 1, 2A, 2B, 4, 5, scenarios, app |

Real and synthetic identities are **never merged**: DataCo and SupplyGraph are analysed in separate experiments and are not joined
to synthetic supplier IDs.
