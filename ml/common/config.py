"""Central configuration: paths, seeds and time windows used by every experiment.

All experiments import from here so that dataset versions, random seeds and the
prediction-timestamp conventions are defined once and recorded in the
experiment registry.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets"
RAW = DATA / "raw"
SYN = DATA / "synthetic"
PROC = DATA / "processed"
EXPDATA = DATA / "experiment"
MODELS = ROOT / "models"
EXPERIMENTS = ROOT / "experiments"
FIGURES = ROOT / "documentation" / "final_report" / "figures"

for _p in (SYN, PROC, EXPDATA, MODELS, EXPERIMENTS, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)

SEED = 42
GNN_SEEDS = [0, 1, 2, 3, 4]

# ---------------------------------------------------------------------------
# Synthetic ecosystem time frame
# ---------------------------------------------------------------------------
SIM_START = "2024-01-01"
SIM_END = "2025-12-31"

# Experiment 1 prediction timestamps: 00:00 on the first day of each month.
# Features use only records with timestamp strictly before the snapshot.
# Label window = (snapshot, snapshot + 30 days].
SNAPSHOT_FIRST = "2024-04-01"   # 90 days of history available before first snapshot
SNAPSHOT_LAST = "2025-12-01"
TRAIN_END = "2025-02-01"        # last train snapshot (label window ends 2025-03-03)
CAL_START = "2025-04-01"        # 1-month embargo between train and calibration
CAL_END = "2025-06-01"
TEST_START = "2025-07-01"       # calibration label windows end 2025-07-01 00:00 -> no overlap
HORIZON_DAYS = 30

# Qualifying disruption: capacity loss >= 25% lasting >= 3 days
QUAL_MIN_SEVERITY = 0.25
QUAL_MIN_DURATION = 3

SYNTHETIC_LABEL = "SYNTHETIC DEMONSTRATION DATA - fictional FMCG ecosystem"
