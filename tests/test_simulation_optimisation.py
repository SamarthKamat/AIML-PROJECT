import numpy as np
import pandas as pd
import pytest

from ml.optimisation.recovery import build_candidates, load_assumptions, material_penalty, optimise_recovery, rule_based_allocation
from ml.simulation.cascade import CascadeSimulator, Disruption, load_network


@pytest.fixture(scope="module")
def sim():
    net = load_network()
    return net, CascadeSimulator(net, horizon=90)


def test_no_disruption_no_loss(sim):
    net, s = sim
    o = s.run([])
    assert o["unmet"].sum() == pytest.approx(0, abs=1e-6)
    assert o["fill"].min() == pytest.approx(1.0)


def test_severity_monotone(sim):
    net, s = sim
    hubs = list(net.sup[(net.sup.tier == 2) & (net.sup.primary_material_id == "M29")].sort_values("size_index", ascending=False).index[:5])
    u = [s.run([Disruption(hubs, sev, 0, 60)])["unmet"].sum() for sev in (0.25, 0.5, 1.0)]
    assert u[0] <= u[1] <= u[2] and u[2] > 0


def test_common_random_numbers_reproducible(sim):
    net, s = sim
    d1 = s.make_draws(np.random.default_rng(7))
    d2 = s.make_draws(np.random.default_rng(7))
    a = s.run([Disruption(["S0122"], 1.0, 0, 40)], None, d1)["unmet"].sum()
    b = s.run([Disruption(["S0122"], 1.0, 0, 40)], None, d2)["unmet"].sum()
    assert a == b


def test_milp_respects_constraints(sim):
    net, s = sim
    A = load_assumptions()
    risk = pd.Series(0.05, index=net.sup.index)
    risk["S0003"] = 0.9  # must be excluded by risk limit
    cands = {"M01": build_candidates(net, "M01", {"S0001"}, risk, A, pd.read_csv("datasets/synthetic/material_substitutes.csv"))}
    orders, info = optimise_recovery({"M01": 5000.0}, cands, 30, material_penalty(net, A), A)
    assert info["status"] == "Optimal"
    by = {}
    for o in orders:
        by[o["supplier_id"]] = by.get(o["supplier_id"], 0) + o["qty"]
        c = next(c for c in cands["M01"] if c["supplier_id"] == o["supplier_id"])
        assert o["qty"] <= c["spare_daily"] * max(0, 30 - o["lead_days"]) * (1 + 1e-6) + 1e-6
    assert "S0003" not in by and "S0001" not in by
    rb = rule_based_allocation({"M01": 5000.0}, cands, 30)
    assert all(o["mode"] == "standard" for o in rb)
