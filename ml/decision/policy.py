"""Executable, cost-based action policy.

Action a has an (illustrative, INR) cost C_a and a mitigation fraction r_a = share of
the expected disruption loss it avoids.  Given a CALIBRATED disruption probability p
and the network exposure E (INR), the policy chooses

    a* = argmin_a  C_a + p * (1 - r_a) * E

which is the Bayes-optimal decision when p is calibrated.  The implied probability
break-even between consecutive actions a -> b is p = (C_b - C_a) / (E (r_b - r_a)),
so thresholds come from documented costs, not from arbitrary cut-offs.
All costs are ILLUSTRATIVE ASSUMPTIONS (see documentation/research/ECONOMIC_ASSUMPTIONS.md).
"""
from __future__ import annotations

ACTIONS = {
    "routine_monitoring": {"risk_category": "Low", "cost_inr": 0.0, "mitigation": 0.00,
                           "description": "Routine monitoring of supplier KPIs"},
    "enhanced_monitoring": {"risk_category": "Medium", "cost_inr": 5_000.0, "mitigation": 0.15,
                            "description": "Weekly supplier check-in, early-warning tracking, order-status verification"},
    "contingency_review": {"risk_category": "High", "cost_inr": 20_000.0, "mitigation": 0.45,
                           "description": "Contingency review: confirm alternates, raise safety stock, split next PO"},
    "escalate_recovery": {"risk_category": "Critical", "cost_inr": 60_000.0, "mitigation": 0.70,
                          "description": "Run recovery simulation & optimisation, pre-book alternate capacity, escalate"},
}


def choose_action(p: float, exposure_inr: float) -> dict:
    costs = {a: v["cost_inr"] + p * (1 - v["mitigation"]) * exposure_inr for a, v in ACTIONS.items()}
    best = min(costs, key=costs.get)
    return {"action": best, "risk_category": ACTIONS[best]["risk_category"], "expected_cost_inr": round(costs[best], 2),
            "expected_cost_by_action_inr": {k: round(v, 2) for k, v in costs.items()}}


def break_even_probabilities(exposure_inr: float) -> dict:
    names = list(ACTIONS)
    out = {}
    for a, b in zip(names[:-1], names[1:]):
        dc = ACTIONS[b]["cost_inr"] - ACTIONS[a]["cost_inr"]
        dr = ACTIONS[b]["mitigation"] - ACTIONS[a]["mitigation"]
        out[f"{a}->{b}"] = None if exposure_inr <= 0 else round(dc / (exposure_inr * dr), 6)
    return out
