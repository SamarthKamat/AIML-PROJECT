"""Response policies A (no intervention), B (rule-based) and C (optimisation-based).

Both B and C receive the same disruption report on the detection day (which suppliers, reported
severity, nominal duration) and the same reactive trigger (material cover < 10 days while
regular arrivals are below plan).  They differ in *knowledge* and *decision logic*:
  B: impact estimated on the OBSERVED graph; same-material approved alternates ranked by
     reliability; standard transport; no risk screen.
  C: impact estimated on the observed + PREDICTED (Exp 2A candidate-link) graph; alternates
     screened by calibrated 30-day risk and DERATED by their estimated dependency on the disrupted
     suppliers (excluded only if > 90 %); MILP over approved + substitute suppliers and
     standard/expedited transport.
"""
from __future__ import annotations

import pandas as pd

from ml.optimisation.recovery import build_candidates, material_penalty, optimise_recovery, rule_based_allocation

MANU = "C001"


def reach_from_edges(rel: pd.DataFrame, sup: pd.DataFrame, mats: pd.DataFrame, sources: list[str]) -> dict:
    """Share of manufacturer supply of each Tier-1 material that depends on `sources` (bottleneck rule)."""
    n_inputs = {s: max(1, len(str(mats.loc[m, "input_materials"]).split(";"))) for s, m in sup.primary_material_id.items()}
    reach = {}
    for _, e in rel[rel.buyer_id == MANU].iterrows():
        reach.setdefault(e.supplier_id, {})
        reach[e.supplier_id][e.material_id] = reach[e.supplier_id].get(e.material_id, 0) + e.dependency_pct
    deeper = rel[rel.buyer_id != MANU]
    bt = deeper.buyer_id.map(sup.tier)
    for tier in (1, 2):
        for _, e in deeper[bt == tier].iterrows():
            if e.buyer_id in reach:
                d = reach.setdefault(e.supplier_id, {})
                for m, r in reach[e.buyer_id].items():
                    d[m] = min(1.0, d.get(m, 0) + e.dependency_pct / n_inputs[e.buyer_id] * r)
    total = {}
    for s in sources:
        for m, r in reach.get(s, {}).items():
            total[m] = min(1.0, total.get(m, 0) + r)
    # Tier-1 suppliers whose output depends on the sources (to be excluded as alternates)
    affected_t1 = set()
    for s in sources:
        affected_t1.add(s)
    frontier = set(sources)
    for _ in range(3):
        nxt = set(deeper[deeper.supplier_id.isin(frontier)].buyer_id)
        affected_t1 |= nxt
        frontier = nxt
    return {"reach": total, "affected": affected_t1}


def dependency_on_sources(rel: pd.DataFrame, sup: pd.DataFrame, mats: pd.DataFrame, sources: set, depth: int = 3) -> dict:
    """Estimated fraction of each supplier's output that depends on `sources` (bottleneck rule, shares normalised
    per buyer-material over the KNOWN graph, which may include predicted edges)."""
    n_inputs = {s: max(1, len(str(mats.loc[m, "input_materials"]).split(";"))) for s, m in sup.primary_material_id.items()}
    inner = rel[rel.buyer_id != MANU].copy()
    inner["w"] = inner.dependency_pct / inner.groupby(["buyer_id", "material_id"]).dependency_pct.transform("sum")
    dep = {s: 1.0 for s in sources}
    for _ in range(depth):
        new = dict(dep)
        for b, g in inner.groupby("buyer_id"):
            if b in sources:
                continue
            v = sum(w * dep.get(u, 0.0) for u, w in zip(g.supplier_id, g.w)) / n_inputs.get(b, 1)
            if v > 0:
                new[b] = min(1.0, v)
        dep = new
    return dep


class NoIntervention:
    name = "A: No intervention"

    def on_day(self, t, state):
        return []


class _Base:
    def __init__(self, net, knowledge_rel, reported, A, risk: pd.Series, subs=None):
        self.net, self.A, self.risk, self.subs = net, A, risk, subs
        self.reported = reported            # dict(supplier_ids, severity)
        self.d0 = A["detection_day"]
        self.window = A["nominal_disruption_duration_days"] + 15
        self.knowledge = knowledge_rel
        self.est = reach_from_edges(knowledge_rel, net.sup, net.mats, reported["supplier_ids"])
        self.reacted = set()
        self.decisions = []

    def initial_deficits(self, sim):
        sev = self.reported["severity"]
        dur = self.A["nominal_disruption_duration_days"]
        tiers = self.net.sup.loc[self.reported["supplier_ids"], "tier"]
        lost_days = max(0.0, dur - 10.0 * (float(tiers.min()) - 1))
        out = {}
        for m, r in self.est["reach"].items():
            if m not in sim.mi:
                continue
            Rm = sim.R[sim.mi[m]]
            buffer = max(0.0, sim.I0[sim.mi[m]] - self.A["safety_cover_days"] * Rm)
            need = r * sev * Rm * lost_days - buffer
            if need > 0:
                out[m] = need
        return out

    def reactive_deficits(self, t, state, sim):
        out = {}
        end = self.d0 + self.A["nominal_disruption_duration_days"]
        for m, j in sim.mi.items():
            Rm = state["R"][j]
            if Rm <= 0 or m in self.reacted:
                continue
            cover = state["I"][j] / Rm
            if cover < self.A["reactive_trigger_cover_days"] and state["reg_ratio"][j] < 0.98:
                remaining = max(10, end - t + 10)
                out[m] = (1 - state["reg_ratio"][j]) * Rm * remaining
                self.reacted.add(m)
        return out

    def candidates(self, mats_needed, exclude):
        return {m: build_candidates(self.net, m, exclude, self.risk, self.A, self.subs) for m in mats_needed}


class RuleBased(_Base):
    name = "B: Rule-based"

    def on_day(self, t, state):
        sim = state["sim"]
        d = {}
        if t == self.d0:
            d = self.initial_deficits(sim)
        elif t > self.d0:
            d = self.reactive_deficits(t, state, sim)
        if not d:
            return []
        exclude = set(self.reported["supplier_ids"]) | self.est["affected"]
        orders = rule_based_allocation(d, self.candidates(d, exclude), self.window)
        self.decisions.append(dict(day=t, deficits=d, n_orders=len(orders)))
        return orders


class Optimised(_Base):
    name = "C: Optimisation"

    def on_day(self, t, state):
        sim = state["sim"]
        d = {}
        if t == self.d0:
            d = self.initial_deficits(sim)
        elif t > self.d0:
            d = self.reactive_deficits(t, state, sim)
        if not d:
            return []
        # quantitative derating instead of hard exclusion: expected availability of each alternate
        if not hasattr(self, "dep"):
            self.dep = dependency_on_sources(self.knowledge, self.net.sup, self.net.mats, set(self.reported["supplier_ids"]))
        sev = self.reported["severity"]
        exclude = set(self.reported["supplier_ids"]) | {j for j, v in self.dep.items() if sev * v > 0.9}
        cands = self.candidates(d, exclude)
        for cs in cands.values():
            for c in cs:
                c["avail"] = 1.0 - sev * self.dep.get(c["supplier_id"], 0.0)
        pen = material_penalty(self.net, self.A)
        orders, info = optimise_recovery(d, cands, self.window, pen, self.A)
        self.decisions.append(dict(day=t, deficits=d, n_orders=len(orders), solver=info))
        return orders
