import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { Wrench } from 'lucide-react'
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { api, compactInr, inr, num, pct } from '../api.js'
import { Card, ErrorBox, Loader, PageHeader, tooltipStyle, useApi } from '../components/ui.jsx'

export default function Recovery() {
  const mats = useApi(() => api('/api/materials'))
  const [f, setF] = useState({ material_id: 'M01', quantity: 8000, deadline_days: 30, exclude: 'S0001,S0002' })
  const [r, setR] = useState(null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState(null)
  const run = async () => {
    setBusy(true); setErr(null)
    try {
      setR(await api('/api/recovery/optimise', { method: 'POST', body: { material_id: f.material_id, quantity: +f.quantity, deadline_days: +f.deadline_days,
        exclude_supplier_ids: f.exclude.split(',').map((s) => s.trim().toUpperCase()).filter(Boolean) } }))
    } catch (e) { setErr(e) } finally { setBusy(false) }
  }
  const t1 = (mats.data || []).filter((m) => m.bom_level === 1)
  const cmp = r ? [{ k: 'Total cost', MILP: r.total_cost_inr, Rule: r.rule_based?.total_cost_inr }, { k: 'Expected delivered', MILP: r.expected_delivered, Rule: r.rule_based?.expected_delivered }] : []
  return (
    <>
      <PageHeader title="Recovery Planner" desc="Mixed-integer recovery optimisation (PuLP/CBC): approved and substitute suppliers, standard vs. expedited transport, capacity before deadline, allocation-share and calibrated-risk limits, expected-yield demand cover." />
      <div className="grid g4" style={{ marginBottom: 18 }}>
        <Card title="Requirement">
          <div className="grid" style={{ gap: 12 }}>
            <label className="field">Material<select value={f.material_id} onChange={(e) => setF({ ...f, material_id: e.target.value })}>{t1.map((m) => <option key={m.material_id} value={m.material_id}>{m.material_id} · {m.material_name}</option>)}</select></label>
            <label className="field">Required quantity<input type="number" value={f.quantity} onChange={(e) => setF({ ...f, quantity: e.target.value })} /></label>
            <label className="field">Delivery deadline (days)<input type="number" value={f.deadline_days} onChange={(e) => setF({ ...f, deadline_days: e.target.value })} /></label>
            <label className="field">Exclude suppliers (disrupted)<input value={f.exclude} onChange={(e) => setF({ ...f, exclude: e.target.value })} /></label>
            <button className="btn primary" onClick={run} disabled={busy} style={{ justifyContent: 'center' }}><Wrench size={15} />{busy ? 'Solving…' : 'Optimise recovery'}</button>
            <ErrorBox error={err} />
          </div>
        </Card>
        <div className="span3 grid" style={{ gap: 18, alignContent: 'start' }}>
          {busy && <Card><Loader text="Solving MILP…" /></Card>}
          {!r && !busy && <Card><div className="callout">Specify a shortfall and press <b>Optimise recovery</b>. The optimiser is compared with the rule-based policy (same material only, most reliable first, standard transport).</div></Card>}
          {r && !busy && <motion.div className="grid" style={{ gap: 18 }} initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
            <div className="grid g3">
              <Card title="Solver"><div className="kpi" style={{ padding: 0 }}><div className="value">{r.solver.status}</div><div className="foot">{r.solver.n_variables} variables · {r.solver.n_constraints} constraints</div></div></Card>
              <Card title="MILP cost"><div className="kpi" style={{ padding: 0 }}><div className="value">{compactInr(r.total_cost_inr)}</div><div className="foot">expected delivered {num(r.expected_delivered)} of {num(r.quantity)}</div></div></Card>
              <Card title="Expected unmet"><div className="kpi" style={{ padding: 0 }}><div className="value">{num(r.solver.expected_shortage?.[r.material_id] || 0)}</div><div className="foot">penalised at lost margin per unit</div></div></Card>
            </div>
            <div className="grid g2">{cmp.map((c) => (
              <Card key={c.k} title={`MILP vs. rule-based · ${c.k}`} sub={c.k === 'Total cost' ? 'INR, incl. freight & onboarding' : 'expected units after yield, risk and compatibility'}>
                <ResponsiveContainer width="100%" height={150}>
                  <BarChart data={[{ n: 'MILP', v: c.MILP }, { n: 'Rule-based', v: c.Rule }]} layout="vertical"><CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" tick={{ fontSize: 10 }} tickFormatter={c.k === 'Total cost' ? compactInr : (v) => num(v)} /><YAxis type="category" dataKey="n" width={90} />
                    <Tooltip {...tooltipStyle} formatter={(v) => (c.k === 'Total cost' ? inr(v) : num(v))} />
                    <Bar dataKey="v" radius={[0, 6, 6, 0]} animationDuration={900}><Cell fill="#34d399" /><Cell fill="#fbbf24" /></Bar></BarChart>
                </ResponsiveContainer>
              </Card>))}</div>
            <Card title="Recommended allocation">
              <div className="table-wrap"><table><thead><tr><th>Supplier</th><th>Supplies</th><th className="num">Quantity</th><th>Mode</th><th className="num">Lead</th><th className="num">Unit price</th><th className="num">Freight</th><th className="num">Fixed</th></tr></thead>
                <tbody>{r.allocation.map((a, i) => <tr key={i}><td className="mono">{a.supplier_id}</td><td>{a.supplies_material}</td><td className="num">{num(a.qty)}</td><td>{a.mode}</td><td className="num">{a.lead_days} d</td>
                  <td className="num">{inr(a.unit_price, 1)}</td><td className="num">{pct(a.freight_rate)}</td><td className="num">{inr(a.fixed_cost)}</td></tr>)}</tbody></table></div>
            </Card>
            <Card title="Candidate suppliers" sub="Material compatibility, spare capacity, price, lead time, reliability, calibrated risk">
              <div className="table-wrap" style={{ maxHeight: 320 }}><table><thead><tr><th>Supplier</th><th>Supplies</th><th className="num">Compat.</th><th className="num">Spare/day</th><th className="num">Price</th>
                <th className="num">Lead std/exp</th><th className="num">Reliability</th><th className="num">Risk p30</th><th>Onboarding</th></tr></thead>
                <tbody>{r.candidates.map((c) => <tr key={c.supplier_id} style={c.risk_p30 > 0.35 ? { opacity: 0.5 } : {}}><td>{c.supplier_name.replace(' (Synthetic)', '')} <span className="dim mono">{c.supplier_id}</span></td><td>{c.supplies_material}</td>
                  <td className="num">{c.compat}</td><td className="num">{num(c.spare_daily, 1)}</td><td className="num">{inr(c.unit_price, 1)}</td><td className="num">{c.lead_standard}/{c.lead_expedited} d</td>
                  <td className="num">{c.reliability.toFixed(3)}</td><td className="num">{pct(c.risk_p30)}</td><td>{c.onboarding ? 'yes' : '—'}</td></tr>)}</tbody></table></div>
            </Card>
          </motion.div>}
        </div>
      </div>
    </>
  )
}
