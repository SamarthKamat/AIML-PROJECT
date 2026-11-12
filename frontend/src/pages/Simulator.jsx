import React, { useMemo, useState } from 'react'
import { motion } from 'framer-motion'
import { Factory, IndianRupee, Layers, PackageX, Timer, Zap } from 'lucide-react'
import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis, Legend } from 'recharts'
import { api, compactInr, inr, num, pct } from '../api.js'
import { Card, ErrorBox, Kpi, Loader, PageHeader, SERIES, tooltipStyle, useApi } from '../components/ui.jsx'

const TYPES = [['supplier_failure', 'Direct supplier failure'], ['capacity_reduction', 'Capacity reduction'], ['weather', 'Weather (regional)'],
  ['transportation', 'Transportation / port'], ['labour_strike', 'Labour strike'], ['quality_failure', 'Quality failure'], ['financial_distress', 'Financial distress']]

export default function Simulator() {
  const sups = useApi(() => api('/api/suppliers?page_size=200&sort=-exposure_inr'))
  const [f, setF] = useState({ supplier_id: 'S0122', disruption_type: 'supplier_failure', capacity_reduction: 1, duration_days: 45, region: '', policy: 'A', replications: 20 })
  const [res, setRes] = useState(null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState(null)
  const run = async () => {
    setBusy(true); setErr(null)
    const body = { ...f }
    if (f.disruption_type === 'weather' && f.region) delete body.supplier_id
    if (f.disruption_type === 'transportation' && !f.supplier_id) delete body.supplier_id
    if (!body.region) delete body.region
    try { setRes(await api('/api/simulation/run', { method: 'POST', body })) } catch (e) { setErr(e) } finally { setBusy(false) }
  }
  const series = useMemo(() => {
    if (!res) return []
    return res.series.day.map((d, i) => {
      const o = { day: d, service: res.series.service_level[i] }
      Object.entries(res.series.materials).forEach(([m, v]) => { o[m] = v[i] })
      return o
    })
  }, [res])
  const s = res?.summary
  return (
    <>
      <PageHeader title="Disruption Simulator" desc="Daily material-flow cascade over the TRUE synthetic network (including relationships hidden from the manufacturer). Monte Carlo replications use common random numbers." />
      <div className="grid g4" style={{ marginBottom: 18 }}>
        <Card title="Scenario" className="span1">
          <div className="grid" style={{ gap: 12 }}>
            <label className="field">Disruption type<select value={f.disruption_type} onChange={(e) => setF({ ...f, disruption_type: e.target.value })}>{TYPES.map(([k, l]) => <option key={k} value={k}>{l}</option>)}</select></label>
            <label className="field">Supplier (top exposure first)
              <select value={f.supplier_id} onChange={(e) => setF({ ...f, supplier_id: e.target.value })}>
                <option value="">— none (region / port) —</option>
                {sups.data?.items.map((x) => <option key={x.supplier_id} value={x.supplier_id}>{x.supplier_id} · T{x.tier} · {x.supplier_name.replace(' (Synthetic)', '')}</option>)}
              </select></label>
            <label className="field">Region (adds all suppliers in region)<select value={f.region} onChange={(e) => setF({ ...f, region: e.target.value })}>
              <option value="">—</option>{sups.data?.regions.map((r) => <option key={r}>{r}</option>)}</select></label>
            <label className="field">Capacity reduction: <b style={{ color: 'var(--text)' }}>{pct(f.capacity_reduction, 0)}</b>
              <input type="range" min={0.05} max={1} step={0.05} value={f.capacity_reduction} onChange={(e) => setF({ ...f, capacity_reduction: +e.target.value })} /></label>
            <label className="field">Duration: <b style={{ color: 'var(--text)' }}>{f.duration_days} days</b>
              <input type="range" min={5} max={150} step={5} value={f.duration_days} onChange={(e) => setF({ ...f, duration_days: +e.target.value })} /></label>
            <label className="field">Response policy<div className="seg">{[['A', 'None'], ['B', 'Rule-based'], ['C', 'Optimised']].map(([k, l]) =>
              <button key={k} className={f.policy === k ? 'on' : ''} onClick={() => setF({ ...f, policy: k })}>{l}</button>)}</div></label>
            <label className="field">Replications: <b style={{ color: 'var(--text)' }}>{f.replications}</b>
              <input type="range" min={1} max={100} value={f.replications} onChange={(e) => setF({ ...f, replications: +e.target.value })} /></label>
            <button className="btn primary" onClick={run} disabled={busy} style={{ justifyContent: 'center' }}><Zap size={15} />{busy ? 'Simulating…' : 'Run simulation'}</button>
            <ErrorBox error={err} />
          </div>
        </Card>
        <div className="span3 grid" style={{ gap: 18, alignContent: 'start' }}>
          {busy && <Card><Loader text={`Running ${f.replications} Monte Carlo replications…`} /></Card>}
          {!res && !busy && <Card><div className="callout">Choose a disruption and press <b>Run simulation</b>. Try the default: S0122, a Tier-2 cocoa-bean trader with 11 Tier-1 buyers — 6 of them hidden from the manufacturer.</div></Card>}
          {res && !busy && <>
            <div className="grid g4">
              <Kpi label="Financial impact" value={s.financial_impact_inr} format={compactInr} foot={`P97.5 ${compactInr(s['financial_impact_inr_p97.5'])}`} icon={IndianRupee} color="#f87171" />
              <Kpi label="Unmet demand" value={s.unmet_demand_units} format={(v) => num(v)} foot={`${s.unmet_demand_pct.toFixed(2)}% of demand`} icon={PackageX} color="#fbbf24" />
              <Kpi label="Affected suppliers" value={s.affected_suppliers} foot={`${s.affected_materials.toFixed(1)} materials · ${s.affected_production_sites.toFixed(1)} sites`} icon={Layers} color="#22d3ee" delay={0.05} />
              <Kpi label="Recovery time" value={s.recovery_time_days} format={(v) => `${v.toFixed(0)} d`} foot={`min service ${pct(s.min_service_level)}`} icon={Timer} color="#a78bfa" delay={0.1} />
            </div>
            <Card title="Service level and Tier-1 material supply" sub="First replication; supply ratio = arrivals ÷ nominal requirement">
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={series}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="day" tick={{ fontSize: 10 }} /><YAxis tick={{ fontSize: 10 }} domain={[0, 1.2]} />
                  <Tooltip {...tooltipStyle} formatter={(v) => pct(v)} /><Legend />
                  <ReferenceLine y={1} stroke="#8ea3c7" strokeDasharray="3 3" />
                  <Line dataKey="service" name="Service level" stroke="#34d399" strokeWidth={3} dot={false} animationDuration={1500} />
                  {Object.keys(res.series.materials).slice(0, 6).map((m, i) => <Line key={m} dataKey={m} name={`${m} ${res.material_names[m] || ''}`} stroke={SERIES[i % 8]} dot={false} strokeWidth={1.6} animationDuration={1500} />)}
                </LineChart>
              </ResponsiveContainer>
            </Card>
            <div className="grid g2">
              <Card title="Cost breakdown (mean)">
                <dl className="kv">
                  <dt>Lost contribution margin</dt><dd>{inr(s.production_loss_cm_inr)}</dd><dt>Shortage penalty</dt><dd>{inr(s.shortage_penalty_inr)}</dd>
                  <dt>Procurement spend / premium</dt><dd>{inr(s.procurement_spend_inr)} / {inr(s.procurement_premium_inr)}</dd>
                  <dt>Transportation</dt><dd>{inr(s.transportation_cost_inr)}</dd><dt>Holding</dt><dd>{inr(s.holding_cost_inr)}</dd>
                  <dt>Interruption</dt><dd>{inr(s.interruption_cost_inr)}</dd><dt>Recovery management</dt><dd>{inr(s.recovery_management_cost_inr)}</dd>
                  <dt>Full-recovery share</dt><dd>{pct(s.feasibility_full_recovery_share)}</dd>
                </dl>
                <p className="note">Simulated INR on synthetic data under the documented assumptions — not observed company losses.</p>
              </Card>
              <Card title="Most affected suppliers" sub="Minimum availability during horizon">
                <div className="table-wrap" style={{ maxHeight: 250 }}><table><thead><tr><th>Supplier</th><th>Tier</th><th className="num">Min avail.</th></tr></thead>
                  <tbody>{res.affected_suppliers.map((a) => <tr key={a.supplier_id}><td>{a.supplier_name.replace(' (Synthetic)', '')} <span className="dim mono">{a.supplier_id}</span></td><td>T{a.tier}</td><td className="num">{pct(a.min_availability, 0)}</td></tr>)}</tbody></table></div>
              </Card>
            </div>
          </>}
        </div>
      </div>
    </>
  )
}
