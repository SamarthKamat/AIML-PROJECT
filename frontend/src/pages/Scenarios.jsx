import React, { useState } from 'react'
import { RefreshCw } from 'lucide-react'
import { Bar, BarChart, CartesianGrid, Cell, ErrorBar, Legend, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis } from 'recharts'
import { api, compactInr, inr, num, pct } from '../api.js'
import { Card, ErrorBox, Loader, PageHeader, POLICY_COLORS, tooltipStyle, useApi } from '../components/ui.jsx'

const METRICS = [
  ['financial_impact_inr', 'Financial impact', compactInr], ['unmet_demand_units', 'Unmet demand (units)', (v) => num(v)],
  ['recovery_time_days', 'Recovery time (days)', (v) => v.toFixed(1)], ['procurement_spend_inr', 'Procurement spend', compactInr],
  ['inventory_shortage_value_inr', 'Inventory shortage value', compactInr], ['transportation_cost_inr', 'Transportation cost', compactInr],
  ['min_service_level', 'Minimum service level', (v) => pct(v)], ['feasibility_full_recovery_share', 'Full-recovery share', (v) => pct(v, 0)],
]

function grouped(rows, key) {
  const sev = ['Mild', 'Moderate', 'Severe']
  return sev.map((s) => {
    const o = { severity: s }
    rows.filter((r) => r.severity === s).forEach((r) => {
      o[r.policy] = r[key]
      if (r[key + '_p2.5'] !== undefined) o[r.policy + '_err'] = [r[key] - r[key + '_p2.5'], r[key + '_p97.5'] - r[key]]
    })
    return o
  })
}

export default function Scenarios() {
  const { data, error, loading } = useApi(() => api('/api/simulation/scenarios'))
  const [metric, setMetric] = useState('financial_impact_inr')
  const [rerun, setRerun] = useState(null)
  const [busy, setBusy] = useState(false)
  if (loading) return <><PageHeader title="Scenario Comparison" /><Loader /></>
  if (error) return <><PageHeader title="Scenario Comparison" /><ErrorBox error={error} /></>
  if (!data.available) return <><PageHeader title="Scenario Comparison" /><div className="callout warn">Run <code>python -m ml.simulation.scenarios</code> to generate the nine-scenario results.</div></>
  const rows = data.results
  const m = METRICS.find((x) => x[0] === metric)
  const g = grouped(rows, metric)
  const pols = ['A: No intervention', 'B: Rule-based', 'C: Optimisation']
  const doRerun = async () => {
    setBusy(true)
    try { setRerun(await api('/api/simulation/scenarios', { method: 'POST', body: { replications: 20 } })) } finally { setBusy(false) }
  }
  const A = data.assumptions
  return (
    <>
      <PageHeader title="Nine-Scenario Business Evaluation"
        desc={`3 disruption severities × 3 response policies. Shock: the five largest Tier-2 cocoa-bean traders lose 25 / 50 / 100 % capacity for a nominal ${A.nominal_disruption_duration_days} days (log-normal duration). ${rows[0].replications} common-random-number replications per scenario; identical network, demand, inventory and economics across policies.`}>
        <button className="btn small" onClick={doRerun} disabled={busy}><RefreshCw size={13} />{busy ? 'Reproducing…' : 'Reproduce (20 reps)'}</button>
      </PageHeader>
      <Card title="Scenario matrix" sub="Means over replications · colour scales within each column" style={{ marginBottom: 18 }}>
        <div className="table-wrap heat"><table>
          <thead><tr><th>Scenario</th><th>Severity</th><th>Policy</th><th className="num">Capacity avail.</th><th className="num">Affected sup.</th><th className="num">Affected mat.</th>
            <th className="num">Inv. shortage</th><th className="num">Unmet units</th><th className="num">Lost margin</th><th className="num">Procurement</th><th className="num">Transport</th>
            <th className="num">Financial impact (95% PI)</th><th className="num">Recovery d</th><th className="num">Alt. util.</th><th className="num">Full recovery</th></tr></thead>
          <tbody>{rows.map((r) => {
            const sameSev = rows.filter((x) => x.severity === r.severity)
            const best = Math.min(...sameSev.map((x) => x.financial_impact_inr))
            return (
              <tr key={r.scenario} style={r.financial_impact_inr === best ? { background: 'rgba(52,211,153,0.07)' } : {}}>
                <td><b>{r.scenario}</b></td><td>{r.severity}</td><td><span className="legend-dot" style={{ background: POLICY_COLORS[r.policy] }} />{r.policy}</td>
                <td className="num">{pct(r.disrupted_capacity_available, 0)}</td><td className="num">{r.affected_suppliers.toFixed(1)}</td><td className="num">{r.affected_materials.toFixed(1)}</td>
                <td className="num">{compactInr(r.inventory_shortage_value_inr)}</td><td className="num">{num(r.unmet_demand_units)}</td><td className="num">{compactInr(r.production_loss_cm_inr)}</td>
                <td className="num">{compactInr(r.procurement_spend_inr)}</td><td className="num">{compactInr(r.transportation_cost_inr)}</td>
                <td className="num"><b>{compactInr(r.financial_impact_inr)}</b> <span className="dim">[{compactInr(r['financial_impact_inr_p2.5'])}–{compactInr(r['financial_impact_inr_p97.5'])}]</span></td>
                <td className="num">{r.recovery_time_days.toFixed(1)}</td><td className="num">{pct(r.alt_supplier_utilisation, 0)}</td><td className="num">{pct(r.feasibility_full_recovery_share, 0)}</td>
              </tr>)
          })}</tbody>
        </table></div>
      </Card>
      <div className="grid g3" style={{ marginBottom: 18 }}>
        <Card title={m[1]} sub="Mean with 95% percentile interval where available" className="span2"
          right={<select value={metric} onChange={(e) => setMetric(e.target.value)} style={{ width: 230 }}>{METRICS.map(([k, l]) => <option key={k} value={k}>{l}</option>)}</select>}>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={g}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="severity" /><YAxis tick={{ fontSize: 10 }} tickFormatter={m[2]} />
              <Tooltip {...tooltipStyle} formatter={(v) => (Array.isArray(v) ? '' : m[2](v))} /><Legend />
              {pols.map((p) => <Bar key={p} dataKey={p} fill={POLICY_COLORS[p]} radius={[6, 6, 0, 0]} animationDuration={1200}>
                {g[0][p + '_err'] && <ErrorBar dataKey={p + '_err'} width={4} stroke="#e7eefb" strokeWidth={1} />}</Bar>)}
            </BarChart>
          </ResponsiveContainer>
        </Card>
        <Card title="Cost vs. risk" sub="Procurement spend vs. unmet demand (bubble = financial impact)">
          <ResponsiveContainer width="100%" height={320}>
            <ScatterChart><CartesianGrid strokeDasharray="3 3" /><XAxis type="number" dataKey="procurement_spend_inr" name="Procurement" tickFormatter={compactInr} tick={{ fontSize: 10 }} />
              <YAxis type="number" dataKey="unmet_demand_units" name="Unmet" tickFormatter={(v) => num(v)} tick={{ fontSize: 10 }} />
              <ZAxis type="number" dataKey="financial_impact_inr" range={[80, 900]} />
              <Tooltip {...tooltipStyle} content={({ payload }) => payload?.[0] ? <div className="card" style={{ padding: 10 }}><b>{payload[0].payload.scenario}</b> {payload[0].payload.severity} · {payload[0].payload.policy}<div className="small muted">impact {compactInr(payload[0].payload.financial_impact_inr)}</div></div> : null} />
              {pols.map((p) => <Scatter key={p} name={p} data={rows.filter((r) => r.policy === p)} fill={POLICY_COLORS[p]} fillOpacity={0.8} />)}
              <Legend />
            </ScatterChart>
          </ResponsiveContainer>
        </Card>
      </div>
      <div className="grid g2">
        <Card title="Paired policy comparison" sub="Mean saving in financial impact (common random numbers), 95% CI">
          <div className="table-wrap"><table><thead><tr><th>Severity</th><th>Comparison</th><th className="num">Mean saving</th><th className="num">95% CI</th><th className="num">Reps 2nd better</th></tr></thead>
            <tbody>{data.paired.map((r, i) => <tr key={i}><td>{r.severity}</td><td className="small">{r.comparison}</td><td className="num"><b style={{ color: r.mean_saving_inr > 0 ? '#34d399' : r.mean_saving_inr < 0 ? '#f87171' : '' }}>{compactInr(r.mean_saving_inr)}</b></td>
              <td className="num small">{compactInr(r.ci95_low)} – {compactInr(r.ci95_high)}</td><td className="num">{pct(r.share_reps_b_better, 0)}</td></tr>)}</tbody></table></div>
        </Card>
        <Card title="Economic assumptions (illustrative, INR)" sub={A.status}>
          <dl className="kv" style={{ fontSize: 12 }}>
            {Object.entries(A).filter(([k]) => !['status', 'currency', 'base_material_prices'].includes(k)).map(([k, v]) => <React.Fragment key={k}><dt>{k.replaceAll('_', ' ')}</dt><dd>{typeof v === 'number' && v > 1000 ? inr(v) : String(v)}</dd></React.Fragment>)}
          </dl>
        </Card>
      </div>
      {rerun && <Card title={`Reproduction run (${rerun.replications} replications)`} sub="Fewer replications → wider uncertainty; ordering should match the stored 500-rep results" style={{ marginTop: 18 }}>
        <div className="table-wrap"><table><thead><tr><th>Scenario</th><th>Severity</th><th>Policy</th><th className="num">Impact</th><th className="num">Unmet</th></tr></thead>
          <tbody>{rerun.results.map((r) => <tr key={r.scenario}><td>{r.scenario}</td><td>{r.severity}</td><td>{r.policy}</td><td className="num">{inr(r.financial_impact_inr)}</td><td className="num">{num(r.unmet_demand_units)}</td></tr>)}</tbody></table></div>
      </Card>}
    </>
  )
}
