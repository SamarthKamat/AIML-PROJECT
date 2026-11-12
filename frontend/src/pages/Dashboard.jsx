import React from 'react'
import { useNavigate } from 'react-router-dom'
import { AlertTriangle, Boxes, Factory, IndianRupee, Layers, PackageX, ShieldAlert, Waypoints } from 'lucide-react'
import { Area, AreaChart, Bar, BarChart, CartesianGrid, Cell, Pie, PieChart, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis, Legend } from 'recharts'
import { api, compactInr, num, pct } from '../api.js'
import { BAND_COLORS, Badge, Card, ErrorBox, Kpi, Loader, PageHeader, RiskBar, SERIES, tooltipStyle, useApi } from '../components/ui.jsx'

export default function Dashboard() {
  const { data: d, error, loading } = useApi(() => api('/api/dashboard/summary'))
  const nav = useNavigate()
  if (loading) return <><PageHeader title="Executive Dashboard" /><Loader text="Scoring 500 suppliers with the calibrated model…" /></>
  if (error) return <><PageHeader title="Executive Dashboard" /><ErrorBox error={error} /></>
  const t = d.totals
  const dist = Object.entries(d.risk_distribution).map(([k, v]) => ({ name: k, value: v }))
  const types = Array.from(new Set(d.disruption_trend.flatMap((r) => Object.keys(r).filter((k) => k !== 'month'))))
  const kp = [
    ['Total suppliers', t.suppliers, null, `T1 ${t.tier1} · T2 ${t.tier2} · T3 ${t.tier3}`, Boxes, '#4f8dff'],
    ['Elevated-risk suppliers', t.high_risk_suppliers, null, 'calibrated p(30d) ≥ 15%', ShieldAlert, '#f87171'],
    ['Critical dependencies', t.critical_dependencies, null, 'Tier-1 materials with HHI > 0.4', Waypoints, '#fbbf24'],
    ['Active disruptions', t.active_disruptions, null, `reported before ${d.as_of}`, AlertTriangle, '#f43f5e'],
    ['Low-cover materials', t.low_cover_materials, null, '< 14 days of inventory', PackageX, '#a78bfa'],
    ['Expected 30-day loss', t.expected_30d_loss_inr, compactInr, 'Σ p × network exposure (simulated)', IndianRupee, '#22d3ee'],
  ]
  return (
    <>
      <PageHeader title="Executive Dashboard"
        desc={`Network-wide risk posture as of ${d.as_of}. Every figure is computed live from the database and the calibrated XGBoost model.`} />
      <div className="grid g6" style={{ marginBottom: 18 }}>
        {kp.map(([l, v, f, foot, I, c], i) => <Kpi key={l} label={l} value={v} format={f || undefined} foot={foot} icon={I} color={c} delay={i * 0.06} />)}
      </div>
      <div className="grid g3" style={{ marginBottom: 18 }}>
        <Card title="Supplier risk distribution" sub="Calibrated 30-day probability bands" delay={0.1}>
          <ResponsiveContainer width="100%" height={230}>
            <PieChart>
              <Pie data={dist} dataKey="value" nameKey="name" innerRadius={62} outerRadius={92} paddingAngle={3} stroke="none" animationDuration={1200}>
                {dist.map((e) => <Cell key={e.name} fill={BAND_COLORS[e.name]} />)}
              </Pie>
              <Tooltip {...tooltipStyle} />
              <Legend verticalAlign="bottom" iconType="circle" />
            </PieChart>
          </ResponsiveContainer>
        </Card>
        <Card title="Disruption trend" sub="Monthly disruption starts by type (history before as-of date)" className="span2" delay={0.15}>
          <ResponsiveContainer width="100%" height={230}>
            <AreaChart data={d.disruption_trend}>
              <defs>{types.map((k, i) => (
                <linearGradient key={k} id={`g${i}`} x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor={SERIES[i % 8]} stopOpacity={0.6} /><stop offset="1" stopColor={SERIES[i % 8]} stopOpacity={0.02} /></linearGradient>
              ))}</defs>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="month" tick={{ fontSize: 10 }} interval={2} />
              <YAxis tick={{ fontSize: 10 }} />
              <Tooltip {...tooltipStyle} />
              {types.map((k, i) => <Area key={k} type="monotone" dataKey={k} stackId="1" stroke={SERIES[i % 8]} fill={`url(#g${i})`} animationDuration={1400} />)}
            </AreaChart>
          </ResponsiveContainer>
        </Card>
      </div>
      <div className="grid g3" style={{ marginBottom: 18 }}>
        <Card title="Geographic risk" sub="Bubble = number of suppliers, colour = mean calibrated risk" className="span2" delay={0.2}>
          <ResponsiveContainer width="100%" height={300}>
            <ScatterChart margin={{ top: 10, right: 10, bottom: 10, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" dataKey="longitude" name="Longitude" domain={[-20, 180]} tick={{ fontSize: 10 }} />
              <YAxis type="number" dataKey="latitude" name="Latitude" domain={[-45, 55]} tick={{ fontSize: 10 }} />
              <ZAxis type="number" dataKey="suppliers" range={[60, 900]} />
              <Tooltip {...tooltipStyle} cursor={{ strokeDasharray: '3 3' }}
                content={({ payload }) => payload && payload[0] ? (
                  <div className="card" style={{ padding: 10 }}><b>{payload[0].payload.region}</b><div className="small muted">
                    {payload[0].payload.suppliers} suppliers · mean risk {pct(payload[0].payload.mean_risk)} · exposure {compactInr(payload[0].payload.exposure)}</div></div>) : null} />
              <Scatter data={d.geographic_risk} animationDuration={1200}>
                {d.geographic_risk.map((g) => <Cell key={g.region} fill={g.mean_risk > 0.12 ? '#f87171' : g.mean_risk > 0.07 ? '#fbbf24' : '#34d399'} fillOpacity={0.75} />)}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
        </Card>
        <Card title="Inventory exposure" sub="Days of cover, lowest Tier-1 materials" delay={0.25}>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={d.inventory_exposure} layout="vertical" margin={{ left: 20 }}>
              <CartesianGrid strokeDasharray="3 3" horizontal={false} />
              <XAxis type="number" tick={{ fontSize: 10 }} />
              <YAxis type="category" dataKey="material_name" width={130} tick={{ fontSize: 10 }} />
              <Tooltip {...tooltipStyle} />
              <Bar dataKey="days_of_cover" radius={[0, 6, 6, 0]} animationDuration={1200}>
                {d.inventory_exposure.map((m) => <Cell key={m.material_id} fill={m.days_of_cover < 10 ? '#f87171' : m.days_of_cover < 14 ? '#fbbf24' : '#4f8dff'} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Card>
      </div>
      <div className="grid g3">
        <Card title="Highest-risk suppliers" sub="Click to open the supplier in the Risk Prediction Centre" className="span2" delay={0.3}>
          <div className="table-wrap"><table>
            <thead><tr><th>Supplier</th><th>Tier</th><th>Region</th><th>Calibrated p(30d)</th><th className="num">Exposure</th><th>Action</th></tr></thead>
            <tbody>{d.top_risk_suppliers.map((s) => (
              <tr key={s.supplier_id} className="clickable" onClick={() => nav(`/risk?supplier=${s.supplier_id}`)}>
                <td><b>{s.supplier_name}</b> <span className="dim mono">{s.supplier_id}</span></td>
                <td><span className="badge b-tier">T{s.tier}</span></td><td>{s.region}</td>
                <td><RiskBar p={s.probability_calibrated} />{pct(s.probability_calibrated)}</td>
                <td className="num">{compactInr(s.exposure_inr)}</td><td><Badge band={s.risk_category} /></td>
              </tr>))}</tbody>
          </table></div>
        </Card>
        <div className="grid" style={{ gap: 18 }}>
          <Card title="Critical dependencies" sub="Concentrated Tier-1 sourcing (observed graph)" delay={0.35}>
            {d.critical_dependencies.slice(0, 6).map((c) => (
              <div key={c.material_id} className="row" style={{ justifyContent: 'space-between', padding: '6px 0', borderBottom: '1px solid rgba(96,148,255,.08)' }}>
                <span><Layers size={13} color="#fbbf24" style={{ verticalAlign: -2 }} /> {c.material_name}</span>
                <span className="small muted">HHI {c.hhi} · {c.suppliers} sup.</span>
              </div>))}
          </Card>
          <Card title="Active disruptions" sub={`${d.active_disruptions.length} ongoing`} delay={0.4}>
            <div style={{ maxHeight: 190, overflowY: 'auto' }}>
              {d.active_disruptions.slice(0, 12).map((a) => (
                <div key={a.disruption_id} className="row" style={{ justifyContent: 'space-between', padding: '6px 0', borderBottom: '1px solid rgba(96,148,255,.08)' }}>
                  <span className="small"><Factory size={12} color="#f43f5e" style={{ verticalAlign: -2 }} /> {a.supplier_name.replace(' (Synthetic)', '')} <span className="dim">T{a.tier}</span></span>
                  <span className="small muted">{a.disruption_type.replaceAll('_', ' ')} · −{pct(a.capacity_loss_pct, 0)}</span>
                </div>))}
            </div>
          </Card>
        </div>
      </div>
      <p className="note" style={{ marginTop: 16 }}>Observed relationships: {num(t.observed_relationships)} · Total network exposure {compactInr(t.total_exposure_inr)} (simulated INR, synthetic data).</p>
    </>
  )
}
