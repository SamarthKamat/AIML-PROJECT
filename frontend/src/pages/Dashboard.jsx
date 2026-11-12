import React, { useMemo } from 'react'
import { useNavigate } from 'react-router-dom'
import { AlertTriangle, Boxes, Factory, IndianRupee, Layers, PackageX, ShieldAlert, Waypoints } from 'lucide-react'
import { Bar, BarChart, CartesianGrid, Cell, Legend, Pie, PieChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { api, compactInr, num, pct } from '../api.js'
import { BAND_COLORS, Badge, Card, ErrorBox, Kpi, Loader, PageHeader, RiskBar, SERIES, tooltipStyle, useApi } from '../components/ui.jsx'

const clean = (s) => (s || '').replace(' (Synthetic)', '')
const nice = (s) => (s || '').replaceAll('_', ' ')
const lerp = (a, b, t) => Math.round(a + (b - a) * t)
const ramp = (t) => { // green -> amber -> red relative scale
  const [c1, c2] = t < 0.5 ? [[52, 211, 153], [251, 191, 36]] : [[251, 191, 36], [248, 113, 113]]
  const u = t < 0.5 ? t * 2 : (t - 0.5) * 2
  return `rgb(${lerp(c1[0], c2[0], u)},${lerp(c1[1], c2[1], u)},${lerp(c1[2], c2[2], u)})`
}
const ACTION = { routine_monitoring: ['Monitor', 'Low'], enhanced_monitoring: ['Enhanced monitoring', 'Medium'],
  contingency_review: ['Contingency review', 'High'], escalate_recovery: ['Escalate & recover', 'Critical'] }

export default function Dashboard() {
  const { data: d, error, loading } = useApi(() => api('/api/dashboard/summary'))
  const nav = useNavigate()

  const trend = useMemo(() => {
    if (!d) return { rows: [], keys: [] }
    const totals = {}
    d.disruption_trend.forEach((r) => Object.entries(r).forEach(([k, v]) => { if (k !== 'month') totals[k] = (totals[k] || 0) + v }))
    const keys = Object.keys(totals).sort((a, b) => totals[b] - totals[a])
    const top = keys.slice(0, 5)
    const rows = d.disruption_trend.map((r) => {
      const o = { month: r.month }
      let other = 0
      keys.forEach((k) => { if (top.includes(k)) o[nice(k)] = r[k] || 0; else other += r[k] || 0 })
      if (keys.length > 5) o.other = other
      return o
    })
    return { rows, keys: top.map(nice).concat(keys.length > 5 ? ['other'] : []) }
  }, [d])

  if (loading) return <><PageHeader title="Executive Dashboard" /><Loader text="Scoring 500 suppliers with the calibrated model…" /></>
  if (error) return <><PageHeader title="Executive Dashboard" /><ErrorBox error={error} /></>
  const t = d.totals
  const dist = ['Low', 'Medium', 'High', 'Critical'].map((k) => ({ name: k, value: d.risk_distribution[k] || 0 }))
  const regions = [...d.geographic_risk].sort((a, b) => b.mean_risk - a.mean_risk).slice(0, 12)
  const kp = [
    ['Total suppliers', t.suppliers, null, `Tier 1 ${t.tier1} · Tier 2 ${t.tier2} · Tier 3 ${t.tier3}`, Boxes, '#4f8dff'],
    ['Elevated risk', t.high_risk_suppliers, null, 'calibrated 30-day p ≥ 15 %', ShieldAlert, '#f87171'],
    ['Critical dependencies', t.critical_dependencies, null, 'Tier-1 materials, HHI > 0.4', Waypoints, '#fbbf24'],
    ['Active disruptions', t.active_disruptions, null, `reported before ${d.as_of}`, AlertTriangle, '#f43f5e'],
    ['Low-cover materials', t.low_cover_materials, null, 'under 14 days of stock', PackageX, '#a78bfa'],
    ['Expected 30-day loss', t.expected_30d_loss_inr, compactInr, 'Σ p × exposure (simulated)', IndianRupee, '#22d3ee'],
  ]
  return (
    <>
      <PageHeader title="Executive Dashboard"
        desc={`Network-wide risk posture as of ${d.as_of}. Every figure is computed live from the database and the calibrated XGBoost model.`} />

      <div className="grid g6" style={{ marginBottom: 18 }}>
        {kp.map(([l, v, f, foot, I, c], i) => <Kpi key={l} label={l} value={v} format={f || undefined} foot={foot} icon={I} color={c} delay={i * 0.05} />)}
      </div>

      <div className="grid g3" style={{ marginBottom: 18 }}>
        <Card title="Supplier risk distribution" sub="Calibrated 30-day disruption probability bands" delay={0.1}>
          <div style={{ position: 'relative', height: 210 }}>
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie data={dist.filter((x) => x.value > 0)} dataKey="value" nameKey="name" innerRadius={66} outerRadius={92} paddingAngle={2} stroke="none" animationDuration={1200}>
                  {dist.filter((x) => x.value > 0).map((e) => <Cell key={e.name} fill={BAND_COLORS[e.name]} />)}
                </Pie>
                <Tooltip {...tooltipStyle} />
              </PieChart>
            </ResponsiveContainer>
            <div className="donut-center"><div><b>{num(t.suppliers)}</b><span className="small muted">suppliers</span></div></div>
          </div>
          <div className="band-legend">
            {dist.map((x) => <span key={x.name} className="muted"><span className="legend-dot" style={{ background: BAND_COLORS[x.name] }} />{x.name} <span className="dim">{x.name === 'Low' ? '< 7 %' : x.name === 'Medium' ? '7–15 %' : x.name === 'High' ? '15–30 %' : '≥ 30 %'}</span><b>{x.value}</b></span>)}
          </div>
        </Card>
        <Card title="Disruption trend" sub="Monthly disruption starts by type (records reported before the as-of date)" className="span2" delay={0.15}>
          <ResponsiveContainer width="100%" height={268}>
            <BarChart data={trend.rows} margin={{ top: 4, right: 8, left: -12, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="month" tick={{ fontSize: 10 }} interval={2} />
              <YAxis tick={{ fontSize: 10 }} allowDecimals={false} />
              <Tooltip {...tooltipStyle} cursor={{ fill: 'rgba(79,141,255,0.08)' }} />
              <Legend iconType="circle" wrapperStyle={{ fontSize: 12 }} />
              {trend.keys.map((k, i) => <Bar key={k} dataKey={k} stackId="a" fill={k === 'other' ? '#5d7299' : SERIES[i % 8]} animationDuration={1100}
                radius={i === trend.keys.length - 1 ? [3, 3, 0, 0] : 0} />)}
            </BarChart>
          </ResponsiveContainer>
        </Card>
      </div>

      <div className="grid g3" style={{ marginBottom: 18 }}>
        <Card title="Risk by sourcing region" sub="Mean calibrated 30-day risk of suppliers per region (top 12; colour relative to range) · hover for supplier count and exposure" className="span2" delay={0.2}>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={regions} layout="vertical" margin={{ top: 0, right: 48, left: 4, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" horizontal={false} />
              <XAxis type="number" tickFormatter={(v) => pct(v, 0)} tick={{ fontSize: 10 }} />
              <YAxis type="category" dataKey="region" width={120} tick={{ fontSize: 11 }} />
              <Tooltip {...tooltipStyle} cursor={{ fill: 'rgba(79,141,255,0.08)' }}
                formatter={(v, n, p) => [`${pct(v)} · ${p.payload.suppliers} suppliers · exposure ${compactInr(p.payload.exposure)}`, 'mean risk']} />
              <Bar dataKey="mean_risk" radius={[0, 6, 6, 0]} animationDuration={1100}
                label={{ position: 'right', fill: '#8ea3c7', fontSize: 11, formatter: (v) => pct(v) }}>
                {regions.map((r) => { const lo = Math.min(...regions.map((x) => x.mean_risk)), hi = Math.max(...regions.map((x) => x.mean_risk)); return <Cell key={r.region} fill={ramp(hi > lo ? (r.mean_risk - lo) / (hi - lo) : 0)} /> })}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Card>
        <Card title="Inventory exposure" sub="Days of cover — lowest Tier-1 materials (dashed = 14-day policy)" delay={0.25}>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={d.inventory_exposure} layout="vertical" margin={{ top: 0, right: 30, left: 4, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" horizontal={false} />
              <XAxis type="number" tick={{ fontSize: 10 }} />
              <YAxis type="category" dataKey="material_name" width={150} tick={{ fontSize: 10.5 }} />
              <ReferenceLine x={14} stroke="#fbbf24" strokeDasharray="4 3" />
              <Tooltip {...tooltipStyle} cursor={{ fill: 'rgba(79,141,255,0.08)' }} formatter={(v) => [`${v} days`, 'cover']} />
              <Bar dataKey="days_of_cover" radius={[0, 6, 6, 0]} animationDuration={1100} label={{ position: 'right', fill: '#8ea3c7', fontSize: 10.5 }}>
                {d.inventory_exposure.map((m) => <Cell key={m.material_id} fill={m.days_of_cover < 10 ? '#f87171' : m.days_of_cover < 14 ? '#fbbf24' : '#4f8dff'} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Card>
      </div>

      <div className="grid g3">
        <Card title="Highest-risk suppliers" sub="Click a row to open it in the Risk Prediction Centre" className="span2" delay={0.3}>
          <div className="table-wrap"><table>
            <thead><tr><th>Supplier</th><th>Tier</th><th>Region</th><th>Calibrated 30-day risk</th><th className="num">Exposure</th><th>Recommended action</th></tr></thead>
            <tbody>{d.top_risk_suppliers.map((s) => (
              <tr key={s.supplier_id} className="clickable" onClick={() => nav(`/risk?supplier=${s.supplier_id}`)}>
                <td style={{ maxWidth: 240 }}><div className="ellipsis"><b>{clean(s.supplier_name)}</b></div><span className="dim mono">{s.supplier_id}</span></td>
                <td><span className="badge b-tier">T{s.tier}</span></td>
                <td>{s.region}</td>
                <td><RiskBar p={s.probability_calibrated} max={0.3} />{pct(s.probability_calibrated)}</td>
                <td className="num">{compactInr(s.exposure_inr)}</td>
                <td><Badge band={(ACTION[s.recommended_action] || ['', s.risk_category])[1]}>{(ACTION[s.recommended_action] || [s.risk_category])[0]}</Badge></td>
              </tr>))}</tbody>
          </table></div>
        </Card>
        <div className="grid" style={{ gap: 18, alignContent: 'start' }}>
          <Card title="Critical dependencies" sub="Concentrated Tier-1 sourcing (observed graph)" delay={0.35}>
            {d.critical_dependencies.slice(0, 6).map((c) => (
              <div key={c.material_id} className="list-row">
                <span className="ellipsis"><Layers size={13} color="#fbbf24" style={{ verticalAlign: -2, marginRight: 6 }} />{c.material_name}</span>
                <span className="meta">HHI {c.hhi.toFixed(2)} · {c.suppliers} supplier{c.suppliers > 1 ? 's' : ''}</span>
              </div>))}
          </Card>
          <Card title="Active disruptions" sub={`${d.active_disruptions.length} ongoing · capacity lost`} delay={0.4}>
            <div style={{ maxHeight: 236, overflowY: 'auto', paddingRight: 4 }}>
              {d.active_disruptions.map((a) => (
                <div key={a.disruption_id} className="list-row">
                  <span className="ellipsis"><Factory size={12} color="#f43f5e" style={{ verticalAlign: -2, marginRight: 6 }} />{clean(a.supplier_name)} <span className="dim">T{a.tier}</span>
                    <div className="small dim" style={{ marginLeft: 18 }}>{nice(a.disruption_type)} · until {a.end_date}</div></span>
                  <span className="meta" style={{ color: '#f87171' }}>−{pct(a.capacity_loss_pct, 0)}</span>
                </div>))}
            </div>
          </Card>
        </div>
      </div>
      <p className="note" style={{ marginTop: 16 }}>Observed relationships: {num(t.observed_relationships)} · Total network exposure {compactInr(t.total_exposure_inr)} (simulated INR on synthetic data).</p>
    </>
  )
}
