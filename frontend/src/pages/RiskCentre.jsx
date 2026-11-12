import React, { useEffect, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { motion } from 'framer-motion'
import { Brain, Play } from 'lucide-react'
import { Bar, BarChart, CartesianGrid, Cell, Line, LineChart, ReferenceLine, ResponsiveContainer, Scatter, ComposedChart, Tooltip, XAxis, YAxis, Legend } from 'recharts'
import { api, compactInr, inr, pct } from '../api.js'
import { Badge, Card, ErrorBox, Loader, PageHeader, RiskBar, tooltipStyle, useApi } from '../components/ui.jsx'

const SNAPS = ['2025-07-01', '2025-08-01', '2025-09-01', '2025-10-01', '2025-11-01', '2025-12-01']

function ModelPanel() {
  const { data } = useApi(() => api('/api/experiments'))
  const r = data?.exp1_supplier_disruption
  if (!r) return null
  const rows = Object.entries(r.results).flatMap(([m, d]) => Object.entries(d).map(([c, v]) => ({ m, c, ...v })))
  return (
    <Card title="Model & calibration evidence (Experiment 1, untouched test split)" sub={`Test: ${r.splits.test.rows} supplier-months, prevalence ${pct(r.splits.test.prevalence)} · calibrators fitted on calibration split only`}>
      <div className="table-wrap" style={{ maxHeight: 330 }}><table>
        <thead><tr><th>Model</th><th>Calibration</th><th className="num">ROC-AUC</th><th className="num">PR-AUC</th><th className="num">Brier</th><th className="num">ECE (10 bins)</th></tr></thead>
        <tbody>{rows.map((x) => (
          <tr key={x.m + x.c} style={x.m === r.primary_model && x.c === r.primary_calibration ? { background: 'rgba(52,211,153,0.08)' } : {}}>
            <td>{x.m}</td><td>{x.c}</td><td className="num">{x.roc_auc.toFixed(3)}</td><td className="num">{x.pr_auc.toFixed(3)}</td>
            <td className="num">{x.brier.toFixed(4)}</td><td className="num">{x.ece_uniform10.toFixed(4)}</td></tr>))}</tbody>
      </table></div>
      <p className="note">Highlighted row = deployed model ({r.primary_model} + {r.primary_calibration}, selected by 5-fold CV Brier inside the calibration split). Synthetic labels: performance reflects the simulated network.</p>
    </Card>
  )
}

function Decision({ sid, ts }) {
  const [d, setD] = useState(null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState(null)
  useEffect(() => { setD(null) }, [sid])
  const run = async () => {
    setBusy(true); setErr(null)
    try { setD(await api('/api/decision/evaluate', { method: 'POST', body: { supplier_id: sid, timestamp: ts } })) } catch (e) { setErr(e) } finally { setBusy(false) }
  }
  return (
    <Card title="Business decision engine" sub="Prediction → exposure → cost-minimising action → recovery MILP" right={<button className="btn primary small" onClick={run} disabled={busy}><Play size={13} />{busy ? 'Evaluating…' : 'Run decision'}</button>}>
      <ErrorBox error={err} />
      {busy && <Loader text="Running inference, cascade simulation and optimisation…" />}
      {!d && !busy && <div className="note">Executes the 13-step workflow for {sid} and records the decision in Reports &amp; History.</div>}
      {d && <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
        <div className="row" style={{ marginBottom: 10 }}><Badge band={d.risk_category} /><b>{d.recommended_action.replaceAll('_', ' ')}</b><span className="small muted">{d.action_description}</span></div>
        <div className="callout" style={{ marginBottom: 12 }}>{d.decision_rationale}</div>
        <dl className="kv">
          <dt>Prediction timestamp</dt><dd>{d.prediction_timestamp.slice(0, 10)} ({d.information_cutoff})</dd>
          <dt>Raw / calibrated</dt><dd>{pct(d.disruption_probability)} / {pct(d.calibrated_probability)} ({d.calibration_method})</dd>
          <dt>Network exposure</dt><dd>{inr(d.network_exposure_inr)} — {d.exposure_graph}</dd>
          <dt>Inventory cover</dt><dd>{Object.entries(d.inventory_coverage_days || {}).map(([m, v]) => `${m}: ${v} d`).join(' · ') || '—'}</dd>
          <dt>30-day outage (no action)</dt><dd>{d.consequence_simulation?.unmet_demand_units?.toLocaleString()} units unmet · {inr(d.consequence_simulation?.lost_contribution_margin_inr)} margin</dd>
        </dl>
        <h4 style={{ margin: '14px 0 6px' }}>Expected cost by action</h4>
        <ResponsiveContainer width="100%" height={130}>
          <BarChart data={Object.entries(d.expected_cost_by_action_inr).map(([k, v]) => ({ k: k.replaceAll('_', ' '), v }))} layout="vertical">
            <XAxis type="number" tick={{ fontSize: 10 }} /><YAxis type="category" dataKey="k" width={140} tick={{ fontSize: 10 }} />
            <Tooltip {...tooltipStyle} formatter={(v) => inr(v)} />
            <Bar dataKey="v" radius={[0, 6, 6, 0]}>{Object.keys(d.expected_cost_by_action_inr).map((k) => <Cell key={k} fill={k === d.recommended_action ? '#34d399' : '#4f8dff'} />)}</Bar>
          </BarChart>
        </ResponsiveContainer>
        {d.recovery_plan && <>
          <h4 style={{ margin: '14px 0 6px' }}>Recovery plan (MILP: {d.recovery_plan.solver.status}) · {inr(d.recovery_plan.estimated_recovery_cost_inr)}</h4>
          <div className="table-wrap"><table><thead><tr><th>Supplier</th><th>Material</th><th className="num">Qty</th><th>Mode</th><th className="num">Lead</th><th className="num">Cost</th></tr></thead>
            <tbody>{d.recovery_plan.allocation.map((a, i) => <tr key={i}><td className="mono">{a.supplier_id}</td><td>{a.supplies_material}</td><td className="num">{a.qty.toLocaleString()}</td><td>{a.mode}</td><td className="num">{a.lead_days} d</td><td className="num">{inr(a.cost_inr)}</td></tr>)}</tbody></table></div>
        </>}
        <h4 style={{ margin: '14px 0 6px' }}>Alternative suppliers</h4>
        <div className="table-wrap" style={{ maxHeight: 200 }}><table><thead><tr><th>Supplier</th><th>Supplies</th><th className="num">Compat.</th><th className="num">Spare/day</th><th className="num">Risk</th><th className="num">Lead</th></tr></thead>
          <tbody>{(d.alternative_suppliers || []).map((a, i) => <tr key={i}><td>{a.supplier_name.replace(' (Synthetic)', '')}</td><td>{a.supplies_material}</td><td className="num">{a.compatibility}</td><td className="num">{a.spare_capacity_per_day}</td><td className="num">{pct(a.calibrated_risk_p30)}</td><td className="num">{a.lead_time_days} d</td></tr>)}</tbody></table></div>
      </motion.div>}
    </Card>
  )
}

export default function RiskCentre() {
  const [params] = useSearchParams()
  const [ts, setTs] = useState('2025-12-01')
  const [sel, setSel] = useState(params.get('supplier'))
  const { data, error, loading } = useApi(() => api('/api/ml/supplier-risk', { method: 'POST', body: { timestamp: ts } }), [ts])
  const exp = useApi(() => sel ? api('/api/ml/supplier-risk', { method: 'POST', body: { timestamp: ts, supplier_ids: [sel], with_explanation: true } }) : Promise.resolve(null), [sel, ts])
  const hist = useApi(() => sel ? api(`/api/ml/predictions/${sel}/history`) : Promise.resolve(null), [sel])
  useEffect(() => { if (!sel && data?.predictions?.length) setSel(data.predictions[0].supplier_id) }, [data])
  const shap = exp.data?.explanations?.[sel] || []
  const p = exp.data?.predictions?.[0]
  return (
    <>
      <PageHeader title="Risk Prediction Centre"
        desc="Probability that a supplier suffers a qualifying disruption (≥25 % capacity loss for ≥3 days) in the 30 days after the prediction timestamp. Features use only records available before that timestamp.">
        <select value={ts} onChange={(e) => setTs(e.target.value)} style={{ width: 170 }}>{SNAPS.map((s) => <option key={s}>{s}</option>)}</select>
      </PageHeader>
      <ErrorBox error={error} />
      <div className="grid g3" style={{ marginBottom: 18 }}>
        <Card title="Risk ranking" sub={data ? `${data.model_version} · ${data.calibration} calibration · t = ${data.prediction_timestamp}` : ''}>
          {loading ? <Loader /> : <div className="table-wrap" style={{ maxHeight: 620 }}><table>
            <thead><tr><th>Supplier</th><th>p(30d)</th><th>Band</th></tr></thead>
            <tbody>{data.predictions.slice(0, 60).map((x) => (
              <tr key={x.supplier_id} className="clickable" onClick={() => setSel(x.supplier_id)} style={x.supplier_id === sel ? { background: 'rgba(79,141,255,0.16)' } : {}}>
                <td className="mono">{x.supplier_id}</td><td><RiskBar p={x.probability_calibrated} />{pct(x.probability_calibrated)}</td><td><Badge band={x.risk_band} /></td></tr>))}</tbody>
          </table></div>}
        </Card>
        <div className="span2 grid" style={{ gap: 18 }}>
          <Card title={sel ? `Explanation · ${sel}` : 'Explanation'} sub="SHAP contributions to the raw XGBoost log-odds" right={p && <div className="row"><Badge band={p.risk_band} /><span className="small">raw {pct(p.probability_raw)} → calibrated <b>{pct(p.probability_calibrated)}</b></span></div>}>
            {exp.loading ? <Loader /> : <ResponsiveContainer width="100%" height={230}>
              <BarChart data={shap.map((s) => ({ ...s, label: `${s.feature} = ${s.value === null ? 'NA' : Number(s.value).toFixed(3)}` }))} layout="vertical" margin={{ left: 30 }}>
                <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                <XAxis type="number" tick={{ fontSize: 10 }} /><YAxis type="category" dataKey="label" width={250} tick={{ fontSize: 10.5 }} />
                <ReferenceLine x={0} stroke="#8ea3c7" />
                <Tooltip {...tooltipStyle} />
                <Bar dataKey="shap_log_odds" radius={4} animationDuration={900}>{shap.map((s, i) => <Cell key={i} fill={s.shap_log_odds > 0 ? '#f87171' : '#34d399'} />)}</Bar>
              </BarChart>
            </ResponsiveContainer>}
          </Card>
          <Card title="Risk trajectory" sub="Calibrated probability per monthly snapshot vs. realised qualifying disruption (synthetic ground truth)">
            {hist.loading ? <Loader /> : <ResponsiveContainer width="100%" height={200}>
              <ComposedChart data={(hist.data || []).map((h) => ({ ...h, hit: h.actual_disruption_next_30d ? h.p_cal : null }))}>
                <CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="timestamp" tick={{ fontSize: 10 }} /><YAxis tick={{ fontSize: 10 }} tickFormatter={(v) => pct(v, 0)} />
                <Tooltip {...tooltipStyle} formatter={(v) => pct(v)} /><Legend />
                <Line type="monotone" dataKey="p_cal" name="calibrated p" stroke="#22d3ee" strokeWidth={2.5} dot={{ r: 3 }} animationDuration={1200} />
                <Line type="monotone" dataKey="p_raw" name="raw p" stroke="#4f8dff" strokeDasharray="4 3" dot={false} />
                <Scatter dataKey="hit" name="disruption followed" fill="#f43f5e" />
              </ComposedChart>
            </ResponsiveContainer>}
          </Card>
          {sel && <Decision sid={sel} ts={ts} />}
        </div>
      </div>
      <ModelPanel />
    </>
  )
}
