import React from 'react'
import { api, compactInr, pct } from '../api.js'
import { Badge, Card, ErrorBox, Loader, PageHeader, useApi } from '../components/ui.jsx'

function ExpTable({ exps }) {
  const e3 = exps.exp3_delivery_delay_dataco
  const e2 = exps.exp2b_graph_risk_classification
  return (
    <div className="grid g2" style={{ marginBottom: 18 }}>
      {e3 && <Card title="Exp 3 · Late-delivery prediction (REAL DataCo)" sub={`Chronological test: ${e3.splits.test.rows.toLocaleString()} order lines, ${e3.splits.test.start.slice(0, 10)} → ${e3.splits.test.end.slice(0, 10)}`}>
        <div className="table-wrap"><table><thead><tr><th>Model</th><th className="num">ROC-AUC</th><th className="num">PR-AUC</th><th className="num">F1</th><th className="num">Brier</th></tr></thead>
          <tbody>{Object.entries(e3.classification).map(([k, v]) => <tr key={k}><td>{k}</td><td className="num">{v.none.roc_auc.toFixed(3)}</td><td className="num">{v.none.pr_auc.toFixed(3)}</td>
            <td className="num">{v.none.f1.toFixed(3)}</td><td className="num">{v.none.brier.toFixed(4)}</td></tr>)}</tbody></table></div>
        <div className="table-wrap" style={{ marginTop: 10 }}><table><thead><tr><th>Delay regression</th><th className="num">MAE</th><th className="num">RMSE</th><th className="num">R²</th></tr></thead>
          <tbody>{Object.entries(e3.regression).map(([k, v]) => <tr key={k}><td>{k}</td><td className="num">{v.mae.toFixed(3)}</td><td className="num">{v.rmse.toFixed(3)}</td><td className="num">{v.r2.toFixed(3)}</td></tr>)}</tbody></table></div>
      </Card>}
      {e2 && <Card title="Exp 2B · Graph vs tabular risk classification" sub="PR-AUC mean ± sd over 5 seeds (Platt-calibrated)">
        {[['Synthetic network', e2.part1_synthetic.results], ['SupplyGraph (REAL FMCG)', e2.part2_supplygraph_real.results]].map(([t, r]) => (
          <div key={t} style={{ marginBottom: 10 }}><div className="small muted" style={{ margin: '6px 0' }}>{t}</div>
            <div className="table-wrap"><table><thead><tr><th>Model</th><th className="num">PR-AUC</th><th className="num">ROC-AUC</th><th className="num">Brier</th></tr></thead>
              <tbody>{Object.entries(r).map(([k, v]) => <tr key={k}><td className="small">{k}</td><td className="num">{v.pr_auc.toFixed(3)} ± {v.pr_auc_sd.toFixed(3)}</td><td className="num">{v.roc_auc.toFixed(3)}</td><td className="num">{v.brier.toFixed(4)}</td></tr>)}</tbody></table></div></div>))}
      </Card>}
    </div>
  )
}

export default function Reports() {
  const { data, error, loading } = useApi(() => api('/api/reports'))
  const exps = useApi(() => api('/api/experiments'))
  return (
    <>
      <PageHeader title="Reports & History" desc="Audit trail of every decision, simulation, recovery plan, upload and assistant query, plus the logged experiment results that the IEEE report cites." />
      {exps.data && <ExpTable exps={exps.data} />}
      {loading ? <Loader /> : error ? <ErrorBox error={error} /> : <div className="grid g2">
        <Card title="Decisions" sub="Recorded by the decision engine">
          <div className="table-wrap" style={{ maxHeight: 340 }}><table><thead><tr><th>#</th><th>Supplier</th><th>p(30d)</th><th>Category</th><th>Action</th><th>When</th></tr></thead>
            <tbody>{data.decisions.map((d) => <tr key={d.decision_id} title={d.rationale}><td>{d.decision_id}</td><td className="mono">{d.supplier_id}</td><td>{pct(d.calibrated_probability)}</td>
              <td><Badge band={d.risk_category} /></td><td className="small">{d.recommended_action}</td><td className="small muted">{String(d.created_at).slice(0, 19)}</td></tr>)}</tbody></table></div>
        </Card>
        <Card title="Simulation runs">
          <div className="table-wrap" style={{ maxHeight: 340 }}><table><thead><tr><th>#</th><th>Type</th><th>Parameters</th><th className="num">Impact</th><th>When</th></tr></thead>
            <tbody>{data.simulations.map((s) => { const p = typeof s.parameters === 'string' ? JSON.parse(s.parameters) : s.parameters; const r = typeof s.results === 'string' ? JSON.parse(s.results) : s.results
              return <tr key={s.run_id}><td>{s.run_id}</td><td>{s.run_type}</td><td className="small">{p?.targets ? `${p.targets.slice(0, 3).join(',')}${p.targets.length > 3 ? '…' : ''} · ${pct(p.capacity_reduction, 0)} · ${p.duration_days}d · ${p.policy}` : JSON.stringify(p)}</td>
                <td className="num">{r?.financial_impact_inr !== undefined ? compactInr(r.financial_impact_inr) : '—'}</td><td className="small muted">{String(s.created_at).slice(0, 19)}</td></tr> })}</tbody></table></div>
        </Card>
        <Card title="Recovery plans">
          <div className="table-wrap" style={{ maxHeight: 280 }}><table><thead><tr><th>#</th><th>Material</th><th>Solver</th><th className="num">Cost</th><th>When</th></tr></thead>
            <tbody>{data.recovery_plans.map((r) => <tr key={r.recommendation_id}><td>{r.recommendation_id}</td><td>{r.material_id}</td><td>{r.solver_status}</td><td className="num">{compactInr(r.total_cost_inr)}</td><td className="small muted">{String(r.created_at).slice(0, 19)}</td></tr>)}</tbody></table></div>
        </Card>
        <Card title="Assistant queries">
          <div className="table-wrap" style={{ maxHeight: 280 }}><table><thead><tr><th>#</th><th>Question</th><th>Engine</th></tr></thead>
            <tbody>{data.chats.map((c) => <tr key={c.chat_id}><td>{c.chat_id}</td><td className="small" style={{ whiteSpace: 'normal' }}>{c.question}</td><td className="small muted">{c.engine}</td></tr>)}</tbody></table></div>
        </Card>
      </div>}
    </>
  )
}
