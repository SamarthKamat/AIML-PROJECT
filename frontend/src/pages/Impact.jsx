import React from 'react'
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { api, compactInr, num, pct } from '../api.js'
import { Card, ErrorBox, Loader, PageHeader, tooltipStyle, useApi } from '../components/ui.jsx'

export default function Impact() {
  const { data, error, loading } = useApi(() => api('/api/experiments'))
  const types = useApi(() => api('/api/simulation/impact-types'))
  if (loading) return <><PageHeader title="Impact Analysis" /><Loader /></>
  if (error) return <><PageHeader title="Impact Analysis" /><ErrorBox error={error} /></>
  const e1 = data.exp1_supplier_disruption
  const ex = e1?.exposure_summary
  const gap = ex ? [1, 2, 3].map((t) => ({ tier: `Tier ${t}`, observed: ex.observed_graph_mean_by_tier[t], true: ex.true_graph_mean_by_tier[t] })) : []
  const bo = e1?.business_outcome_test_inr || {}
  const lp = data.exp2a_link_prediction
  return (
    <>
      <PageHeader title="Impact Analysis" desc="Where the black box hides risk: exposure seen through the observed network vs. the true network, disruption-type consequences, and measured business outcomes of the decision policy." />
      <div className="grid g2" style={{ marginBottom: 18 }}>
        <Card title="Hidden-dependency gap" sub="Mean network exposure per supplier (INR) — observed graph vs. synthetic ground truth">
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={gap}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="tier" /><YAxis tickFormatter={compactInr} tick={{ fontSize: 10 }} />
              <Tooltip {...tooltipStyle} formatter={(v) => compactInr(v)} /><Legend />
              <Bar dataKey="observed" name="Observed graph (what the buyer sees)" fill="#4f8dff" radius={[6, 6, 0, 0]} animationDuration={1200} />
              <Bar dataKey="true" name="True graph (incl. hidden edges)" fill="#f43f5e" radius={[6, 6, 0, 0]} animationDuration={1400} />
            </BarChart>
          </ResponsiveContainer>
          {gap.length > 0 && <p className="note">Tier-2 exposure is underestimated {(gap[1].true / gap[1].observed).toFixed(1)}× and Tier-3 {(gap[2].true / Math.max(gap[2].observed, 1)).toFixed(1)}× when only observed relationships are used.</p>}
        </Card>
        <Card title="Decision-policy business outcome (Exp 1 test split)" sub="Realised cost = action costs + unmitigated true-exposure losses of suppliers that were disrupted">
          <div className="table-wrap"><table><thead><tr><th>Policy</th><th className="num">Realised cost (simulated INR)</th></tr></thead>
            <tbody>{Object.entries(bo).map(([k, v]) => <tr key={k}><td>{k}</td><td className="num">{compactInr(v)}</td></tr>)}</tbody></table></div>
          <p className="note">Honest finding: with ~3–4 weeks of inventory cover, most single-supplier disruptions are absorbed, so the calibrated policy rarely escalates; value concentrates in network-level shocks (see Scenario Comparison).</p>
        </Card>
      </div>
      <Card title="Experiment 4 — cascade consequences by disruption type (no intervention)" sub="Means over Monte Carlo replications on the true synthetic network" style={{ marginBottom: 18 }}>
        {types.loading ? <Loader /> : types.error ? <ErrorBox error={types.error} /> :
          <div className="table-wrap"><table><thead><tr><th>Disruption</th><th className="num">Disrupted</th><th className="num">Affected sup.</th><th className="num">Affected mat.</th>
            <th className="num">Sites</th><th className="num">Unmet units</th><th className="num">Lost margin</th><th className="num">Financial impact</th><th className="num">P97.5</th><th className="num">Recovery d</th><th className="num">Min service</th></tr></thead>
            <tbody>{types.data.map((r) => <tr key={r.case}><td>{r.case}</td><td className="num">{num(r.n_disrupted)}</td><td className="num">{r.affected_suppliers.toFixed(1)}</td>
              <td className="num">{r.affected_materials.toFixed(1)}</td><td className="num">{r.affected_production_sites.toFixed(1)}</td><td className="num">{num(r.unmet_demand_units)}</td>
              <td className="num">{compactInr(r.production_loss_cm_inr)}</td><td className="num"><b>{compactInr(r.financial_impact_inr)}</b></td><td className="num">{compactInr(r['financial_impact_p97.5'])}</td>
              <td className="num">{r.recovery_time_days.toFixed(1)}</td><td className="num">{pct(r.min_service_level)}</td></tr>)}</tbody></table></div>}
      </Card>
      {lp && <Card title="Hidden-relationship discovery (Exp 2A, A2)" sub={`${lp.A2_candidates.n} compatible unobserved pairs, ${lp.A2_candidates.positives} true hidden edges (base rate ${pct(lp.A2_candidates.base_rate)})`}>
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={Object.entries(lp.A2_results).map(([k, v]) => ({ k, AP: v.ap, 'P@100': v['P@100'], 'ROC-AUC': v.roc_auc }))}>
            <CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="k" tick={{ fontSize: 10 }} /><YAxis domain={[0, 1]} tick={{ fontSize: 10 }} />
            <Tooltip {...tooltipStyle} /><Legend />
            <Bar dataKey="ROC-AUC" fill="#4f8dff" radius={[4, 4, 0, 0]} /><Bar dataKey="AP" fill="#22d3ee" radius={[4, 4, 0, 0]} /><Bar dataKey="P@100" fill="#a78bfa" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
        <p className="note">Predicted links are candidates for supplier verification (e.g., supplier surveys, bill-of-lading checks) — never treated as confirmed relationships.</p>
      </Card>}
    </>
  )
}
