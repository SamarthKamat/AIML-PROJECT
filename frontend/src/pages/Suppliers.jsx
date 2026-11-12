import React, { useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { Search, X } from 'lucide-react'
import { api, compactInr, num, pct } from '../api.js'
import { Badge, Card, ErrorBox, Loader, PageHeader, RiskBar, useApi } from '../components/ui.jsx'

export function SupplierDrawer({ id, onClose }) {
  const { data: d, error, loading } = useApi(() => api(`/api/suppliers/${id}`), [id])
  return (
    <motion.div initial={{ x: 480, opacity: 0 }} animate={{ x: 0, opacity: 1 }} exit={{ x: 480, opacity: 0 }} transition={{ type: 'spring', damping: 26 }}
      className="card" style={{ position: 'fixed', right: 18, top: 18, bottom: 18, width: 460, zIndex: 50, overflowY: 'auto' }}>
      <div className="card-h"><h3>Supplier profile</h3><button className="btn small ghost" onClick={onClose}><X size={14} /></button></div>
      {loading && <Loader />}<ErrorBox error={error} />
      {d && <>
        <h2 style={{ margin: '0 0 4px' }}>{d.supplier.supplier_name}</h2>
        <div className="row small muted" style={{ marginBottom: 14 }}><span className="mono">{d.supplier.supplier_id}</span>
          <span className="badge b-tier">Tier {d.supplier.tier}</span><Badge band={d.risk.band} /><span>{d.supplier.region}, {d.supplier.country}</span></div>
        <dl className="kv">
          <dt>Supplies</dt><dd>{d.supplier.material_name} ({d.supplier.primary_material_id})</dd>
          <dt>Calibrated p(30d)</dt><dd>{pct(d.risk.calibrated_probability)} → {d.risk.recommended_action?.replaceAll('_', ' ')}</dd>
          <dt>Network exposure</dt><dd>{compactInr(d.risk.exposure_inr)}</dd>
          <dt>Reliability</dt><dd>{d.supplier.reliability_score}</dd>
          <dt>Utilisation</dt><dd>{pct(d.supplier.capacity_utilisation, 0)}</dd>
          <dt>Financial risk score</dt><dd>{d.supplier.financial_risk_score}</dd>
          <dt>PageRank / reach</dt><dd>{d.centrality.pagerank.toFixed(4)} · {d.centrality.downstream_reach} downstream</dd>
          <dt>Shipments (90d hist.)</dt><dd>{d.shipment_history.n} · delay rate {d.shipment_history.delay_rate == null ? '—' : pct(d.shipment_history.delay_rate)}</dd>
        </dl>
        <h4 style={{ margin: '18px 0 8px' }}>Upstream (observed)</h4>
        {d.upstream.length === 0 && <div className="small dim">No observed upstream suppliers — possible blind spot.</div>}
        {d.upstream.map((u) => <div key={u.supplier_id + u.material_id} className="small" style={{ padding: '3px 0' }}>{u.supplier_name} <span className="dim">T{u.tier} · {u.material_id} · {pct(u.dependency_pct, 0)}</span></div>)}
        <h4 style={{ margin: '18px 0 8px' }}>Downstream</h4>
        {d.downstream.map((u, i) => <div key={i} className="small" style={{ padding: '3px 0' }}>{u.buyer_id === 'C001' ? 'Manufacturer (C001)' : u.buyer_id} <span className="dim">{u.material_id} · share {pct(u.dependency_pct, 0)}</span></div>)}
        <h4 style={{ margin: '18px 0 8px' }}>Candidate links <span className="badge b-pred">predicted · unverified</span></h4>
        {d.candidate_links.slice(0, 6).map((c, i) => <div key={i} className="small" style={{ padding: '3px 0' }}>{c.supplier_id} → {c.buyer_id} <span className="dim">rank {c.rank} · score {c.score.toFixed(2)}</span></div>)}
        <h4 style={{ margin: '18px 0 8px' }}>Alternatives (same material, lowest risk)</h4>
        {d.alternatives.map((a) => <div key={a.supplier_id} className="small" style={{ padding: '3px 0' }}>{a.supplier_name} <span className="dim">{a.region} · rel {a.reliability} · risk {pct(a.risk)}</span></div>)}
        <h4 style={{ margin: '18px 0 8px' }}>Disruption history (reported)</h4>
        {d.disruption_history.slice(0, 6).map((h) => <div key={h.disruption_id} className="small" style={{ padding: '3px 0' }}>{h.start_date} · {h.disruption_type.replaceAll('_', ' ')} <span className="dim">{h.duration_days} d · −{pct(h.capacity_loss_pct, 0)}</span></div>)}
      </>}
    </motion.div>
  )
}

export default function Suppliers() {
  const [f, setF] = useState({ search: '', tier: '', region: '', risk_band: '', sort: '-probability_calibrated', page: 1 })
  const [sel, setSel] = useState(null)
  const qs = new URLSearchParams(Object.entries({ ...f, page_size: 20 }).filter(([, v]) => v !== '')).toString()
  const { data, error, loading } = useApi(() => api(`/api/suppliers?${qs}`), [qs])
  const up = (k) => (e) => setF({ ...f, [k]: e.target.value, page: 1 })
  const pages = data ? Math.ceil(data.total / data.page_size) : 1
  return (
    <>
      <PageHeader title="Supplier Directory" desc="All 500 suppliers across three tiers with live calibrated risk, network exposure and the cost-optimal action." />
      <Card>
        <div className="filters">
          <div style={{ position: 'relative' }}><Search size={15} style={{ position: 'absolute', left: 11, top: 11 }} color="#8ea3c7" />
            <input placeholder="Search name or ID…" value={f.search} onChange={up('search')} style={{ paddingLeft: 34 }} /></div>
          <select value={f.tier} onChange={up('tier')}><option value="">All tiers</option><option value="1">Tier 1</option><option value="2">Tier 2</option><option value="3">Tier 3</option></select>
          <select value={f.region} onChange={up('region')}><option value="">All regions</option>{data?.regions?.map((r) => <option key={r}>{r}</option>)}</select>
          <select value={f.risk_band} onChange={up('risk_band')}><option value="">All risk bands</option>{['Low', 'Medium', 'High', 'Critical'].map((b) => <option key={b}>{b}</option>)}</select>
          <select value={f.sort} onChange={up('sort')}>
            <option value="-probability_calibrated">Risk ↓</option><option value="-exposure_inr">Exposure ↓</option>
            <option value="-reliability_score">Reliability ↓</option><option value="supplier_id">ID ↑</option></select>
        </div>
        <ErrorBox error={error} />
        {loading ? <Loader /> : data && <>
          <div className="table-wrap"><table>
            <thead><tr><th>ID</th><th>Supplier</th><th>Tier</th><th>Material</th><th>Region</th><th className="num">Reliability</th>
              <th className="num">Utilisation</th><th>Calibrated p(30d)</th><th>Band</th><th className="num">Exposure</th><th>Action</th></tr></thead>
            <tbody>{data.items.map((s, i) => (
              <motion.tr key={s.supplier_id} className="clickable" onClick={() => setSel(s.supplier_id)} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: i * 0.015 }}>
                <td className="mono dim">{s.supplier_id}</td><td><b>{s.supplier_name.replace(' (Synthetic)', '')}</b></td>
                <td><span className="badge b-tier">T{s.tier}</span></td><td className="mono">{s.primary_material_id}</td><td>{s.region}</td>
                <td className="num">{s.reliability_score.toFixed(3)}</td><td className="num">{pct(s.capacity_utilisation, 0)}</td>
                <td><RiskBar p={s.probability_calibrated} />{pct(s.probability_calibrated)}</td><td><Badge band={s.risk_band} /></td>
                <td className="num">{compactInr(s.exposure_inr)}</td><td className="small muted">{s.recommended_action?.replaceAll('_', ' ')}</td>
              </motion.tr>))}</tbody>
          </table></div>
          <div className="row" style={{ justifyContent: 'space-between', marginTop: 14 }}>
            <span className="small muted">{num(data.total)} suppliers · page {data.page} of {pages}</span>
            <div className="row">
              <button className="btn small" disabled={f.page <= 1} onClick={() => setF({ ...f, page: f.page - 1 })}>Previous</button>
              <button className="btn small" disabled={f.page >= pages} onClick={() => setF({ ...f, page: f.page + 1 })}>Next</button>
            </div>
          </div>
        </>}
      </Card>
      <AnimatePresence>{sel && <SupplierDrawer id={sel} onClose={() => setSel(null)} />}</AnimatePresence>
    </>
  )
}
