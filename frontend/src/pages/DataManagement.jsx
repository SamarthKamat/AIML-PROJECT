import React, { useState } from 'react'
import { Upload } from 'lucide-react'
import { api, num } from '../api.js'
import { Card, ErrorBox, Loader, PageHeader, useApi } from '../components/ui.jsx'

function Report({ rep }) {
  return (
    <div>
      <div className="row" style={{ marginBottom: 8 }}><span className={`badge ${rep.status === 'pass' ? 'b-ok' : 'b-fail'}`}>{rep.status.toUpperCase()}</span>
        <span className="small muted">{rep.checks_passed}/{rep.checks_run} checks passed · {num(rep.rows)} rows · {rep.columns} columns</span></div>
      {rep.errors?.map((e, i) => <div key={i} className="small" style={{ color: 'var(--bad)' }}>✗ {e}</div>)}
      {rep.warnings?.map((e, i) => <div key={i} className="small" style={{ color: 'var(--warn)' }}>! {e}</div>)}
      {Object.keys(rep.missing_values || {}).length > 0 && <div className="small muted">Missing values: {Object.entries(rep.missing_values).map(([k, v]) => `${k}=${v}`).join(', ')}</div>}
      {Object.keys(rep.outliers_3iqr_investigated_not_removed || {}).length > 0 && <div className="small dim">Outliers (3×IQR, flagged not removed): {Object.entries(rep.outliers_3iqr_investigated_not_removed).map(([k, v]) => `${k}=${v}`).join(', ')}</div>}
    </div>
  )
}

export default function DataManagement() {
  const tables = useApi(() => api('/api/data/tables'))
  const [hist, setHist] = useState(0)
  const history = useApi(() => api('/api/data/history'), [hist])
  const [prev, setPrev] = useState(null)
  const [file, setFile] = useState(null)
  const [target, setTarget] = useState('suppliers')
  const [up, setUp] = useState(null)
  const [mapping, setMapping] = useState({})
  const [mapped, setMapped] = useState(null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState(null)
  const [val, setVal] = useState(null)
  const send = async () => {
    if (!file) return
    setBusy(true); setErr(null); setMapped(null)
    const fd = new FormData(); fd.append('file', file); fd.append('target_table', target)
    try { const r = await api('/api/data/upload', { method: 'POST', form: fd }); setUp(r); setMapping(r.suggested_mapping); setHist(hist + 1) } catch (e) { setErr(e) } finally { setBusy(false) }
  }
  const remap = async () => {
    const fd = new FormData(); fd.append('file', file); fd.append('target_table', target); fd.append('mapping', JSON.stringify(mapping))
    try { setMapped(await api('/api/data/map-columns', { method: 'POST', form: fd })) } catch (e) { setErr(e) }
  }
  const cols = tables.data?.find((t) => t.table === target)?.columns.map((c) => c.name) || []
  return (
    <>
      <PageHeader title="Data Management" desc="Master tables, CSV / Excel upload with schema validation, column mapping, duplicate & missing-value detection and processing history. Uploaded files are validated and logged; master data are never overwritten." />
      <div className="grid g3" style={{ marginBottom: 18 }}>
        <Card title="Master tables" sub="Rows currently in the database">
          {tables.loading ? <Loader /> : <div className="table-wrap" style={{ maxHeight: 420 }}><table><thead><tr><th>Table</th><th className="num">Rows</th><th className="num">Cols</th></tr></thead>
            <tbody>{tables.data?.map((t) => <tr key={t.table} className="clickable" onClick={() => api(`/api/data/preview/${t.table}`).then((d) => setPrev({ t: t.table, d }))}>
              <td className="mono">{t.table}</td><td className="num">{num(t.rows)}</td><td className="num">{t.columns.length}</td></tr>)}</tbody></table></div>}
          <button className="btn small" style={{ marginTop: 12 }} onClick={() => { setVal('loading'); api('/api/data/validation').then(setVal) }}>Validate all master tables</button>
        </Card>
        <Card title="Upload & validate" className="span2">
          <div className="row" style={{ flexWrap: 'nowrap', marginBottom: 12 }}>
            <input type="file" accept=".csv,.xlsx,.xls" onChange={(e) => setFile(e.target.files[0])} />
            <select value={target} onChange={(e) => setTarget(e.target.value)} style={{ width: 240 }}>{tables.data?.map((t) => <option key={t.table}>{t.table}</option>)}</select>
            <button className="btn primary" onClick={send} disabled={!file || busy}><Upload size={14} />{busy ? 'Validating…' : 'Upload'}</button>
          </div>
          <ErrorBox error={err} />
          {up && <>
            <Report rep={up.validation} />
            <h4 style={{ margin: '14px 0 6px' }}>Column mapping → {up.target_table}</h4>
            <div className="grid g3" style={{ gap: 8 }}>{Object.keys(mapping).map((c) => (
              <label key={c} className="field small">{c}<select value={mapping[c] || ''} onChange={(e) => setMapping({ ...mapping, [c]: e.target.value || null })}>
                <option value="">(ignore)</option>{cols.map((x) => <option key={x}>{x}</option>)}</select></label>))}</div>
            <button className="btn small" style={{ marginTop: 10 }} onClick={remap}>Apply mapping &amp; re-validate</button>
            {mapped && <div style={{ marginTop: 10 }}><Report rep={mapped.validation} /></div>}
            <h4 style={{ margin: '14px 0 6px' }}>Preview</h4>
            <div className="table-wrap" style={{ maxHeight: 240 }}><table><thead><tr>{up.columns.map((c) => <th key={c}>{c}</th>)}</tr></thead>
              <tbody>{up.preview.map((r, i) => <tr key={i}>{up.columns.map((c) => <td key={c}>{String(r[c] ?? '')}</td>)}</tr>)}</tbody></table></div>
          </>}
        </Card>
      </div>
      {val && <Card title="Master-table validation" style={{ marginBottom: 18 }}>{val === 'loading' ? <Loader /> :
        <div className="grid g3">{Object.entries(val).map(([t, r]) => <div key={t} className="card" style={{ padding: 12 }}><b className="mono">{t}</b><Report rep={r} /></div>)}</div>}</Card>}
      {prev && <Card title={`Preview · ${prev.t}`} style={{ marginBottom: 18 }}>
        <div className="table-wrap" style={{ maxHeight: 300 }}><table><thead><tr>{Object.keys(prev.d[0] || {}).map((c) => <th key={c}>{c}</th>)}</tr></thead>
          <tbody>{prev.d.map((r, i) => <tr key={i}>{Object.values(r).map((v, j) => <td key={j}>{String(v ?? '')}</td>)}</tr>)}</tbody></table></div></Card>}
      <Card title="Processing history">
        <div className="table-wrap"><table><thead><tr><th>#</th><th>File</th><th>Target</th><th className="num">Rows</th><th>Status</th><th>When</th></tr></thead>
          <tbody>{(history.data || []).map((h) => <tr key={h.upload_id}><td>{h.upload_id}</td><td>{h.filename}</td><td className="mono">{h.target_table}</td><td className="num">{h.row_count}</td>
            <td><span className={`badge ${h.status === 'pass' ? 'b-ok' : 'b-fail'}`}>{h.status}</span></td><td className="small muted">{String(h.created_at)}</td></tr>)}</tbody></table></div>
      </Card>
    </>
  )
}
