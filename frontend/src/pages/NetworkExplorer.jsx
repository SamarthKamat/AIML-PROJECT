import React, { useEffect, useMemo, useRef, useState } from 'react'
import ForceGraph2D from 'react-force-graph-2d'
import { AnimatePresence, motion } from 'framer-motion'
import { api, compactInr, pct } from '../api.js'
import { BAND_COLORS, Badge, ErrorBox, Loader, PageHeader, TIER_COLORS, useApi } from '../components/ui.jsx'

export default function NetworkExplorer() {
  const [pred, setPred] = useState(false)
  const [maxTier, setMaxTier] = useState(3)
  const [colorBy, setColorBy] = useState('tier')
  const [sel, setSel] = useState(null)
  const [detail, setDetail] = useState(null)
  const [hover, setHover] = useState(null)
  const wrap = useRef(null)
  const fg = useRef(null)
  const [size, setSize] = useState({ w: 900, h: 600 })
  const { data, error, loading } = useApi(() => api(`/api/graph/network?include_predicted=${pred ? 1 : 0}&max_tier=${maxTier}`), [pred, maxTier])

  useEffect(() => {
    const ro = new ResizeObserver(() => wrap.current && setSize({ w: wrap.current.offsetWidth, h: wrap.current.offsetHeight }))
    if (wrap.current) ro.observe(wrap.current)
    return () => ro.disconnect()
  }, [])
  useEffect(() => { if (sel) { setDetail(null); api(`/api/suppliers/${sel}`).then(setDetail).catch(() => setDetail(null)) } }, [sel])

  const graph = useMemo(() => {
    if (!data) return null
    return { nodes: data.nodes.map((n) => ({ ...n })), links: data.links.map((l) => ({ ...l })) }
  }, [data])
  const neighbours = useMemo(() => {
    const s = new Set()
    if (!graph || !(hover || sel)) return s
    const id = hover || sel
    graph.links.forEach((l) => {
      const a = typeof l.source === 'object' ? l.source.id : l.source
      const b = typeof l.target === 'object' ? l.target.id : l.target
      if (a === id) s.add(b); if (b === id) s.add(a)
    })
    s.add(id)
    return s
  }, [graph, hover, sel])

  useEffect(() => {
    if (fg.current) {
      fg.current.d3Force('charge').strength(-38)
      fg.current.d3Force('link').distance((l) => (l.status === 'predicted' ? 60 : 34))
    }
  }, [graph])

  const color = (n) => (n.tier === 0 ? '#ffffff' : colorBy === 'tier' ? TIER_COLORS[n.tier] : BAND_COLORS[n.band] || '#94a3b8')
  return (
    <>
      <PageHeader title="Supplier Network Explorer"
        desc="Directed multi-tier network (supplier → buyer). Particles flow towards the manufacturer; dashed violet edges are link-prediction candidate relationships — predictions, not confirmed links." />
      <div className="graph-wrap" ref={wrap}>
        <div className="graph-toolbar">
          <div className="seg">{[1, 2, 3].map((t) => <button key={t} className={maxTier === t ? 'on' : ''} onClick={() => setMaxTier(t)}>≤ Tier {t}</button>)}</div>
          <div className="seg"><button className={colorBy === 'tier' ? 'on' : ''} onClick={() => setColorBy('tier')}>Colour: tier</button>
            <button className={colorBy === 'risk' ? 'on' : ''} onClick={() => setColorBy('risk')}>Colour: risk</button></div>
          <button className={`btn small ${pred ? 'primary' : ''}`} onClick={() => setPred(!pred)}>{pred ? '✓ ' : ''}Show predicted links</button>
        </div>
        {loading && <Loader text="Building network…" />}<ErrorBox error={error} />
        {graph && (
          <ForceGraph2D ref={fg} graphData={graph} width={size.w} height={size.h} backgroundColor="rgba(0,0,0,0)"
            cooldownTicks={180} d3VelocityDecay={0.3}
            nodeRelSize={4}
            nodeVal={(n) => (n.tier === 0 ? 14 : 1.2 + (n.degree || 1) * 0.35)}
            linkColor={(l) => {
              const a = typeof l.source === 'object' ? l.source.id : l.source
              const b = typeof l.target === 'object' ? l.target.id : l.target
              const on = neighbours.size === 0 || (neighbours.has(a) && neighbours.has(b))
              if (l.status === 'predicted') return on ? 'rgba(167,139,250,0.75)' : 'rgba(167,139,250,0.08)'
              return on ? 'rgba(96,148,255,0.45)' : 'rgba(96,148,255,0.05)'
            }}
            linkWidth={(l) => (l.dependency ? 0.5 + 2.4 * l.dependency : 0.6)}
            linkLineDash={(l) => (l.status === 'predicted' ? [3, 3] : null)}
            linkDirectionalParticles={(l) => (l.status === 'observed' ? 1 : 0)}
            linkDirectionalParticleSpeed={0.006} linkDirectionalParticleWidth={1.8}
            linkDirectionalParticleColor={() => '#22d3ee'}
            onNodeHover={(n) => setHover(n ? n.id : null)}
            onNodeClick={(n) => { if (n.id !== 'C001') setSel(n.id); fg.current?.centerAt(n.x, n.y, 600); fg.current?.zoom(2.6, 600) }}
            onBackgroundClick={() => setSel(null)}
            nodeCanvasObject={(n, ctx, scale) => {
              const r = n.tier === 0 ? 9 : 2.5 + Math.min(5, (n.degree || 1) * 0.45)
              const dim = neighbours.size > 0 && !neighbours.has(n.id)
              const c = color(n)
              ctx.globalAlpha = dim ? 0.15 : 1
              if (n.band === 'High' || n.band === 'Critical') {
                ctx.beginPath(); ctx.arc(n.x, n.y, r + 3.5, 0, 2 * Math.PI)
                ctx.fillStyle = 'rgba(244,63,94,0.25)'; ctx.fill()
              }
              ctx.beginPath(); ctx.arc(n.x, n.y, r, 0, 2 * Math.PI)
              ctx.shadowColor = c; ctx.shadowBlur = n.tier === 0 ? 25 : 8
              ctx.fillStyle = c; ctx.fill(); ctx.shadowBlur = 0
              if (n.id === sel) { ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5 / scale * 2; ctx.stroke() }
              if (n.tier === 0 || scale > 2.2 || n.id === hover) {
                ctx.font = `${n.tier === 0 ? 12 : 10 / Math.max(1, scale / 2.2)}px Inter`
                ctx.fillStyle = '#e7eefb'; ctx.textAlign = 'center'
                ctx.fillText(n.tier === 0 ? 'MANUFACTURER' : (n.label || '').replace(' (Synthetic)', ''), n.x, n.y + r + 9)
              }
              ctx.globalAlpha = 1
            }}
          />
        )}
        <div className="card graph-legend">
          {[0, 1, 2, 3].map((t) => <span key={t}><span className="legend-dot" style={{ background: TIER_COLORS[t] }} />{t === 0 ? 'Manufacturer' : `Tier ${t}`}</span>)}
          <span><span className="legend-dot" style={{ background: 'rgba(244,63,94,0.5)' }} />Elevated risk halo</span>
          <span style={{ color: '#a78bfa' }}>- - Predicted (unverified)</span>
          {data && <span className="dim">{data.nodes.length} nodes · {data.links.length} edges</span>}
        </div>
        <AnimatePresence>
          {sel && (
            <motion.div className="card graph-panel" initial={{ opacity: 0, x: 30 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: 30 }}>
              {!detail ? <Loader /> : <>
                <div className="card-h"><h3>{detail.supplier.supplier_name.replace(' (Synthetic)', '')}</h3><Badge band={detail.risk.band} /></div>
                <div className="small muted" style={{ marginBottom: 10 }}>{detail.supplier.supplier_id} · Tier {detail.supplier.tier} · {detail.supplier.material_name} · {detail.supplier.region}</div>
                <dl className="kv">
                  <dt>Calibrated risk</dt><dd>{pct(detail.risk.calibrated_probability)}</dd>
                  <dt>Exposure</dt><dd>{compactInr(detail.risk.exposure_inr)}</dd>
                  <dt>Degree</dt><dd>{detail.centrality.degree.toFixed(4)}</dd>
                  <dt>Betweenness</dt><dd>{detail.centrality.betweenness.toFixed(5)}</dd>
                  <dt>PageRank</dt><dd>{detail.centrality.pagerank.toFixed(5)}</dd>
                  <dt>Downstream reach</dt><dd>{detail.centrality.downstream_reach} nodes</dd>
                </dl>
                <h4 style={{ margin: '14px 0 6px' }}>Upstream</h4>
                {detail.upstream.length ? detail.upstream.map((u) => <div key={u.supplier_id + u.material_id} className="small">{u.supplier_name.replace(' (Synthetic)', '')} <span className="dim">{u.material_id} {pct(u.dependency_pct, 0)}</span></div>) : <div className="small dim">None observed — blind spot.</div>}
                <h4 style={{ margin: '14px 0 6px' }}>Downstream</h4>
                {detail.downstream.map((u, i) => <div key={i} className="small">{u.buyer_id} <span className="dim">{u.material_id} {pct(u.dependency_pct, 0)}</span></div>)}
                <h4 style={{ margin: '14px 0 6px' }}>Alternatives</h4>
                {detail.alternatives.slice(0, 5).map((a) => <div key={a.supplier_id} className="small">{a.supplier_name.replace(' (Synthetic)', '')} <span className="dim">risk {pct(a.risk)}</span></div>)}
              </>}
            </motion.div>)}
        </AnimatePresence>
      </div>
    </>
  )
}
