import React, { useEffect, useRef, useState } from 'react'
import { motion, animate } from 'framer-motion'

export const fade = {
  initial: { opacity: 0, y: 14 },
  animate: { opacity: 1, y: 0, transition: { duration: 0.45, ease: [0.22, 1, 0.36, 1] } },
}

export function Card({ title, sub, right, children, className = '', delay = 0, style }) {
  return (
    <motion.div className={`card glow ${className}`} style={style}
      initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay, ease: [0.22, 1, 0.36, 1] }}>
      {(title || right) && (
        <div className="card-h">
          <div><h3>{title}</h3>{sub && <div className="sub">{sub}</div>}</div>
          {right}
        </div>
      )}
      {children}
    </motion.div>
  )
}

export function Counter({ value, format = (v) => Math.round(v).toLocaleString('en-IN'), duration = 1.2 }) {
  const ref = useRef(null)
  const prev = useRef(0)
  useEffect(() => {
    if (value === null || value === undefined || Number.isNaN(value)) { if (ref.current) ref.current.textContent = '—'; return }
    const c = animate(prev.current, value, {
      duration, ease: [0.22, 1, 0.36, 1],
      onUpdate: (v) => { if (ref.current) ref.current.textContent = format(v) },
    })
    prev.current = value
    return () => c.stop()
  }, [value])
  return <span ref={ref}>{format(0)}</span>
}

export function Kpi({ label, value, format, foot, icon: Icon, color = 'var(--accent)', delay = 0 }) {
  return (
    <motion.div className="card kpi glow" initial={{ opacity: 0, y: 16, scale: 0.98 }} animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.5, delay }}>
      {Icon && <div className="icon" style={{ background: `color-mix(in srgb, ${color} 18%, transparent)`, color }}><Icon size={18} /></div>}
      <div className="label">{label}</div>
      <div className="value" style={{ color: 'var(--text)' }}><Counter value={value} format={format} /></div>
      {foot && <div className="foot">{foot}</div>}
      <div className="bar" style={{ background: `linear-gradient(90deg, ${color}, transparent)` }} />
    </motion.div>
  )
}

export function PageHeader({ title, desc, children, real = false }) {
  return (
    <div className="topbar">
      <motion.div className="page-title" initial={{ opacity: 0, x: -12 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: 0.4 }}>
        <h2>{title}</h2>
        {desc && <p>{desc}</p>}
      </motion.div>
      <div className="top-actions">
        {children}
        <span className={`chip-synthetic ${real ? 'chip-real' : ''}`}><span className="pulse" />{real ? 'REAL PUBLIC DATA' : 'SYNTHETIC DEMO DATA'}</span>
      </div>
    </div>
  )
}

export function Loader({ text = 'Loading…' }) {
  return <div className="loader"><div className="spinner" /><div className="small">{text}</div></div>
}

export function ErrorBox({ error }) {
  if (!error) return null
  return <div className="err">⚠ {String(error.message || error)}</div>
}

export function Badge({ band, children }) {
  return <span className={`badge b-${band}`}>{children || band}</span>
}

export function RiskBar({ p, max = 0.4 }) {
  const w = Math.min(100, (100 * (p || 0)) / max)
  return <span className="riskbar"><i style={{ width: `${w}%` }} /></span>
}

export function useApi(fn, deps = []) {
  const [state, set] = useState({ data: null, error: null, loading: true })
  useEffect(() => {
    let alive = true
    set((s) => ({ ...s, loading: true, error: null }))
    fn().then((d) => alive && set({ data: d, error: null, loading: false }))
      .catch((e) => alive && set({ data: null, error: e, loading: false }))
    return () => { alive = false }
  }, deps)
  return state
}

export const TIER_COLORS = { 0: '#ffffff', 1: '#4f8dff', 2: '#22d3ee', 3: '#a78bfa' }
export const BAND_COLORS = { Low: '#34d399', Medium: '#fbbf24', High: '#f87171', Critical: '#f43f5e', Manufacturer: '#ffffff' }
export const SERIES = ['#4f8dff', '#22d3ee', '#a78bfa', '#34d399', '#fbbf24', '#f87171', '#f472b6', '#94a3b8']
export const POLICY_COLORS = { 'A: No intervention': '#f87171', 'B: Rule-based': '#fbbf24', 'C: Optimisation': '#34d399' }

export const tooltipStyle = {
  contentStyle: { background: 'rgba(8,20,48,0.96)', border: '1px solid rgba(96,148,255,0.42)', borderRadius: 10, fontSize: 12 },
  labelStyle: { color: '#e7eefb', fontWeight: 600 },
  itemStyle: { color: '#c9d6ee' },
}
