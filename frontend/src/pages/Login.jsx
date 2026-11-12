import React, { useEffect, useRef, useState } from 'react'
import { motion } from 'framer-motion'
import { Brain, GitBranch, Network, ShieldCheck, Wrench } from 'lucide-react'
import { api, setSession } from '../api.js'

function NetworkCanvas() {
  const ref = useRef(null)
  useEffect(() => {
    const c = ref.current
    const ctx = c.getContext('2d')
    let w, h, raf
    const colors = ['#4f8dff', '#22d3ee', '#a78bfa']
    const nodes = Array.from({ length: 70 }, (_, i) => ({ x: Math.random(), y: Math.random(), vx: (Math.random() - 0.5) * 0.0006, vy: (Math.random() - 0.5) * 0.0006, t: i % 3 }))
    const resize = () => { w = c.width = c.offsetWidth * devicePixelRatio; h = c.height = c.offsetHeight * devicePixelRatio }
    resize()
    window.addEventListener('resize', resize)
    let tick = 0
    const draw = () => {
      tick++
      ctx.clearRect(0, 0, w, h)
      for (const n of nodes) { n.x += n.vx; n.y += n.vy; if (n.x < 0 || n.x > 1) n.vx *= -1; if (n.y < 0 || n.y > 1) n.vy *= -1 }
      for (let i = 0; i < nodes.length; i++) for (let j = i + 1; j < nodes.length; j++) {
        const a = nodes[i], b = nodes[j]
        const d = Math.hypot((a.x - b.x) * w, (a.y - b.y) * h)
        if (d < 170 * devicePixelRatio) {
          ctx.strokeStyle = `rgba(96,148,255,${0.22 * (1 - d / (170 * devicePixelRatio))})`
          ctx.lineWidth = devicePixelRatio
          ctx.beginPath(); ctx.moveTo(a.x * w, a.y * h); ctx.lineTo(b.x * w, b.y * h); ctx.stroke()
          if ((i + j + tick / 3) % 97 < 1.2) {
            const f = (tick % 60) / 60
            ctx.fillStyle = '#22d3ee'
            ctx.beginPath(); ctx.arc(a.x * w + (b.x - a.x) * w * f, a.y * h + (b.y - a.y) * h * f, 2 * devicePixelRatio, 0, 7); ctx.fill()
          }
        }
      }
      for (const n of nodes) {
        ctx.fillStyle = colors[n.t]
        ctx.shadowColor = colors[n.t]; ctx.shadowBlur = 12
        ctx.beginPath(); ctx.arc(n.x * w, n.y * h, (n.t === 0 ? 3.2 : 2.2) * devicePixelRatio, 0, 7); ctx.fill()
        ctx.shadowBlur = 0
      }
      raf = requestAnimationFrame(draw)
    }
    draw()
    return () => { cancelAnimationFrame(raf); window.removeEventListener('resize', resize) }
  }, [])
  return <canvas ref={ref} style={{ width: '100%', height: '100%' }} />
}

export default function Login({ onAuthed }) {
  const [mode, setMode] = useState('login')
  const [f, setF] = useState({ email: '', password: '', full_name: '' })
  const [err, setErr] = useState(null)
  const [busy, setBusy] = useState(false)
  const submit = async (e) => {
    e.preventDefault()
    setBusy(true); setErr(null)
    try {
      const r = await api(`/api/auth/${mode === 'login' ? 'login' : 'register'}`, { method: 'POST', body: f })
      setSession(r.token, r.user)
      onAuthed()
    } catch (e2) { setErr(e2.message) } finally { setBusy(false) }
  }
  const feats = [[Network, 'Multi-tier dependency discovery with graph link prediction'], [Brain, 'Calibrated 30-day supplier disruption probabilities'],
    [GitBranch, 'Cascading-disruption Monte Carlo over the supply network'], [Wrench, 'MILP recovery optimisation with alternative suppliers'],
    [ShieldCheck, 'Every recommendation traceable to model outputs']]
  return (
    <div className="auth">
      <div className="auth-hero">
        <NetworkCanvas />
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.7 }}>
          <span className="chip-synthetic"><span className="pulse" />AIMLBT · Goa Institute of Management</span>
          <h1><span className="grad">Supply Chain<br />Black Box</span></h1>
          <p>AI-powered hidden supplier dependency discovery, cascading risk prediction and intelligent supply chain recovery for a
            chocolate &amp; malt-beverage FMCG manufacturer.</p>
          <div className="feature-list">
            {feats.map(([I, t], i) => (
              <motion.div key={t} initial={{ opacity: 0, x: -14 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.3 + i * 0.1 }}>
                <I size={17} color="#22d3ee" />{t}
              </motion.div>
            ))}
          </div>
        </motion.div>
      </div>
      <div className="auth-form">
        <motion.form className="card auth-card" onSubmit={submit} initial={{ opacity: 0, scale: 0.96 }} animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5, delay: 0.15 }}>
          <div className="seg" style={{ marginBottom: 20 }}>
            <button type="button" className={mode === 'login' ? 'on' : ''} onClick={() => setMode('login')}>Sign in</button>
            <button type="button" className={mode === 'register' ? 'on' : ''} onClick={() => setMode('register')}>Create account</button>
          </div>
          <h3 style={{ margin: '0 0 4px', fontSize: 20 }}>{mode === 'login' ? 'Welcome back' : 'Create your analyst account'}</h3>
          <p className="muted small" style={{ marginTop: 0 }}>Access the supply chain control tower.</p>
          <div className="grid" style={{ gap: 14, marginTop: 18 }}>
            {mode === 'register' && <label className="field">Full name<input value={f.full_name} onChange={(e) => setF({ ...f, full_name: e.target.value })} /></label>}
            <label className="field">Email<input type="email" required value={f.email} onChange={(e) => setF({ ...f, email: e.target.value })} /></label>
            <label className="field">Password<input type="password" required minLength={8} value={f.password} onChange={(e) => setF({ ...f, password: e.target.value })} /></label>
            {err && <div className="err">{err}</div>}
            <button className="btn primary" disabled={busy} style={{ justifyContent: 'center', padding: 12 }}>
              {busy ? 'Please wait…' : mode === 'login' ? 'Sign in' : 'Create account'}
            </button>
            <p className="note" style={{ margin: 0 }}>Passwords are stored as salted hashes; sessions use 12-hour JWTs.</p>
          </div>
        </motion.form>
      </div>
    </div>
  )
}
