import React, { useEffect, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { Bot, Send } from 'lucide-react'
import { api } from '../api.js'
import { PageHeader } from '../components/ui.jsx'

const SUGGEST = ['Which supplier is most critical?', 'What happens if S0122 fails for 60 days?', 'Which raw materials are at risk?',
  'What is the current inventory coverage?', 'Which alternative supplier is suitable for S0003?', 'Compare mild and severe disruption scenarios',
  'Why was S0122 classified this way?', 'What should we do about S0041?']

export default function Assistant() {
  const [msgs, setMsgs] = useState([{ role: 'bot', text: 'Hello — I am the Supply Chain Black Box analyst. I answer only from the database, the calibrated risk model, the cascade simulator and the recovery optimiser, and show the evidence behind every answer.' }])
  const [q, setQ] = useState('')
  const [busy, setBusy] = useState(false)
  const end = useRef(null)
  useEffect(() => { end.current?.scrollIntoView({ behavior: 'smooth' }) }, [msgs, busy])
  const ask = async (text) => {
    const t = (text ?? q).trim()
    if (!t || busy) return
    setQ('')
    setMsgs((m) => [...m, { role: 'user', text: t }])
    setBusy(true)
    try {
      const r = await api('/api/chat/query', { method: 'POST', body: { question: t } })
      setMsgs((m) => [...m, { role: 'bot', text: r.answer, engine: r.engine, calls: r.tool_calls }])
    } catch (e) {
      setMsgs((m) => [...m, { role: 'bot', text: `Error: ${e.message}` }])
    } finally { setBusy(false) }
  }
  return (
    <>
      <PageHeader title="AI Assistant" desc="Query understanding → tool selection → database retrieval → model inference → simulation / optimisation → evidence-backed answer. Gemini function-calling is used when GEMINI_API_KEY is configured; otherwise a deterministic intent router calls the same tools." />
      <div className="card chat">
        <div className="messages">
          <AnimatePresence initial={false}>
            {msgs.map((m, i) => (
              <motion.div key={i} className={`msg ${m.role}`} initial={{ opacity: 0, y: 10, scale: 0.98 }} animate={{ opacity: 1, y: 0, scale: 1 }} transition={{ duration: 0.3 }}>
                {m.role === 'bot' && <Bot size={14} color="#22d3ee" style={{ verticalAlign: -2, marginRight: 6 }} />}{m.text}
                {m.engine && <div className="meta">engine: {m.engine} · tools: {(m.calls || []).map((c) => c.tool).join(', ') || 'none'}</div>}
                {m.calls?.length > 0 && <details className="evidence"><summary>Show evidence (tool outputs)</summary><pre>{JSON.stringify(m.calls, null, 2)}</pre></details>}
              </motion.div>))}
          </AnimatePresence>
          {busy && <div className="msg bot typing"><span /><span /><span /></div>}
          <div ref={end} />
        </div>
        <div>
          <div className="suggest">{SUGGEST.map((s) => <button key={s} onClick={() => ask(s)}>{s}</button>)}</div>
          <form className="row" onSubmit={(e) => { e.preventDefault(); ask() }} style={{ flexWrap: 'nowrap' }}>
            <input placeholder="Ask about suppliers, materials, disruptions, scenarios…" value={q} onChange={(e) => setQ(e.target.value)} />
            <button className="btn primary" disabled={busy}><Send size={15} />Send</button>
          </form>
        </div>
      </div>
    </>
  )
}
