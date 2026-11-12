import React, { useEffect, useState } from 'react'
import { NavLink, Navigate, Route, Routes, useLocation, useNavigate } from 'react-router-dom'
import { AnimatePresence, motion } from 'framer-motion'
import {
  Activity, Bot, Boxes, Database, FileBarChart, GitBranch, LayoutDashboard, LogOut, Menu, Network, ShieldAlert, Sparkles, Wrench, X, Zap,
} from 'lucide-react'
import { getToken, getUser, setSession } from './api.js'
import Login from './pages/Login.jsx'
import Dashboard from './pages/Dashboard.jsx'
import Suppliers from './pages/Suppliers.jsx'
import NetworkExplorer from './pages/NetworkExplorer.jsx'
import RiskCentre from './pages/RiskCentre.jsx'
import Simulator from './pages/Simulator.jsx'
import Scenarios from './pages/Scenarios.jsx'
import Impact from './pages/Impact.jsx'
import Recovery from './pages/Recovery.jsx'
import Assistant from './pages/Assistant.jsx'
import DataManagement from './pages/DataManagement.jsx'
import Reports from './pages/Reports.jsx'

const NAV = [
  ['Command', [['/', 'Executive Dashboard', LayoutDashboard], ['/suppliers', 'Supplier Directory', Boxes], ['/network', 'Network Explorer', Network]]],
  ['Intelligence', [['/risk', 'Risk Prediction Centre', ShieldAlert], ['/simulator', 'Disruption Simulator', Zap],
    ['/scenarios', 'Scenario Comparison', GitBranch], ['/impact', 'Impact Analysis', Activity], ['/recovery', 'Recovery Planner', Wrench]]],
  ['Workspace', [['/assistant', 'AI Assistant', Bot], ['/data', 'Data Management', Database], ['/reports', 'Reports & History', FileBarChart]]],
]

function Shell({ onLogout }) {
  const loc = useLocation()
  const user = getUser()
  const [open, setOpen] = useState(false)
  useEffect(() => { setOpen(false) }, [loc.pathname])
  return (
    <div className="shell">
      <div className="mobile-bar">
        <div className="brand" style={{ padding: 0 }}><div className="brand-logo"><span /></div><b style={{ fontSize: 14 }}>SUPPLY CHAIN BLACK BOX</b></div>
        <button className="btn small" onClick={() => setOpen(!open)} aria-label="Toggle menu">{open ? <X size={16} /> : <Menu size={16} />}</button>
      </div>
      <aside className={`sidebar ${open ? 'open' : ''}`}>
        <div className="brand">
          <div className="brand-logo"><span /></div>
          <div><h1>SUPPLY CHAIN<br />BLACK BOX</h1><small>Control Tower</small></div>
        </div>
        <nav className="nav">
          {NAV.map(([sec, items]) => (
            <div key={sec}>
              <div className="nav-section">{sec}</div>
              {items.map(([to, label, Icon]) => (
                <NavLink key={to} to={to} end={to === '/'}><Icon size={17} />{label}</NavLink>
              ))}
            </div>
          ))}
        </nav>
        <div className="sidebar-foot">
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}><Sparkles size={14} color="#22d3ee" />
            <b style={{ color: 'var(--text)' }}>Cocoa &amp; Malt Foods</b></div>
          Fictional FMCG manufacturer · 500 suppliers · 3 tiers. All supplier data are synthetic; DataCo &amp; SupplyGraph experiments use real public data.
          <div style={{ marginTop: 10, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span className="dim" style={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>{user.email || ''}</span>
            <button className="btn small ghost" onClick={onLogout} title="Log out"><LogOut size={14} /></button>
          </div>
        </div>
      </aside>
      <main className="main">
        <AnimatePresence mode="wait">
          <motion.div key={loc.pathname} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.3 }}>
            <Routes location={loc}>
              <Route path="/" element={<Dashboard />} />
              <Route path="/suppliers" element={<Suppliers />} />
              <Route path="/network" element={<NetworkExplorer />} />
              <Route path="/risk" element={<RiskCentre />} />
              <Route path="/simulator" element={<Simulator />} />
              <Route path="/scenarios" element={<Scenarios />} />
              <Route path="/impact" element={<Impact />} />
              <Route path="/recovery" element={<Recovery />} />
              <Route path="/assistant" element={<Assistant />} />
              <Route path="/data" element={<DataManagement />} />
              <Route path="/reports" element={<Reports />} />
              <Route path="*" element={<Navigate to="/" />} />
            </Routes>
          </motion.div>
        </AnimatePresence>
      </main>
    </div>
  )
}

export default function App() {
  const [authed, setAuthed] = useState(!!getToken())
  const nav = useNavigate()
  useEffect(() => {
    const h = () => setAuthed(false)
    window.addEventListener('scbb-logout', h)
    return () => window.removeEventListener('scbb-logout', h)
  }, [])
  if (!authed) return <Login onAuthed={() => { setAuthed(true); nav('/') }} />
  return <Shell onLogout={() => { setSession(null); setAuthed(false) }} />
}
