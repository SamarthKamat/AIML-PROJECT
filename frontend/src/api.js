// Thin fetch wrapper: JWT from localStorage, JSON in/out, readable errors.
const BASE = import.meta.env.VITE_API_BASE || ''

export function getToken() {
  try { return localStorage.getItem('scbb_token') } catch { return null }
}
export function setSession(token, user) {
  try {
    if (token) { localStorage.setItem('scbb_token', token); localStorage.setItem('scbb_user', JSON.stringify(user || {})) }
    else { localStorage.removeItem('scbb_token'); localStorage.removeItem('scbb_user') }
  } catch { /* storage unavailable */ }
}
export function getUser() {
  try { return JSON.parse(localStorage.getItem('scbb_user') || '{}') } catch { return {} }
}

export async function api(path, { method = 'GET', body, form } = {}) {
  const headers = {}
  const t = getToken()
  if (t) headers.Authorization = `Bearer ${t}`
  let payload
  if (form) payload = form
  else if (body !== undefined) { headers['Content-Type'] = 'application/json'; payload = JSON.stringify(body) }
  const res = await fetch(BASE + path, { method, headers, body: payload })
  let data = null
  let parseError = false
  try { data = await res.json() } catch { parseError = true }
  if (res.ok && parseError) throw new Error(`Invalid JSON from ${path}`)
  if (res.status === 401 && !path.startsWith('/api/auth')) {
    setSession(null)
    window.dispatchEvent(new Event('scbb-logout'))
  }
  if (!res.ok) throw new Error((data && (data.message || data.error)) || `HTTP ${res.status}`)
  return data
}

export const inr = (v, d = 0) => (v === null || v === undefined || Number.isNaN(v)) ? '—'
  : '₹' + Number(v).toLocaleString('en-IN', { maximumFractionDigits: d })
export const compactInr = (v) => {
  if (v === null || v === undefined) return '—'
  const a = Math.abs(v)
  if (a >= 1e7) return '₹' + (v / 1e7).toFixed(2) + ' Cr'
  if (a >= 1e5) return '₹' + (v / 1e5).toFixed(2) + ' L'
  if (a >= 1e3) return '₹' + (v / 1e3).toFixed(1) + 'k'
  return '₹' + Number(v).toFixed(0)
}
export const pct = (v, d = 1) => (v === null || v === undefined) ? '—' : (100 * v).toFixed(d) + '%'
export const num = (v, d = 0) => (v === null || v === undefined) ? '—' : Number(v).toLocaleString('en-IN', { maximumFractionDigits: d })
