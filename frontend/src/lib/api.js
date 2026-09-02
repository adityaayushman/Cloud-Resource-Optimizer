/* Thin API client for the CloudOptima backend on Render.
 *
 * Two things worth knowing about the deployment:
 *  - Render free instances sleep after inactivity, so the first request of a
 *    session can take 30-60s. `wakeUp` surfaces that as a state the UI can show
 *    instead of an unexplained hang.
 *  - Every error is normalised to an ApiError carrying the backend's `detail`
 *    string, because the backend deliberately returns actionable messages
 *    (e.g. "Model is not trained. Run scripts/train.py").
 */

/* Resolving the API base URL.
 *
 * Vite inlines `VITE_*` variables at BUILD time, so a deployed bundle has its
 * endpoint frozen in. That is awkward here: the dashboard is deployed before
 * the API is, and pointing it at a backend would otherwise mean a rebuild.
 *
 * Resolution order, first match wins:
 *   1. ?api=<url>            - shareable link to a specific backend
 *   2. localStorage override - what the user typed on the ignition screen
 *   3. VITE_API_BASE_URL     - build-time default (local dev, CI)
 *   4. same-origin :8000     - sensible guess when developing locally
 */
const STORAGE_KEY = 'cloudoptima.apiBaseUrl'

function readStored() {
  try { return localStorage.getItem(STORAGE_KEY) } catch { return null }
}

function resolveBase() {
  let fromQuery = null
  try {
    fromQuery = new URLSearchParams(window.location.search).get('api')
  } catch { /* non-browser context */ }

  if (fromQuery) {
    try { localStorage.setItem(STORAGE_KEY, fromQuery) } catch { /* private mode */ }
    return fromQuery
  }
  return readStored()
    || import.meta.env.VITE_API_BASE_URL
    || `${window.location.protocol}//${window.location.hostname}:8000`
}

let BASE = resolveBase().replace(/\/$/, '')

export function getApiBase() { return BASE }

export function setApiBase(url) {
  BASE = String(url || '').trim().replace(/\/$/, '')
  try { localStorage.setItem(STORAGE_KEY, BASE) } catch { /* private mode */ }
  return BASE
}

export function clearApiBase() {
  try { localStorage.removeItem(STORAGE_KEY) } catch { /* private mode */ }
  BASE = resolveBase().replace(/\/$/, '')
  return BASE
}

export class ApiError extends Error {
  constructor(message, status, payload) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.payload = payload
  }
}

async function request(path, { method = 'GET', body, timeout = 90000 } = {}) {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeout)
  try {
    const res = await fetch(`${getApiBase()}${path}`, {
      method,
      headers: body ? { 'Content-Type': 'application/json' } : undefined,
      body: body ? JSON.stringify(body) : undefined,
      signal: controller.signal,
    })
    const text = await res.text()
    let payload = null
    try { payload = text ? JSON.parse(text) : null } catch { payload = { detail: text } }

    if (!res.ok) {
      throw new ApiError(payload?.detail || `Request failed (${res.status})`, res.status, payload)
    }
    return payload
  } catch (err) {
    if (err.name === 'AbortError') {
      throw new ApiError(
        'The API did not respond in time. A sleeping free-tier instance can take up to a minute to wake.',
        408, null,
      )
    }
    if (err instanceof ApiError) throw err
    throw new ApiError(
      `Cannot reach the API at ${getApiBase()}.`,
      0, null,
    )
  } finally {
    clearTimeout(timer)
  }
}


export const api = {
  health: () => request('/api/health'),
  meta: () => request('/api/meta'),

  modelMetrics: () => request('/api/models/metrics'),
  predict: (payload) => request('/api/models/predict', { method: 'POST', body: payload }),
  workloadHistory: (limit = 288, offset = 0) =>
    request(`/api/workload/history?limit=${limit}&offset=${offset}`),

  scoreProviders: (payload) => request('/api/providers/score', { method: 'POST', body: payload }),
  priceSeries: (hours = 24, points = 96) =>
    request(`/api/providers/price-series?hours=${hours}&points=${points}`),

  checkAnomaly: (payload) => request('/api/anomaly/check', { method: 'POST', body: payload }),

  createSession: (payload) => request('/api/session', { method: 'POST', body: payload }),
  getSession: (id) => request(`/api/session/${id}`),
  deleteSession: (id) => request(`/api/session/${id}`, { method: 'DELETE' }),
  step: (id, payload) => request(`/api/session/${id}/step`, { method: 'POST', body: payload }),
  inject: (id, payload) => request(`/api/session/${id}/inject`, { method: 'POST', body: payload }),
  scale: (id, payload) => request(`/api/session/${id}/scale`, { method: 'POST', body: payload }),
  decommission: (id, vmId) => request(`/api/session/${id}/vm/${vmId}`, { method: 'DELETE' }),
  fault: (id, vmId) => request(`/api/session/${id}/fault/${vmId}`, { method: 'POST' }),
  scenario: (id, scenario) =>
    request(`/api/session/${id}/scenario`, { method: 'POST', body: { scenario } }),
  explain: (id) => request(`/api/session/${id}/explain`),
  rlState: (id) => request(`/api/session/${id}/rl`),

  ablation: (payload) => request('/api/ablation', { method: 'POST', body: payload, timeout: 180000 }),
  ablationCached: () => request('/api/ablation/cached'),
}

/** Poll /api/health until the (possibly sleeping) instance answers. */
export async function wakeUp({ attempts = 6, onAttempt } = {}) {
  let lastError = null
  for (let i = 0; i < attempts; i += 1) {
    try {
      onAttempt?.(i + 1, attempts)
      return await request('/api/health', { timeout: 20000 })
    } catch (err) {
      lastError = err
      await new Promise((r) => setTimeout(r, 2500))
    }
  }
  throw lastError
}
