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

const BASE = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000').replace(/\/$/, '')

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
    const res = await fetch(`${BASE}${path}`, {
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
      `Cannot reach the API at ${BASE}. Check VITE_API_BASE_URL and that the backend is running.`,
      0, null,
    )
  } finally {
    clearTimeout(timer)
  }
}

export const apiBase = BASE

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
