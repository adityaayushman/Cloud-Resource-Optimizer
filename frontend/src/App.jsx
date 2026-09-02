import { useCallback, useEffect, useRef, useState } from 'react'
import { api, apiBase, ApiError, wakeUp } from './lib/api'
import Ignition from './components/Ignition'
import CommandCenter from './pages/CommandCenter'
import Prediction from './pages/Prediction'
import MultiCloud from './pages/MultiCloud'
import Results from './pages/Results'

const PAGES = [
  { id: 'command', label: 'Command Center' },
  { id: 'prediction', label: 'Prediction & XAI' },
  { id: 'multicloud', label: 'Multi-Cloud' },
  { id: 'results', label: 'Results & Ablation' },
]

export default function App() {
  const [online, setOnline] = useState(false)
  const [booting, setBooting] = useState(false)
  const [bootLines, setBootLines] = useState([])
  const [health, setHealth] = useState(null)
  const [meta, setMeta] = useState(null)
  const [error, setError] = useState(null)

  const [page, setPage] = useState('command')
  const [session, setSession] = useState(null)
  const [busy, setBusy] = useState(false)

  const autoRef = useRef(null)
  const [auto, setAuto] = useState(false)

  const say = (line) => setBootLines((prev) => [...prev, line])

  const ignite = useCallback(async () => {
    setBooting(true)
    setError(null)
    setBootLines([])
    try {
      say('Establishing link to optimisation engine…')
      const h = await wakeUp({
        onAttempt: (n, total) => { if (n > 1) say(`Instance asleep — retry ${n}/${total}…`) },
      })
      setHealth(h)
      say(`Engine online (v${h.version}) — artifacts ${h.artifacts_ready ? 'ready' : 'INCOMPLETE'}`)

      const m = await api.meta()
      setMeta(m)
      say(`Loaded ${m.features.length} predictor features, ${m.rl_actions.length} RL actions`)

      say('Provisioning simulation session…')
      const s = await api.createSession({
        predictor_algo: 'xgboost',
        anomaly_method: 'isolation_forest',
        strategy: 'full',
        initial_fleet: 3,
      })
      setSession(s)
      say('Session ready. Entering command center.')
      setTimeout(() => { setOnline(true); setBooting(false) }, 450)
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err))
      setBooting(false)
    }
  }, [])

  const guard = useCallback(async (fn) => {
    setBusy(true)
    setError(null)
    try {
      const next = await fn()
      if (next) setSession(next)
      return next
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err))
      if (err instanceof ApiError && err.status === 404) setAuto(false)
      return null
    } finally {
      setBusy(false)
    }
  }, [])

  const step = useCallback(
    (ticks = 1) => guard(() => api.step(session.id, { ticks, train_rl: true })),
    [guard, session],
  )

  // Auto-advance loop. Chained with a timeout rather than setInterval so a slow
  // API response can never queue up overlapping requests.
  useEffect(() => {
    if (!auto || !session || busy) return undefined
    autoRef.current = setTimeout(() => { step(1) }, 1400)
    return () => clearTimeout(autoRef.current)
  }, [auto, session, busy, step])

  if (!online) {
    return (
      <Ignition
        onIgnite={ignite} booting={booting} lines={bootLines}
        error={error} apiBase={apiBase}
      />
    )
  }

  const shared = { session, setSession, guard, busy, meta, health, step, error }

  return (
    <div className="shell">
      <header className="topbar">
        <div>
          <div className="brand">CLOUD<span>OPTIMA</span></div>
          <div className="brand-sub">Cloud Resource Optimizer</div>
        </div>

        <nav className="nav" aria-label="Sections">
          {PAGES.map((p) => (
            <button
              key={p.id} onClick={() => setPage(p.id)}
              aria-current={page === p.id ? 'page' : undefined}
            >
              {p.label}
            </button>
          ))}
        </nav>

        <div style={{ marginLeft: 'auto' }} className="btn-row">
          {busy && <span className="spinner" aria-label="Working" />}
          <span className={`tag ${health?.artifacts_ready ? 'tag-ok' : 'tag-warn'}`}>
            {health?.artifacts_ready ? 'models ready' : 'degraded'}
          </span>
          <button className="btn" onClick={() => setAuto((a) => !a)} disabled={!session}>
            {auto ? '⏸ Pause' : '▶ Auto-run'}
          </button>
          <button className="btn btn-primary" onClick={() => step(1)} disabled={busy || !session}>
            Step
          </button>
        </div>
      </header>

      <main className="content">
        {error && (
          <div className="banner banner-error" role="alert">
            <strong>API error.</strong> {error}
          </div>
        )}
        {health && !health.artifacts_ready && (
          <div className="banner">
            Trained model artifacts are missing on the server, so predictions and the
            RL agent are unavailable. Run <code>python scripts/train.py</code> in the
            backend and redeploy.
          </div>
        )}

        {page === 'command' && <CommandCenter {...shared} />}
        {page === 'prediction' && <Prediction {...shared} />}
        {page === 'multicloud' && <MultiCloud {...shared} />}
        {page === 'results' && <Results {...shared} />}
      </main>
    </div>
  )
}
