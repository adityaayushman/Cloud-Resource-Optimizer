import { useEffect, useRef, useState } from 'react'
import { getApiBase, setApiBase } from '../lib/api'
import HeroCanvas from './HeroCanvas'

const WORDMARK = 'CLOUDOPTIMA'
const ACCENT_FROM = 5                    // "OPTIMA" takes the accent colour

const CAPABILITIES = [
  { k: 'Forecast', v: 'XGBoost · LR · persistence' },
  { k: 'Control', v: 'Deep Q-Network, NumPy' },
  { k: 'Placement', v: 'AWS · Azure · GCP' },
  { k: 'Evidence', v: '5 workloads · 60 tests' },
]

export default function Ignition({ onIgnite, booting, lines, error,
                                   needsEndpoint, onEndpointSet }) {
  const [endpoint, setEndpoint] = useState(getApiBase())
  const [saved, setSaved] = useState(false)
  const logRef = useRef(null)

  // Keep the newest boot line in view without yanking the whole page.
  useEffect(() => {
    const el = logRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [lines])

  const apply = (e) => {
    e.preventDefault()
    setApiBase(endpoint)
    setSaved(true)
    onEndpointSet?.()
    onIgnite()
  }

  return (
    <div className="ignition">
      <HeroCanvas intensity={booting ? 1.35 : 1} />
      <div className="ignition-veil" aria-hidden="true" />

      <div className="ignition-inner">
        <div className="ignition-eyebrow reveal" style={{ '--d': '0ms' }}>
          <span className="pulse-dot" aria-hidden="true" />
          Cloud Computing Resource Optimizer
        </div>

        <h1 className="wordmark" aria-label="CloudOptima">
          {WORDMARK.split('').map((ch, i) => (
            <span
              key={`${ch}-${i}`}
              className={i >= ACCENT_FROM ? 'accent' : undefined}
              style={{ '--d': `${120 + i * 38}ms` }}
              aria-hidden="true"
            >
              {ch}
            </span>
          ))}
        </h1>

        <p className="ignition-tagline reveal" style={{ '--d': '640ms' }}>
          Predictive · Reinforcement-Learned · Multi-Cloud
        </p>

        <p className="ignition-lede reveal" style={{ '--d': '740ms' }}>
          A demand forecaster feeds a Deep Q-Network that sizes a simulated
          multi-cloud fleet, with exact SHAP attribution on every prediction — and
          a persistence baseline beside every claim, because on some workloads
          that baseline wins.
        </p>

        <ul className="capability-row reveal" style={{ '--d': '840ms' }}>
          {CAPABILITIES.map((c) => (
            <li key={c.k}>
              <span className="capability-k">{c.k}</span>
              <span className="capability-v">{c.v}</span>
            </li>
          ))}
        </ul>

        {!booting && !error && !needsEndpoint && (
          <div className="reveal" style={{ '--d': '960ms' }}>
            <button className="btn btn-primary btn-lg btn-glow" onClick={onIgnite}>
              <span className="btn-lg-label">Initialise engine</span>
              <span className="btn-lg-arrow" aria-hidden="true">→</span>
            </button>
          </div>
        )}

        {booting && (
          <div className="boot-card" role="status" aria-live="polite">
            <div className="boot-head">
              <span className="boot-rings" aria-hidden="true">
                <span /><span /><span />
              </span>
              Boot sequence
            </div>
            <div className="boot-log" ref={logRef}>
              {lines.map((l, i) => (
                <div className="boot-line" key={i} style={{ '--d': `${i * 60}ms` }}>
                  <span className="boot-caret" aria-hidden="true">▸</span>
                  {l}
                </div>
              ))}
              <div className="boot-line boot-cursor" aria-hidden="true">
                <span className="boot-caret">▸</span><span className="caret-blink" />
              </div>
            </div>
          </div>
        )}

        {needsEndpoint && !error && (
          <div className="boot-card">
            <div className="boot-head">Connect your backend</div>
            <p className="boot-error-detail">
              This dashboard is live, but it does not know where its API is yet.
              Paste the URL of your deployed backend — or set{' '}
              <code className="mono">VITE_API_BASE_URL</code> in the hosting
              project and redeploy to skip this step for everyone.
            </p>
            <form onSubmit={apply}>
              <div className="field">
                <label htmlFor="endpoint-setup">API endpoint</label>
                <input
                  id="endpoint-setup" type="url" value={endpoint} spellCheck="false"
                  onChange={(e) => { setEndpoint(e.target.value); setSaved(false) }}
                  placeholder="https://your-service.onrender.com"
                  required
                />
              </div>
              <div className="btn-row">
                <button className="btn btn-primary" type="submit">Connect</button>
                {saved && <span className="tag tag-ok">saved to this browser</span>}
              </div>
            </form>
            <p className="panel-note">
              The value is kept in this browser only. A link of the form{' '}
              <code className="mono">?api=https://…</code> also works, which is
              handy for sharing a dashboard pointed at a particular backend.
            </p>
          </div>
        )}

        {error && (
          <div className="boot-card boot-card-error" role="alert">
            <div className="boot-head boot-head-error">Could not reach the API</div>
            <p className="boot-error-detail">{error}</p>

            {/* The endpoint is settable here rather than only at build time.
                A deployed dashboard is often live before its backend is, and
                Vite freezes VITE_* variables into the bundle - without this the
                only way to point the page at a backend is a rebuild. */}
            <form onSubmit={apply}>
              <div className="field">
                <label htmlFor="endpoint">API endpoint</label>
                <input
                  id="endpoint" type="url" value={endpoint} spellCheck="false"
                  onChange={(e) => { setEndpoint(e.target.value); setSaved(false) }}
                  placeholder="https://your-service.onrender.com"
                />
              </div>
              <div className="btn-row">
                <button className="btn btn-primary" type="submit">Save &amp; retry</button>
                {saved && <span className="tag tag-ok">saved to this browser</span>}
              </div>
            </form>

            <p className="panel-note">
              Running locally? Start the backend with{' '}
              <code className="mono">uvicorn app.main:app --host 127.0.0.1 --port 8000</code>{' '}
              and use <code className="mono">http://127.0.0.1:8000</code> — on Windows,
              &ldquo;localhost&rdquo; resolves to IPv6 first and will not reach an
              IPv4-bound server. On a free Render instance the service sleeps when
              idle and the first request can take up to a minute — press retry.
            </p>
          </div>
        )}

        <p className="ignition-credit reveal" style={{ '--d': '1080ms' }}>
          Aditya Ayushman Sahoo · Sarthak Kar — SRM Institute of Science and Technology
        </p>
      </div>
    </div>
  )
}
