import { useState } from 'react'
import { getApiBase, setApiBase } from '../lib/api'

export default function Ignition({ onIgnite, booting, lines, error }) {
  const [endpoint, setEndpoint] = useState(getApiBase())
  const [saved, setSaved] = useState(false)

  const apply = (e) => {
    e.preventDefault()
    setApiBase(endpoint)
    setSaved(true)
    onIgnite()
  }

  return (
    <div className="ignition">
      <div style={{ maxWidth: 640, width: '100%' }}>
        <h1>CLOUD<span>OPTIMA</span></h1>
        <p>Predictive · Reinforcement-Learned · Multi-Cloud</p>

        <p style={{
          color: 'var(--text-secondary)', fontFamily: 'var(--sans)', letterSpacing: 0,
          textTransform: 'none', fontSize: '0.92rem', maxWidth: 500, margin: '18px auto 26px',
        }}>
          An XGBoost demand forecaster feeds a Deep Q-Network that sizes a
          simulated multi-cloud fleet, with SHAP attribution on every prediction.
        </p>

        {!booting && (
          <button
            className="btn btn-primary" onClick={onIgnite}
            style={{ padding: '12px 26px', fontSize: '0.92rem' }}
          >
            Initialise engine
          </button>
        )}

        {booting && (
          <div className="panel" style={{ textAlign: 'left', marginTop: 18 }}>
            <div className="panel-title">Boot sequence <span className="spinner" /></div>
            {lines.map((l, i) => <div className="boot-line" key={i}>&gt; {l}</div>)}
          </div>
        )}

        {error && (
          <div className="banner banner-error" style={{ textAlign: 'left', marginTop: 18 }}>
            <strong>Could not reach the API.</strong>
            <div style={{ marginTop: 6 }}>{error}</div>

            {/* The endpoint is settable here rather than only at build time.
                A deployed dashboard is often live before its backend is, and
                Vite freezes VITE_* variables into the bundle - without this the
                only way to point the page at a backend is a rebuild. */}
            <form onSubmit={apply} style={{ marginTop: 14 }}>
              <div className="field" style={{ marginBottom: 8 }}>
                <label htmlFor="endpoint">API endpoint</label>
                <input
                  id="endpoint" type="url" value={endpoint} spellCheck="false"
                  onChange={(e) => { setEndpoint(e.target.value); setSaved(false) }}
                  placeholder="https://your-service.onrender.com"
                />
              </div>
              <div className="btn-row">
                <button className="btn btn-primary" type="submit">
                  Save &amp; retry
                </button>
                {saved && <span className="tag tag-ok">saved to this browser</span>}
              </div>
            </form>

            <div className="panel-note" style={{ marginTop: 12 }}>
              Running locally? Start the backend with{' '}
              <code className="mono">uvicorn app.main:app --host 127.0.0.1 --port 8000</code>{' '}
              and use <code className="mono">http://127.0.0.1:8000</code> — on Windows,
              &ldquo;localhost&rdquo; resolves to IPv6 first and will not reach an
              IPv4-bound server.<br />
              On a free Render instance the service sleeps when idle and the first
              request can take up to a minute — press retry.
            </div>
          </div>
        )}

        <p className="panel-note" style={{ marginTop: 22, letterSpacing: 0, textTransform: 'none' }}>
          Aditya Ayushman Sahoo · Sarthak Kar — SRM Institute of Science and Technology
        </p>
      </div>
    </div>
  )
}
