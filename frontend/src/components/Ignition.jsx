export default function Ignition({ onIgnite, booting, lines, error, apiBase }) {
  return (
    <div className="ignition">
      <div style={{ maxWidth: 620, width: '100%' }}>
        <h1>CLOUD<span>OPTIMA</span></h1>
        <p>Predictive · Reinforcement-Learned · Multi-Cloud</p>

        <p style={{
          color: 'var(--text-secondary)', fontFamily: 'var(--sans)', letterSpacing: 0,
          textTransform: 'none', fontSize: '0.92rem', maxWidth: 500, margin: '18px auto 26px',
        }}>
          An XGBoost demand forecaster feeds a Deep Q-Network that sizes a simulated
          multi-cloud fleet, with SHAP attribution on every prediction.
        </p>

        {!booting && (
          <button className="btn btn-primary" onClick={onIgnite} style={{ padding: '12px 26px', fontSize: '0.92rem' }}>
            Initialise engine
          </button>
        )}

        {booting && (
          <div className="panel" style={{ textAlign: 'left', marginTop: 18 }}>
            <div className="panel-title">
              Boot sequence <span className="spinner" />
            </div>
            {lines.map((l, i) => (
              <div className="boot-line" key={i}>&gt; {l}</div>
            ))}
          </div>
        )}

        {error && (
          <div className="banner banner-error" style={{ textAlign: 'left', marginTop: 18 }}>
            <strong>Could not reach the API.</strong>
            <div style={{ marginTop: 6 }}>{error}</div>
            <div className="panel-note">
              Endpoint: <code className="mono">{apiBase}</code><br />
              If the backend is deployed on a Render free instance it sleeps when idle and
              the first request can take up to a minute — press initialise again.
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
