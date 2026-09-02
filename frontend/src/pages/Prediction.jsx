import { useEffect, useState } from 'react'
import { api } from '../lib/api'
import { BarChart, LineChart, StatTile } from '../components/charts'

const ALGOS = [
  { id: 'xgboost', label: 'XGBoost' },
  { id: 'rf', label: 'Random Forest' },
  { id: 'lr', label: 'Linear Regression' },
]

const PRETTY_FEATURE = {
  num_tasks: 'Task arrival rate',
  cpu_per_task: 'CPU per task',
  ram_per_task: 'RAM per task',
  hour_sin: 'Hour (sin)',
  hour_cos: 'Hour (cos)',
  day_of_week: 'Day of week',
  is_weekend: 'Weekend flag',
  cpu_lag_1: 'CPU t−1',
  cpu_lag_4: 'CPU t−4',
  cpu_rolling_mean_4: 'CPU mean (4)',
  cpu_rolling_std_8: 'CPU std (8)',
  ram_lag_1: 'RAM t−1',
}

function ModelComparison({ metrics }) {
  if (!metrics?.predictors) return null

  const rows = ALGOS.map(({ id, label }) => {
    const target = metrics.predictors[id]?.targets?.['cpu_demand_t+1']
    return {
      id, label,
      test: target?.test,
      naive: target?.naive_persistence_test,
    }
  }).filter((r) => r.test)

  if (!rows.length) return null
  const best = rows.reduce((a, b) => (b.test.r2 > a.test.r2 ? b : a))
  const naive = rows[0].naive

  return (
    <div className="panel">
      <h2 className="panel-title">
        Predictor comparison
        <span className="mono">held-out test block</span>
      </h2>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Model</th><th>R²</th><th>MAE</th><th>RMSE</th><th>MAPE</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.id} className={r.id === best.id ? 'highlight' : undefined}>
                <td>{r.label}</td>
                <td className="num">{r.test.r2.toFixed(4)}</td>
                <td className="num">{r.test.mae.toFixed(3)}</td>
                <td className="num">{r.test.rmse.toFixed(3)}</td>
                <td className="num">{r.test.mape.toFixed(2)}%</td>
              </tr>
            ))}
            {naive && (
              <tr style={{ color: 'var(--text-muted)' }}>
                <td>Persistence baseline</td>
                <td className="num">{naive.r2.toFixed(4)}</td>
                <td className="num">{naive.mae.toFixed(3)}</td>
                <td className="num">{naive.rmse.toFixed(3)}</td>
                <td className="num">{naive.mape.toFixed(2)}%</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
      <p className="panel-note">
        One-step-ahead forecast of CPU demand. Train/validation/test are contiguous
        blocks in time order (70/15/15); hyperparameters were selected on validation
        and the test block scored once. The persistence row — "next interval equals
        this one" — is the bar any model has to clear to be worth deploying.
      </p>
    </div>
  )
}

function Explanation({ explanation }) {
  if (!explanation) {
    return (
      <div className="panel">
        <h2 className="panel-title">Feature attribution</h2>
        <p className="panel-note">Run a prediction to see its attribution.</p>
      </div>
    )
  }

  const rows = (explanation.contributions ?? []).slice(0, 9).map((c) => ({
    label: PRETTY_FEATURE[c.feature] || c.feature,
    value: c.contribution,
    raw: c.value,
  }))

  const methodLabel = {
    'treeshap-exact': 'Exact TreeSHAP',
    'linear-shap-exact': 'Exact linear SHAP',
    'impurity-importance-approx': 'Impurity importance (approximation)',
  }[explanation.method] || explanation.method

  return (
    <div className="panel">
      <h2 className="panel-title">
        Feature attribution
        <span className="mono">{methodLabel}</span>
      </h2>
      <BarChart
        rows={rows} diverging
        valueFormat={(v) => `${v >= 0 ? '+' : ''}${v.toFixed(2)}`}
      />
      <p className="panel-note">
        Contribution of each feature to this prediction, in CPU cores, relative to a
        base value of {explanation.base_value?.toFixed(2)}. Blue pushes the forecast
        up, orange pushes it down; the bars sum to the prediction.
        {explanation.method === 'impurity-importance-approx' && (
          <> <strong>Note:</strong> Random Forest has no exact TreeSHAP here, so this
            shows impurity importance scaled to the prediction — it is directional,
            not additive. Switch to XGBoost for exact attribution.</>
        )}
      </p>
    </div>
  )
}

export default function Prediction({ session, guard, busy, meta }) {
  const [metrics, setMetrics] = useState(null)
  const [algo, setAlgo] = useState('xgboost')
  const [form, setForm] = useState({ num_tasks: 40, cpu_per_task: 0.42, ram_per_task: 0.9, hour: 14, day_of_week: 2 })
  const [result, setResult] = useState(null)
  const [historyRows, setHistoryRows] = useState([])
  const [loadError, setLoadError] = useState(null)

  useEffect(() => {
    api.modelMetrics().then(setMetrics).catch((e) => setLoadError(e.message))
    api.workloadHistory(288, 0)
      .then((d) => setHistoryRows(d.rows ?? []))
      .catch(() => {})
  }, [])

  const runPrediction = () =>
    guard(async () => {
      const r = await api.predict({ ...form, algo, explain: true })
      setResult(r)
      return null
    })

  const sessionExplain = () =>
    guard(async () => {
      const r = await api.explain(session.id)
      setResult({
        predicted_cpu: r.prediction.cpu,
        predicted_ram: r.prediction.ram,
        algo: session.status?.predictor_algo,
        explanation: r,
        fromSession: true,
      })
      return null
    })

  const chartData = historyRows.map((r, i) => ({
    i,
    cpu: r.cpu_demand,
    ram: r.ram_demand,
    tasks: r.num_tasks,
    ts: r.timestamp,
  }))

  return (
    <div className="grid" style={{ gap: 16 }}>
      {loadError && <div className="banner">{loadError}</div>}

      <div className="grid cols-3">
        <StatTile
          label="Best model R²"
          value={metrics?.predictors?.xgboost?.targets?.['cpu_demand_t+1']?.test?.r2?.toFixed(4) ?? '—'}
          sub="XGBoost, one-step-ahead CPU"
          tone="good"
        />
        <StatTile
          label="Training rows"
          value={metrics?.dataset?.rows?.toLocaleString() ?? '—'}
          sub={metrics?.dataset ? `${metrics.dataset.span_start?.slice(0, 10)} → ${metrics.dataset.span_end?.slice(0, 10)}` : ''}
        />
        <StatTile
          label="Anomaly recall"
          value={metrics?.anomaly?.isolation_forest?.recall
            ? (metrics.anomaly.isolation_forest.recall * 100).toFixed(1) : '—'}
          unit="%"
          sub={`Isolation Forest · ${metrics?.anomaly?.isolation_forest?.events_detected ?? 0}/${metrics?.anomaly?.isolation_forest?.events ?? 0} burst events`}
        />
      </div>

      <ModelComparison metrics={metrics} />

      <div className="grid sidebar">
        <div className="panel">
          <h2 className="panel-title">Training workload — first 24 h</h2>
          <LineChart
            data={chartData} xKey="i" height={250} areaFirst
            series={[
              { key: 'cpu', label: 'CPU demand (cores)', color: 'var(--series-1)' },
              { key: 'tasks', label: 'Task arrivals', color: 'var(--series-4)' },
            ]}
            xFormat={(v, row) => (row?.ts ? String(row.ts).slice(11, 16) : v)}
            valueFormat={(v) => Number(v).toFixed(0)}
          />
          <p className="panel-note">
            The nightly batch window (02:00–04:00) and the weekday afternoon peak are
            step changes rather than smooth curves — that non-smooth structure is what
            the tree ensembles capture and the linear baseline cannot.
          </p>
        </div>

        <div className="panel">
          <h2 className="panel-title">Ad-hoc prediction</h2>

          <div className="field">
            <label htmlFor="algo">Model</label>
            <select id="algo" value={algo} onChange={(e) => setAlgo(e.target.value)}>
              {ALGOS.map((a) => <option key={a.id} value={a.id}>{a.label}</option>)}
            </select>
          </div>

          {[
            { k: 'num_tasks', label: 'Task arrivals', min: 1, max: 150, step: 1 },
            { k: 'hour', label: 'Hour of day', min: 0, max: 23, step: 1 },
            { k: 'day_of_week', label: 'Day of week (0=Mon)', min: 0, max: 6, step: 1 },
          ].map((f) => (
            <div className="field" key={f.k}>
              <label htmlFor={f.k}>{f.label} — {form[f.k]}</label>
              <input
                id={f.k} type="range" min={f.min} max={f.max} step={f.step}
                value={form[f.k]}
                onChange={(e) => setForm({ ...form, [f.k]: Number(e.target.value) })}
              />
            </div>
          ))}

          <div className="btn-row">
            <button className="btn btn-primary" onClick={runPrediction} disabled={busy}>
              Predict
            </button>
            <button className="btn" onClick={sessionExplain} disabled={busy || !session}>
              Explain live session
            </button>
          </div>

          {result && (
            <div style={{ marginTop: 16 }}>
              <div className="stat-label">Forecast for next interval</div>
              <div className="stat-value" style={{ fontSize: '1.5rem' }}>
                {result.predicted_cpu.toFixed(2)} <span className="stat-unit">cores</span>
              </div>
              <div className="stat-sub">{result.predicted_ram.toFixed(2)} GB RAM · {result.algo}</div>
            </div>
          )}
        </div>
      </div>

      <Explanation explanation={result?.explanation} />
    </div>
  )
}
