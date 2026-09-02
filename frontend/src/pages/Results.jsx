import { useEffect, useState } from 'react'
import { api } from '../lib/api'
import { BarChart, LineChart, StatTile } from '../components/charts'

const delta = (v) => (v === null || v === undefined ? '—'
  : `${v > 0 ? '+' : ''}${v.toFixed(1)}%`)

const deltaColor = (v, goodIsUp) => {
  if (v === null || v === undefined || Math.abs(v) < 0.05) return 'var(--text-muted)'
  const good = goodIsUp ? v > 0 : v < 0
  return good ? '#7ee787' : '#ff9a9a'
}

export default function Results({ session }) {
  const [ablation, setAblation] = useState(null)
  const [rl, setRl] = useState(null)
  const [training, setTraining] = useState(null)
  const [status, setStatus] = useState('loading')
  const [error, setError] = useState(null)

  useEffect(() => {
    api.ablationCached()
      .then((d) => { setAblation(d); setStatus('ready') })
      .catch(() => {
        // No cached study on the server — run one live.
        setStatus('running')
        api.ablation({ ticks: 288, seed: 42 })
          .then((d) => { setAblation(d); setStatus('ready') })
          .catch((e) => { setError(e.message); setStatus('error') })
      })
    api.modelMetrics().then(setTraining).catch(() => {})
  }, [])

  useEffect(() => {
    if (session?.id) api.rlState(session.id).then(setRl).catch(() => {})
  }, [session?.id])

  const rows = ablation?.rows ?? []
  const full = rows.find((r) => r.strategy === 'full')
  const base = rows.find((r) => r.strategy === 'ml_predictive')

  // The curve plotted is the greedy evaluation on one fixed held-out seed, not
  // the per-episode training reward. Training episodes each use a different
  // seed, so their reward moves with trace difficulty rather than policy
  // quality and is not a learning signal.
  const curveData = (training?.rl?.eval_curve ?? []).map((p) => ({
    ep: p.episode,
    reward: p.reward,
    utilisation: p.utilisation,
    cost: p.cost_per_day,
  }))

  return (
    <div className="grid" style={{ gap: 16 }}>
      {status === 'running' && (
        <div className="banner">
          <span className="spinner" /> No cached study on the server — running all seven
          configurations live. This takes a minute or two.
        </div>
      )}
      {error && <div className="banner banner-error">{error}</div>}

      {full && base && (
        <div className="grid cols-4">
          <StatTile
            label="Utilisation" value={full.utilisation.toFixed(1)} unit="%"
            tone="good" sub={`${delta(full.delta_utilisation_pct)} vs ML-only baseline`}
          />
          <StatTile
            label="Cost" value={`$${full.cost_per_day.toFixed(2)}`} unit="/day"
            tone={full.delta_cost_per_day_pct < 0 ? 'good' : undefined}
            sub={`${delta(full.delta_cost_per_day_pct)} vs baseline`}
          />
          <StatTile
            label="Response latency" value={full.response_latency_s.toFixed(0)} unit="s"
            tone="good" sub={`${delta(full.delta_response_latency_s_pct)} vs baseline`}
          />
          <StatTile
            label="Task failure rate" value={full.task_failure_rate.toFixed(2)} unit="%"
            tone={full.task_failure_rate < 1 ? 'good' : 'warning'}
            sub={`SLA compliance ${full.sla_compliance.toFixed(1)}%`}
          />
        </div>
      )}

      <div className="panel">
        <h2 className="panel-title">
          Ablation study — measured
          {ablation && (
            <span className="mono">
              {ablation.protocol.repeats} seeds × {ablation.protocol.ticks} ticks
              ({ablation.protocol.simulated_hours} h)
            </span>
          )}
        </h2>

        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Configuration</th>
                <th>Utilisation</th>
                <th>Cost $/day</th>
                <th>Latency</th>
                <th>Fail %</th>
                <th>SLA %</th>
                <th>CO₂ kg</th>
                <th>Nodes</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.strategy} className={r.strategy === 'full' ? 'highlight' : undefined}>
                  <td>{r.label}</td>
                  <td className="num">
                    {r.utilisation.toFixed(1)}
                    <span className="muted" style={{ fontSize: '0.7rem' }}> ±{r.utilisation_sd.toFixed(1)}</span>
                  </td>
                  <td className="num">
                    {r.cost_per_day.toFixed(2)}
                    <span className="muted" style={{ fontSize: '0.7rem' }}> ±{r.cost_per_day_sd.toFixed(2)}</span>
                  </td>
                  <td className="num">{r.response_latency_s.toFixed(0)}s</td>
                  <td className="num">{r.task_failure_rate.toFixed(2)}</td>
                  <td className="num">{r.sla_compliance.toFixed(1)}</td>
                  <td className="num">{r.co2_kg.toFixed(2)}</td>
                  <td className="num">{r.mean_fleet_size.toFixed(1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {ablation && (
          <p className="panel-note">
            {ablation.protocol.note} Baseline: <strong>{ablation.protocol.baseline}</strong>.
            Generated {ablation.generated_at}.
          </p>
        )}
      </div>

      {rows.length > 0 && (
        <div className="grid cols-2">
          <div className="panel">
            <h2 className="panel-title">Cost per day by configuration</h2>
            <BarChart
              rows={rows.map((r) => ({
                label: r.label.replace(/ \(.*\)/, ''),
                value: r.cost_per_day,
                best: r.strategy === 'full',
              }))}
              highlightKey="best"
              valueFormat={(v) => `$${v.toFixed(2)}`}
            />
          </div>

          <div className="panel">
            <h2 className="panel-title">Scaling response latency</h2>
            <BarChart
              rows={rows.map((r) => ({
                label: r.label.replace(/ \(.*\)/, ''),
                value: r.response_latency_s,
                best: r.strategy === 'full',
              }))}
              highlightKey="best"
              color="var(--series-2)"
              valueFormat={(v) => `${v.toFixed(0)}s`}
            />
            <p className="panel-note">
              Mean duration of an under-provisioned episode, plus the modelled 45 s
              instance boot time. Lower is better.
            </p>
          </div>
        </div>
      )}

      <div className="grid cols-2">
        <div className="panel">
          <h2 className="panel-title">DQN learning curve</h2>
          {curveData.length ? (
            <>
              <LineChart
                data={curveData} xKey="ep" height={230}
                series={[{ key: 'reward', label: 'Mean episode reward', color: 'var(--series-1)' }]}
                xFormat={(v) => `ep ${v}`}
                valueFormat={(v) => Number(v).toFixed(2)}
              />
              <p className="panel-note">
                Mean reward per training episode. First quarter{' '}
                <span className="mono">{training.rl.first_quarter_mean?.toFixed(3)}</span>,
                last quarter <span className="mono">{training.rl.last_quarter_mean?.toFixed(3)}</span>
                {' '}(improvement {training.rl.improvement?.toFixed(3)}) over{' '}
                {training.rl.learn_steps?.toLocaleString()} gradient steps.
              </p>
            </>
          ) : <p className="panel-note">No RL training report available.</p>}
        </div>

        <div className="panel">
          <h2 className="panel-title">Live agent state</h2>
          {rl ? (
            <>
              <div className="grid cols-2" style={{ gap: 10 }}>
                <div><div className="stat-label">Epsilon</div><div className="mono">{rl.epsilon.toFixed(4)}</div></div>
                <div><div className="stat-label">Learn steps</div><div className="mono">{rl.learn_steps.toLocaleString()}</div></div>
                <div><div className="stat-label">Replay buffer</div><div className="mono">{rl.replay_size.toLocaleString()}</div></div>
                <div><div className="stat-label">Env steps</div><div className="mono">{rl.env_steps.toLocaleString()}</div></div>
              </div>
              {rl.reward_curve?.length > 1 && (
                <div style={{ marginTop: 14 }}>
                  <LineChart
                    data={rl.reward_curve.map((v, i) => ({ i, reward: v }))}
                    xKey="i" height={160}
                    series={[{ key: 'reward', label: 'Reward this session', color: 'var(--series-3)' }]}
                    xFormat={(v) => `t${v}`}
                    valueFormat={(v) => Number(v).toFixed(2)}
                  />
                </div>
              )}
              <p className="panel-note">
                Actions are headroom setpoints: {rl.actions?.join(', ')}.
              </p>
            </>
          ) : <p className="panel-note">Step the simulation to populate agent state.</p>}
        </div>
      </div>

      {training?.anomaly && (
        <div className="panel">
          <h2 className="panel-title">Anomaly detection — event-based scoring</h2>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Detector</th><th>Events</th><th>Detected</th>
                  <th>Precision</th><th>Recall</th><th>F1</th><th>Alarm rate</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(training.anomaly).map(([name, a]) => (
                  <tr key={name} className={name === 'isolation_forest' ? 'highlight' : undefined}>
                    <td>{name === 'isolation_forest' ? 'Isolation Forest' : 'Z-score'}</td>
                    <td className="num">{a.events ?? '—'}</td>
                    <td className="num">{a.events_detected ?? '—'}</td>
                    <td className="num">{a.precision?.toFixed(3) ?? '—'}</td>
                    <td className="num">{a.recall?.toFixed(3) ?? '—'}</td>
                    <td className="num">{a.f1?.toFixed(3) ?? '—'}</td>
                    <td className="num">{a.alarm_rate ? `${(a.alarm_rate * 100).toFixed(2)}%` : '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="panel-note">
            Scored per burst <em>event</em> with a tolerance window, not per interval:
            a burst spans several intervals but only its onset is detectable, so
            point-wise scoring would penalise a correct detector for the decay tail.
            Both detectors are set to the same recall so precision is comparable.
          </p>
        </div>
      )}
    </div>
  )
}
