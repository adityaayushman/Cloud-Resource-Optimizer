import { useState } from 'react'
import { api } from '../lib/api'
import { LineChart, ShareBar, StatTile, seriesColor } from '../components/charts'

const SEVERITY_CLASS = {
  critical: 'alert-critical',
  warning: 'alert-warning',
  optimisation: 'alert-optimisation',
}

const utilTone = (u) => (u >= 92 ? 'critical' : u >= 55 ? 'good' : 'warning')

function FleetPanel({ session, guard, busy }) {
  const [selected, setSelected] = useState(null)
  const fleet = session?.fleet ?? []

  const act = async (kind) => {
    if (!selected) return
    if (kind === 'fault') await guard(() => api.fault(session.id, selected))
    else await guard(() => api.decommission(session.id, selected))
    setSelected(null)
  }

  return (
    <div className="panel">
      <h2 className="panel-title">
        Infrastructure topology
        <span className="mono">{fleet.length} nodes</span>
      </h2>

      {fleet.length === 0 && <p className="panel-note">Fleet is empty.</p>}

      <div className="fleet">
        {fleet.map((vm) => {
          const u = vm.cpu_utilization
          const tone = u > 92 ? 'var(--critical)' : u > 75 ? 'var(--warning)' : 'var(--good)'
          return (
            <button
              key={vm.id}
              className="node"
              onClick={() => setSelected(selected === vm.id ? null : vm.id)}
              style={{
                borderBottomColor: tone,
                borderColor: selected === vm.id ? 'var(--accent)' : undefined,
                background: 'none',
                color: 'inherit',
                font: 'inherit',
              }}
              title={`${vm.id} · ${vm.type} · ${vm.provider} · ${vm.region}
CPU ${vm.cpu_usage}/${vm.cpu_capacity} · RAM ${vm.ram_usage}/${vm.ram_capacity}
$${vm.cost_per_hour}/hr · ${vm.power_watts} W`}
            >
              <div className="node-provider">{vm.provider}</div>
              <div className="node-id">{vm.id.replace('vm-', 'NODE-')}</div>
              <div className="node-util" style={{ color: tone }}>{u.toFixed(0)}%</div>
              <div className="node-type">{vm.type} · {vm.task_count} tasks</div>
            </button>
          )
        })}
      </div>

      <div className="btn-row" style={{ marginTop: 14 }}>
        <button className="btn" disabled={!selected || busy} onClick={() => act('decommission')}>
          Decommission node
        </button>
        <button className="btn btn-danger" disabled={!selected || busy} onClick={() => act('fault')}>
          Simulate failure
        </button>
        <span className="panel-note" style={{ margin: 0 }}>
          {selected ? `Selected ${selected}` : 'Select a node to act on it'}
        </span>
      </div>
    </div>
  )
}

function AdvisoryPanel({ advisory }) {
  const warnings = advisory?.warnings ?? []
  const recs = advisory?.recommendations ?? []

  return (
    <div className="panel">
      <h2 className="panel-title">
        Predictive advisory
        {advisory?.potential_hourly_saving > 0 && (
          <span className="mono" style={{ color: '#7ee787' }}>
            ${advisory.potential_hourly_saving.toFixed(2)}/hr recoverable
          </span>
        )}
      </h2>

      {warnings.length === 0 && recs.length === 0 && (
        <p className="panel-note">
          No advisories. Capacity is tracking forecast demand within tolerance.
        </p>
      )}

      {warnings.map((w, i) => (
        <div className={`alert ${SEVERITY_CLASS[w.severity] || ''}`} key={`w${i}`}>
          <div className="alert-head" style={{
            color: w.severity === 'critical' ? '#ff9a9a'
              : w.severity === 'warning' ? '#f0c674' : 'var(--accent)',
          }}>
            {w.severity === 'critical' ? '●' : w.severity === 'warning' ? '▲' : '◆'}
            {w.type.replace(/_/g, ' ')} · {w.severity}
          </div>
          <div className="alert-msg">{w.message}</div>
          <div className="alert-meta">Impact window: {w.eta}</div>
        </div>
      ))}

      {recs.map((r, i) => (
        <div className="alert alert-optimisation" key={`r${i}`}>
          <div className="alert-head" style={{ color: 'var(--accent)' }}>
            ▶ recommended action · urgency {r.urgency}
          </div>
          <div className="alert-msg">{r.action}</div>
          <div className="alert-meta">{r.benefit}</div>
        </div>
      ))}
    </div>
  )
}

function Console({ logs }) {
  return (
    <div className="panel">
      <h2 className="panel-title">Engine output</h2>
      <div className="console">
        {(logs ?? []).slice().reverse().map((l, i) => (
          <div className="console-line" key={i}>
            <span className="console-src">[{l.source}]</span>
            <span>{l.message}</span>
          </div>
        ))}
        {(!logs || logs.length === 0) && <div className="console-line">[IDLE] awaiting activity…</div>}
      </div>
    </div>
  )
}

function Controls({ session, guard, busy, step }) {
  const [tasks, setTasks] = useState(25)

  return (
    <div className="panel">
      <h2 className="panel-title">Controls</h2>

      <div className="field">
        <label htmlFor="scenario">Guided scenario</label>
        <select
          id="scenario" defaultValue=""
          onChange={(e) => {
            if (!e.target.value) return
            guard(() => api.scenario(session.id, e.target.value))
            e.target.value = ''
          }}
          disabled={busy}
        >
          <option value="">Select a scenario…</option>
          <option value="over_provisioned">Over-provisioned (cost leak)</option>
          <option value="under_provisioned">Under-provisioned (demand surge)</option>
          <option value="balanced">Balanced fleet</option>
          <option value="reset">Reset to clean fleet</option>
        </select>
      </div>

      <div className="field">
        <label htmlFor="inject">Inject load — {tasks} tasks</label>
        <input
          id="inject" type="range" min="1" max="120" value={tasks}
          onChange={(e) => setTasks(Number(e.target.value))}
        />
      </div>

      <div className="btn-row" style={{ marginBottom: 14 }}>
        <button
          className="btn" disabled={busy}
          onClick={() => guard(() => api.inject(session.id, {
            task_count: tasks, cpu_per_task: 0.4, ram_per_task: 1.0, duration: 300,
          }))}
        >
          Inject pulse
        </button>
        <button
          className="btn" disabled={busy}
          onClick={() => guard(() => api.scale(session.id, { instance_type: 'medium', count: 1 }))}
        >
          + Medium node
        </button>
      </div>

      <div className="btn-row">
        <button className="btn" disabled={busy} onClick={() => step(6)}>Step ×6 (30 min)</button>
        <button className="btn" disabled={busy} onClick={() => step(24)}>Step ×24 (2 h)</button>
      </div>

      <p className="panel-note">
        One tick is {session?.config?.tick_seconds ?? 300}s of simulated time. The RL
        agent chooses a headroom setpoint each tick and the fleet is resized to match.
      </p>
    </div>
  )
}

export default function CommandCenter({ session, guard, busy, step }) {
  if (!session) return <p className="muted">No active session.</p>

  const m = session.metrics
  const history = session.history ?? []
  const providers = Object.entries(m.by_provider ?? {}).map(([label, value], i) => ({
    label, value, color: seriesColor(i),
  }))

  const chartData = history.map((h) => ({
    tick: h.tick,
    demand: h.demand_cpu,
    predicted: h.predicted_cpu,
    capacity: h.capacity_cpu,
  }))

  return (
    <div className="grid" style={{ gap: 16 }}>
      <div className="grid cols-4">
        <StatTile
          label="CPU utilisation" value={m.cpu_utilization.toFixed(1)} unit="%"
          tone={utilTone(m.cpu_utilization)}
          sub={`${m.cpu_used} of ${m.cpu_capacity} cores`}
        />
        <StatTile
          label="Operating cost" value={`$${m.daily_cost.toFixed(2)}`} unit="/day"
          sub={`$${m.hourly_cost.toFixed(3)}/hr · ${m.fleet_size} nodes`}
        />
        <StatTile
          label="SLA compliance" value={m.sla_compliance.toFixed(1)} unit="%"
          tone={m.sla_compliance >= 95 ? 'good' : m.sla_compliance >= 80 ? 'warning' : 'critical'}
          sub={`${m.hot_nodes ?? 0} node(s) saturated · ${m.task_failure_rate.toFixed(2)}% work rejected`}
        />
        <StatTile
          label="Carbon" value={m.co2_kg_per_hour.toFixed(3)} unit="kg/hr"
          sub={`${m.power_watts.toFixed(0)} W drawn`}
        />
      </div>

      <div className="grid sidebar">
        <div className="panel">
          <h2 className="panel-title">
            Demand, forecast and provisioned capacity
            <span className="mono">tick {session.tick_index}</span>
          </h2>
          <LineChart
            data={chartData}
            xKey="tick"
            height={280}
            areaFirst
            series={[
              { key: 'demand', label: 'Actual demand', color: 'var(--series-1)' },
              { key: 'predicted', label: 'Forecast (t+1)', color: 'var(--series-2)', dashed: true },
              { key: 'capacity', label: 'Provisioned capacity', color: 'var(--series-3)' },
            ]}
            xFormat={(v) => `t${v}`}
            valueFormat={(v) => `${Number(v).toFixed(0)}`}
          />
          <p className="panel-note">
            CPU cores. Capacity above demand is headroom; capacity below it means work is
            being rejected. The forecast line is the model's next-interval prediction —
            the signal the agent acts on.
          </p>
        </div>

        <Controls session={session} guard={guard} busy={busy} step={step} />
      </div>

      <div className="grid sidebar">
        <FleetPanel session={session} guard={guard} busy={busy} />
        <AdvisoryPanel advisory={session.advisory} />
      </div>

      <div className="grid sidebar">
        <Console logs={session.logs} />
        <div className="panel">
          <h2 className="panel-title">Fleet distribution</h2>
          <div style={{ marginBottom: 18 }}>
            <div className="stat-label" style={{ marginBottom: 8 }}>By provider</div>
            {providers.length
              ? <ShareBar parts={providers} />
              : <p className="panel-note">No nodes.</p>}
          </div>
          <div>
            <div className="stat-label" style={{ marginBottom: 8 }}>By instance type</div>
            <ShareBar
              parts={Object.entries(session.metrics.by_instance_type ?? {})
                .map(([label, value], i) => ({ label, value, color: seriesColor(i) }))}
            />
          </div>
          <p className="panel-note">
            Task failure rate {m.task_failure_rate.toFixed(2)}% ·
            {' '}{m.tasks_completed} completed · mean latency {m.mean_latency_ms} ms
          </p>
        </div>
      </div>
    </div>
  )
}
