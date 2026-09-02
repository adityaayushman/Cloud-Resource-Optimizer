import { useCallback, useEffect, useState } from 'react'
import { api } from '../lib/api'
import { BarChart, LineChart, StatTile } from '../components/charts'

const PROVIDER_COLOR = {
  AWS: 'var(--series-1)',
  Azure: 'var(--series-2)',
  GCP: 'var(--series-3)',
}

export default function MultiCloud() {
  const [weights, setWeights] = useState({ cost: 55, latency: 25, carbon: 20 })
  const [instance, setInstance] = useState('medium')
  const [region, setRegion] = useState('US-East')
  const [board, setBoard] = useState(null)
  const [series, setSeries] = useState([])
  const [error, setError] = useState(null)

  const load = useCallback(async () => {
    setError(null)
    try {
      const total = weights.cost + weights.latency + weights.carbon || 1
      const res = await api.scoreProviders({
        instance_type: instance,
        region,
        weight_cost: weights.cost / total,
        weight_latency: weights.latency / total,
        weight_carbon: weights.carbon / total,
      })
      setBoard(res)
    } catch (e) { setError(e.message) }
  }, [weights, instance, region])

  useEffect(() => { load() }, [load])

  useEffect(() => {
    api.priceSeries(24, 96)
      .then((d) => setSeries(d.series ?? []))
      .catch(() => {})
  }, [])

  const rows = board?.scoreboard ?? []

  return (
    <div className="grid" style={{ gap: 16 }}>
      <div className="grid cols-3">
        <StatTile
          label="Recommended provider"
          value={board?.recommended ?? '—'}
          tone="good"
          sub={`For ${instance} instances in ${region}`}
        />
        <StatTile
          label="Spread, best vs worst"
          value={board ? `${board.saving_vs_worst_pct.toFixed(1)}` : '—'}
          unit="%"
          sub="Hourly on-demand price difference"
        />
        <StatTile
          label="Cheapest hourly rate"
          value={rows.length ? `$${Math.min(...rows.map((r) => r.hourly_cost)).toFixed(4)}` : '—'}
          sub="Per instance, current quote"
        />
      </div>

      {error && <div className="banner banner-error">{error}</div>}

      <div className="grid sidebar">
        <div className="panel">
          <h2 className="panel-title">
            Provider scoreboard
            <span className="mono">lower score is better</span>
          </h2>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Provider</th><th>Score</th><th>$/hr</th>
                  <th>Price idx</th><th>Latency</th><th>Carbon</th><th>Reliability</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => (
                  <tr key={r.provider} className={r.rank === 1 ? 'highlight' : undefined}>
                    <td>
                      <span style={{
                        display: 'inline-block', width: 9, height: 9, borderRadius: 2,
                        background: PROVIDER_COLOR[r.provider], marginRight: 8,
                      }} />
                      {r.provider}{r.rank === 1 ? ' ★' : ''}
                    </td>
                    <td className="num">{r.score.toFixed(3)}</td>
                    <td className="num">${r.hourly_cost.toFixed(4)}</td>
                    <td className="num">{r.price_index.toFixed(3)}</td>
                    <td className="num">{r.latency_ms} ms</td>
                    <td className="num">{r.carbon_kg_per_kwh}</td>
                    <td className="num">{(r.reliability * 100).toFixed(2)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="panel-note">
            Cost, latency and carbon are min-max normalised across the candidate set
            before being combined, so three incommensurable units can share one score.
            Prices are deterministic for a given 5-minute bucket — reloading this page
            does not reshuffle the ranking.
          </p>
        </div>

        <div className="panel">
          <h2 className="panel-title">Selection weights</h2>
          {[
            { k: 'cost', label: 'Cost' },
            { k: 'latency', label: 'Latency' },
            { k: 'carbon', label: 'Carbon intensity' },
          ].map((w) => (
            <div className="field" key={w.k}>
              <label htmlFor={w.k}>{w.label} — {weights[w.k]}</label>
              <input
                id={w.k} type="range" min="0" max="100"
                value={weights[w.k]}
                onChange={(e) => setWeights({ ...weights, [w.k]: Number(e.target.value) })}
              />
            </div>
          ))}

          <div className="field">
            <label htmlFor="inst">Instance type</label>
            <select id="inst" value={instance} onChange={(e) => setInstance(e.target.value)}>
              <option value="small">Small — 2 vCPU / 4 GB</option>
              <option value="medium">Medium — 4 vCPU / 8 GB</option>
              <option value="large">Large — 8 vCPU / 16 GB</option>
              <option value="memory">Memory — 4 vCPU / 16 GB</option>
            </select>
          </div>

          <div className="field">
            <label htmlFor="reg">Region</label>
            <select id="reg" value={region} onChange={(e) => setRegion(e.target.value)}>
              <option value="US-East">US-East (0.40 kg/kWh)</option>
              <option value="EU-West">EU-West (0.30 kg/kWh)</option>
              <option value="Asia-South">Asia-South (0.70 kg/kWh)</option>
            </select>
          </div>

          <p className="panel-note">
            Push carbon to maximum and the ranking shifts toward the cleaner region's
            cheapest reliable provider — the trade-off the report calls multi-cloud
            arbitrage.
          </p>
        </div>
      </div>

      <div className="panel">
        <h2 className="panel-title">Price index — next 24 hours</h2>
        <LineChart
          data={series} xKey="offset_hours" height={250}
          series={[
            { key: 'AWS', label: 'AWS', color: PROVIDER_COLOR.AWS },
            { key: 'Azure', label: 'Azure', color: PROVIDER_COLOR.Azure },
            { key: 'GCP', label: 'GCP', color: PROVIDER_COLOR.GCP },
          ]}
          xFormat={(v) => `+${Number(v).toFixed(0)}h`}
          valueFormat={(v) => Number(v).toFixed(2)}
        />
        <p className="panel-note">
          Relative to AWS on-demand = 1.00. AWS carries the widest spot band (±18%),
          GCP the narrowest (±6%), which is what creates the arbitrage window the
          allocator exploits when placing new nodes.
        </p>
      </div>

      <div className="panel">
        <h2 className="panel-title">Hourly cost by provider</h2>
        <BarChart
          rows={rows.map((r) => ({ label: r.provider, value: r.hourly_cost, best: r.rank === 1 }))}
          highlightKey="best"
          valueFormat={(v) => `$${v.toFixed(4)}`}
        />
      </div>
    </div>
  )
}
