/* Chart primitives, built as inline SVG.
 *
 * Written by hand rather than pulled from a chart library so the mark specs are
 * exact: 2px strokes, >=8px hit targets, a 2px surface gap between adjacent
 * fills, recessive grid/axes, a crosshair tooltip on every time series, and a
 * legend whenever more than one series is present so identity never rests on
 * colour alone. Series colours come from the validated --series-* tokens.
 */

import { useEffect, useId, useMemo, useRef, useState } from 'react'

const SERIES_TOKENS = ['var(--series-1)', 'var(--series-2)', 'var(--series-3)',
  'var(--series-4)', 'var(--series-5)']

export const seriesColor = (i) => SERIES_TOKENS[i % SERIES_TOKENS.length]

const fmt = (v, digits = 1) =>
  v === null || v === undefined || Number.isNaN(v) ? '—' : Number(v).toFixed(digits)

function niceTicks(min, max, count = 5) {
  if (!Number.isFinite(min) || !Number.isFinite(max) || min === max) {
    return [min || 0, (min || 0) + 1]
  }
  const span = max - min
  const raw = span / count
  const mag = 10 ** Math.floor(Math.log10(raw))
  const norm = raw / mag
  const step = (norm >= 7.5 ? 10 : norm >= 3.5 ? 5 : norm >= 1.5 ? 2 : 1) * mag
  const start = Math.floor(min / step) * step
  const out = []
  for (let v = start; v <= max + step * 0.5; v += step) out.push(Number(v.toFixed(10)))
  return out
}

/* ------------------------------------------------------------------ legend */

export function Legend({ series }) {
  if (!series || series.length < 2) return null
  return (
    <div className="legend">
      {series.map((s) => (
        <span className="legend-item" key={s.key}>
          <span
            className={`legend-swatch${s.dashed ? ' dashed' : ''}`}
            style={s.dashed ? { color: s.color } : { background: s.color }}
          />
          {s.label}
        </span>
      ))}
    </div>
  )
}

/* -------------------------------------------------------------- line chart */

export function LineChart({
  data,
  series,
  xKey = 'x',
  height = 260,
  yLabel = '',
  xFormat = (v) => v,
  valueFormat = (v) => fmt(v, 1),
  areaFirst = false,
}) {
  const clipId = useId().replace(/:/g, '')
  const wrapRef = useRef(null)
  const [hover, setHover] = useState(null)
  const [width, setWidth] = useState(720)

  // useEffect rather than a callback ref: React 18 callback refs cannot return
  // a cleanup function, so observing there leaks a ResizeObserver per remount.
  useEffect(() => {
    const node = wrapRef.current
    if (!node) return undefined
    const update = () => setWidth(Math.max(320, node.clientWidth))
    update()
    const ro = new ResizeObserver(update)
    ro.observe(node)
    return () => ro.disconnect()
  }, [])

  const pad = { top: 12, right: 14, bottom: 26, left: 46 }
  const innerW = Math.max(10, width - pad.left - pad.right)
  const innerH = Math.max(10, height - pad.top - pad.bottom)

  const { yMin, yMax, ticks } = useMemo(() => {
    let lo = Infinity
    let hi = -Infinity
    data.forEach((row) => {
      series.forEach((s) => {
        const v = row[s.key]
        if (typeof v === 'number' && Number.isFinite(v)) {
          if (v < lo) lo = v
          if (v > hi) hi = v
        }
      })
    })
    if (!Number.isFinite(lo)) { lo = 0; hi = 1 }
    const padSpan = (hi - lo) * 0.1 || 1
    const min = Math.max(0, lo - padSpan)
    const max = hi + padSpan
    return { yMin: min, yMax: max, ticks: niceTicks(min, max, 4) }
  }, [data, series])

  if (!data.length) {
    return <p className="panel-note">No data yet — step the simulation to populate this chart.</p>
  }

  const sx = (i) => (data.length === 1 ? innerW / 2 : (i / (data.length - 1)) * innerW)
  const sy = (v) => innerH - ((v - yMin) / (yMax - yMin || 1)) * innerH

  const pathFor = (key) => {
    let d = ''
    let open = false
    data.forEach((row, i) => {
      const v = row[key]
      if (typeof v !== 'number' || !Number.isFinite(v)) { open = false; return }
      d += `${open ? 'L' : 'M'}${sx(i).toFixed(2)},${sy(v).toFixed(2)} `
      open = true
    })
    return d.trim()
  }

  const areaFor = (key) => {
    const line = pathFor(key)
    if (!line) return ''
    return `${line} L${sx(data.length - 1).toFixed(2)},${innerH} L${sx(0).toFixed(2)},${innerH} Z`
  }

  const onMove = (e) => {
    const rect = e.currentTarget.getBoundingClientRect()
    const x = e.clientX - rect.left - pad.left
    const idx = Math.round((x / innerW) * (data.length - 1))
    if (idx >= 0 && idx < data.length) setHover(idx)
  }

  const tipLeft = hover !== null
    ? Math.min(Math.max(pad.left + sx(hover) - 84, 4), Math.max(4, width - 176))
    : 0

  return (
    <div ref={wrapRef} style={{ position: 'relative' }}>
      <Legend series={series} />
      <svg
        width="100%" height={height} viewBox={`0 0 ${width} ${height}`}
        role="img" aria-label={yLabel || 'time series chart'}
        onMouseMove={onMove} onMouseLeave={() => setHover(null)}
        style={{ display: 'block', touchAction: 'none' }}
      >
        <defs>
          <clipPath id={`clip-${clipId}`}>
            <rect x="0" y="0" width={innerW} height={innerH} />
          </clipPath>
        </defs>
        <g transform={`translate(${pad.left},${pad.top})`}>
          {ticks.map((t) => (
            <g key={t}>
              <line x1="0" x2={innerW} y1={sy(t)} y2={sy(t)} stroke="var(--grid)" strokeWidth="1" />
              <text
                x="-8" y={sy(t)} dy="0.32em" textAnchor="end"
                fill="var(--text-muted)" fontSize="10" fontFamily="var(--mono)"
              >
                {valueFormat(t)}
              </text>
            </g>
          ))}
          <line x1="0" x2={innerW} y1={innerH} y2={innerH} stroke="var(--axis)" strokeWidth="1" />

          <g clipPath={`url(#clip-${clipId})`}>
            {areaFirst && series[0] && (
              <path d={areaFor(series[0].key)} fill={series[0].color} opacity="0.13" />
            )}
            {series.map((s) => (
              <path
                key={s.key} d={pathFor(s.key)} fill="none" stroke={s.color}
                strokeWidth="2" strokeLinejoin="round" strokeLinecap="round"
                strokeDasharray={s.dashed ? '5 4' : undefined}
              />
            ))}
          </g>

          {hover !== null && (
            <g>
              <line
                x1={sx(hover)} x2={sx(hover)} y1="0" y2={innerH}
                stroke="var(--accent)" strokeWidth="1" opacity="0.55"
              />
              {series.map((s) => {
                const v = data[hover][s.key]
                if (typeof v !== 'number' || !Number.isFinite(v)) return null
                return (
                  <circle
                    key={s.key} cx={sx(hover)} cy={sy(v)} r="4"
                    fill={s.color} stroke="var(--surface-1)" strokeWidth="2"
                  />
                )
              })}
            </g>
          )}

          {data.length > 1 && [0, Math.floor((data.length - 1) / 2), data.length - 1].map((i) => (
            <text
              key={i} x={sx(i)} y={innerH + 16} textAnchor={i === 0 ? 'start' : i === data.length - 1 ? 'end' : 'middle'}
              fill="var(--text-muted)" fontSize="10" fontFamily="var(--mono)"
            >
              {xFormat(data[i][xKey], data[i])}
            </text>
          ))}
        </g>
      </svg>

      {hover !== null && (
        <div
          style={{
            position: 'absolute', left: tipLeft, top: 4, pointerEvents: 'none',
            background: 'var(--surface-1)', border: '1px solid var(--border-strong)',
            borderRadius: 4, padding: '7px 9px', fontSize: '0.74rem', minWidth: 150, zIndex: 5,
          }}
        >
          <div className="mono muted" style={{ marginBottom: 4 }}>
            {xFormat(data[hover][xKey], data[hover])}
          </div>
          {series.map((s) => (
            <div key={s.key} style={{ display: 'flex', justifyContent: 'space-between', gap: 12 }}>
              <span style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
                <span style={{ width: 9, height: 3, borderRadius: 2, background: s.color }} />
                {s.label}
              </span>
              <span className="mono">{valueFormat(data[hover][s.key])}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

/* ------------------------------------------------------- horizontal bars */

export function BarChart({
  rows, valueKey = 'value', labelKey = 'label',
  valueFormat = (v) => fmt(v, 1), color = 'var(--series-1)',
  highlightKey = null, height = 26, diverging = false,
}) {
  const [hover, setHover] = useState(null)
  if (!rows?.length) return <p className="panel-note">No data.</p>

  const values = rows.map((r) => r[valueKey] ?? 0)
  const maxAbs = Math.max(...values.map(Math.abs), 1e-9)
  const max = Math.max(...values, 0)
  const min = Math.min(...values, 0)

  return (
    <div>
      {rows.map((row, i) => {
        const v = row[valueKey] ?? 0
        const isHi = highlightKey && row[highlightKey]
        const barColor = diverging
          ? (v >= 0 ? 'var(--series-1)' : 'var(--series-2)')
          : (isHi ? 'var(--accent)' : color)

        // Diverging bars grow from a centre baseline; ordinary bars from 0.
        const pct = diverging
          ? (Math.abs(v) / maxAbs) * 50
          : ((v - Math.min(0, min)) / (max - Math.min(0, min) || 1)) * 100

        return (
          <div
            key={row[labelKey] + i}
            onMouseEnter={() => setHover(i)}
            onMouseLeave={() => setHover(null)}
            style={{
              display: 'grid', gridTemplateColumns: 'minmax(96px, 34%) 1fr auto',
              alignItems: 'center', gap: 10, marginBottom: 7,
              background: hover === i ? 'var(--surface-2)' : 'transparent',
              borderRadius: 3, padding: '2px 3px',
            }}
          >
            <span style={{ fontSize: '0.78rem', color: isHi ? 'var(--text-primary)' : 'var(--text-secondary)', fontWeight: isHi ? 700 : 500 }}>
              {row[labelKey]}
            </span>
            <div style={{ position: 'relative', height, background: 'var(--surface-1)', borderRadius: 3 }}>
              {diverging && (
                <div style={{ position: 'absolute', left: '50%', top: 0, bottom: 0, width: 1, background: 'var(--axis)' }} />
              )}
              <div
                style={{
                  position: 'absolute', top: 3, bottom: 3,
                  left: diverging ? (v >= 0 ? '50%' : `${50 - pct}%`) : 0,
                  width: `${Math.max(pct, 0.6)}%`,
                  background: barColor,
                  borderRadius: diverging
                    ? (v >= 0 ? '0 4px 4px 0' : '4px 0 0 4px')
                    : '0 4px 4px 0',
                }}
              />
            </div>
            <span className="mono" style={{ fontSize: '0.78rem', minWidth: 68, textAlign: 'right' }}>
              {valueFormat(v)}
            </span>
          </div>
        )
      })}
    </div>
  )
}

/* ------------------------------------------------------------ stat tile */

export function StatTile({ label, value, unit, sub, tone }) {
  const toneColor = tone === 'good' ? '#7ee787'
    : tone === 'warning' ? '#f0c674'
      : tone === 'critical' ? '#ff9a9a' : 'var(--text-primary)'
  return (
    <div className="panel stat">
      <span className="stat-label">{label}</span>
      <span className="stat-value" style={{ color: toneColor }}>
        {value}
        {unit && <span className="stat-unit"> {unit}</span>}
      </span>
      {sub && <span className="stat-sub">{sub}</span>}
    </div>
  )
}

/* -------------------------------------------------------------- sparkline */

export function Sparkline({ values, color = 'var(--series-1)', height = 38 }) {
  if (!values?.length) return null
  const w = 160
  const lo = Math.min(...values)
  const hi = Math.max(...values)
  const span = hi - lo || 1
  const d = values.map((v, i) =>
    `${i ? 'L' : 'M'}${((i / (values.length - 1 || 1)) * w).toFixed(1)},${(height - ((v - lo) / span) * height).toFixed(1)}`,
  ).join(' ')
  return (
    <svg width="100%" height={height} viewBox={`0 0 ${w} ${height}`} preserveAspectRatio="none" aria-hidden="true">
      <path d={d} fill="none" stroke={color} strokeWidth="2" strokeLinejoin="round" strokeLinecap="round" />
    </svg>
  )
}

/* ------------------------------------------------------- donut / share */

export function ShareBar({ parts }) {
  const total = parts.reduce((a, p) => a + p.value, 0) || 1
  return (
    <div>
      {/* 2px gaps between adjacent segments keep the boundaries readable. */}
      <div style={{ display: 'flex', gap: 2, height: 26, borderRadius: 4, overflow: 'hidden' }}>
        {parts.filter((p) => p.value > 0).map((p, i) => (
          <div
            key={p.label}
            title={`${p.label}: ${p.value}`}
            style={{ width: `${(p.value / total) * 100}%`, background: p.color || seriesColor(i) }}
          />
        ))}
      </div>
      <div className="legend" style={{ marginTop: 10, marginBottom: 0 }}>
        {parts.map((p, i) => (
          <span className="legend-item" key={p.label}>
            <span className="legend-swatch" style={{ background: p.color || seriesColor(i) }} />
            {p.label} <span className="mono muted">{p.value}</span>
          </span>
        ))}
      </div>
    </div>
  )
}
