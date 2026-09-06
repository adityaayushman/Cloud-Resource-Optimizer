import { useEffect, useRef } from 'react'

/**
 * The animated backdrop for the ignition screen.
 *
 * It draws what the system actually does rather than generic particles: a
 * lattice of nodes (the fleet) under a travelling demand wave, where each node
 * brightens as the wave passes over it and dims behind it. That is the loop the
 * whole project implements - capacity following forecast demand - so the motion
 * carries meaning instead of decorating.
 *
 * Cheap by construction: one canvas, no library, no per-frame allocation, and it
 * stops entirely when the tab is hidden or the viewer prefers reduced motion.
 */
export default function HeroCanvas({ intensity = 1 }) {
  const ref = useRef(null)

  useEffect(() => {
    const canvas = ref.current
    if (!canvas) return undefined
    const ctx = canvas.getContext('2d', { alpha: true })
    if (!ctx) return undefined

    const reduced = window.matchMedia?.('(prefers-reduced-motion: reduce)')
    let raf = 0
    let width = 0
    let height = 0
    let dpr = 1
    let nodes = []

    const SPACING = 46          // lattice pitch in CSS pixels
    const WAVE_SPEED = 0.00022  // radians per millisecond

    function layout() {
      const rect = canvas.getBoundingClientRect()
      dpr = Math.min(window.devicePixelRatio || 1, 2)
      width = rect.width
      height = rect.height
      canvas.width = Math.round(width * dpr)
      canvas.height = Math.round(height * dpr)
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)

      // Jitter each node off the exact lattice so it reads as a datacentre
      // rather than graph paper. Seeded off the index so it is stable across
      // resizes - nodes should not teleport when the window changes.
      nodes = []
      const cols = Math.ceil(width / SPACING) + 1
      const rows = Math.ceil(height / SPACING) + 1
      for (let r = 0; r < rows; r += 1) {
        for (let c = 0; c < cols; c += 1) {
          const seed = Math.sin(r * 127.1 + c * 311.7) * 43758.5453
          const jx = (seed - Math.floor(seed) - 0.5) * SPACING * 0.55
          const seed2 = Math.sin(r * 269.5 + c * 183.3) * 43758.5453
          const jy = (seed2 - Math.floor(seed2) - 0.5) * SPACING * 0.55
          nodes.push({ x: c * SPACING + jx, y: r * SPACING + jy })
        }
      }
    }

    /** Demand at a horizontal position: a few summed sines, so it is smooth but
     *  not obviously periodic - much like the workloads the system forecasts. */
    function demandAt(x, t) {
      const u = x / Math.max(width, 1)
      return (
        Math.sin(u * 6.1 + t * 1.0) * 0.42 +
        Math.sin(u * 11.3 - t * 0.6) * 0.26 +
        Math.sin(u * 19.7 + t * 1.7) * 0.13
      )
    }

    function frame(now) {
      const t = now * WAVE_SPEED
      ctx.clearRect(0, 0, width, height)

      const midY = height * 0.56
      const amp = Math.min(height * 0.2, 130)

      // --- the fleet ------------------------------------------------------
      for (let i = 0; i < nodes.length; i += 1) {
        const n = nodes[i]
        const waveY = midY + demandAt(n.x, t) * amp
        const distance = Math.abs(n.y - waveY)
        // Brightness falls off with distance from the wave; nodes below it (in
        // its wake) stay warmer than those ahead, the way provisioned capacity
        // lingers after a spike.
        const ahead = n.y < waveY
        const reach = ahead ? 92 : 150
        const glow = Math.max(0, 1 - distance / reach)
        const alpha = (0.06 + glow * glow * 0.68) * intensity
        if (alpha < 0.02) continue

        const radius = 1.1 + glow * 2.0
        ctx.beginPath()
        ctx.arc(n.x, n.y, radius, 0, Math.PI * 2)
        ctx.fillStyle = ahead
          ? `rgba(88, 166, 255, ${alpha})`
          : `rgba(0, 242, 254, ${alpha * 0.9})`
        ctx.fill()
      }

      // --- the demand curve ----------------------------------------------
      ctx.beginPath()
      for (let x = 0; x <= width; x += 6) {
        const y = midY + demandAt(x, t) * amp
        if (x === 0) ctx.moveTo(x, y)
        else ctx.lineTo(x, y)
      }
      const stroke = ctx.createLinearGradient(0, 0, width, 0)
      stroke.addColorStop(0, 'rgba(0, 242, 254, 0)')
      stroke.addColorStop(0.25, `rgba(0, 242, 254, ${0.55 * intensity})`)
      stroke.addColorStop(0.75, `rgba(144, 133, 233, ${0.55 * intensity})`)
      stroke.addColorStop(1, 'rgba(144, 133, 233, 0)')
      ctx.strokeStyle = stroke
      ctx.lineWidth = 1.6
      ctx.stroke()

      raf = window.requestAnimationFrame(frame)
    }

    function start() {
      if (raf) return
      raf = window.requestAnimationFrame(frame)
    }
    function stop() {
      if (!raf) return
      window.cancelAnimationFrame(raf)
      raf = 0
    }

    function renderStatic() {
      // Reduced motion still gets the picture, just frozen.
      stop()
      frame(3000)
      stop()
    }

    function onVisibility() {
      if (document.hidden) stop()
      else if (!reduced?.matches) start()
    }

    function onResize() {
      layout()
      if (reduced?.matches) renderStatic()
    }

    layout()
    if (reduced?.matches) renderStatic()
    else start()

    window.addEventListener('resize', onResize)
    document.addEventListener('visibilitychange', onVisibility)
    reduced?.addEventListener?.('change', onResize)

    return () => {
      stop()
      window.removeEventListener('resize', onResize)
      document.removeEventListener('visibilitychange', onVisibility)
      reduced?.removeEventListener?.('change', onResize)
    }
  }, [intensity])

  return <canvas ref={ref} className="hero-canvas" aria-hidden="true" />
}
