/**
 * A thin React wrapper over plotly.js.
 *
 * Written rather than pulled in: `react-plotly.js` is a ~40-line wrapper whose
 * published types and peer ranges lag React 18, and this is the same 40 lines
 * without the dependency. It also lets the chart own its resize behaviour,
 * which is what actually breaks when a Plotly chart lives in a flex layout.
 *
 * Plotly is loaded once here so every chart shares one instance, and the module
 * is only imported by the routes that chart — the Dashboard does not pay for
 * the EDA Studio's bundle.
 */

import { useEffect, useRef } from 'react'
import Plotly from 'plotly.js-dist-min'
import { PLOT_CONFIG } from './theme'

interface PlotProps {
  data: Plotly.Data[]
  layout: Partial<Plotly.Layout>
  /** Tailwind height class; Plotly needs a sized container to draw into. */
  className?: string
  ariaLabel: string
}

export default function Plot({ data, layout, className, ariaLabel }: PlotProps) {
  const node = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const element = node.current
    if (!element) return

    void Plotly.react(element, data, layout, PLOT_CONFIG)

    // Plotly sizes to the container at draw time, so a flex/grid parent that
    // settles afterwards leaves the chart at the wrong width until something
    // else forces a redraw. Observing the container fixes it at the source.
    const observer = new ResizeObserver(() => Plotly.Plots.resize(element))
    observer.observe(element)

    return () => {
      observer.disconnect()
      Plotly.purge(element)
    }
  }, [data, layout])

  return (
    <div
      ref={node}
      className={className ?? 'h-72 w-full'}
      role="img"
      aria-label={ariaLabel}
    />
  )
}
