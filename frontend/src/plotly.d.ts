/**
 * `plotly.js-dist-min` ships no types of its own — it is the pre-bundled build
 * of `plotly.js`, whose types we do have. Pointing one at the other is more
 * honest than an `any`: the chart code is type-checked against the real Plotly
 * API, and a bad trace property is still a compile error.
 */
declare module 'plotly.js-dist-min' {
  import type * as Plotly from 'plotly.js'
  const plotly: typeof Plotly
  export = plotly
}
