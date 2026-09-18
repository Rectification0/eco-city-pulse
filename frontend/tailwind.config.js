/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        // Air-quality band colours, reused later by the map legend and KPI
        // tiles so a reading is coloured identically everywhere (Phase 10).
        aqi: {
          good: '#4ade80',
          moderate: '#facc15',
          unhealthy: '#fb923c',
          severe: '#f87171',
          hazardous: '#a855f7',
        },
      },
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        mono: ['ui-monospace', 'SFMono-Regular', 'Menlo', 'monospace'],
      },
    },
  },
  plugins: [],
}
