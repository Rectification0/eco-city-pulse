import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 5173,
    // Dev-server equivalent of the Nginx proxy used in the container, so the
    // frontend calls the same relative /api/v1 paths in both environments.
    proxy: {
      '/api': {
        target: process.env.VITE_API_PROXY_TARGET ?? 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
    rollupOptions: {
      output: {
        // Plotly is ~4 MB and Leaflet ~150 kB. Left alone they land in the
        // entry chunk, so the app shell cannot render until the whole charting
        // stack has downloaded. Split out, the shell paints immediately, the
        // two vendor chunks load in parallel, and — because they change far
        // less often than application code — they stay cached across deploys.
        manualChunks: {
          plotly: ['plotly.js-dist-min'],
          leaflet: ['leaflet', 'react-leaflet'],
        },
      },
    },
    // The plotly chunk is legitimately large; warning about it on every build
    // trains everyone to ignore the warning.
    chunkSizeWarningLimit: 5000,
  },
})
