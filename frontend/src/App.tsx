/**
 * Routing — task 10.2.
 *
 * The three secondary screens are lazy, so the Dashboard's first paint does not
 * wait on code it will not run. Plotly and Leaflet are split into their own
 * vendor chunks (see `vite.config.ts`) rather than being bundled into any one
 * route: the Dashboard charts too, so hiding Plotly behind a lazy route would
 * only move the download, not avoid it.
 */

import { Suspense, lazy } from 'react'
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'

const EdaStudio = lazy(() => import('./pages/EdaStudio'))
const ModelLab = lazy(() => import('./pages/ModelLab'))
const Admin = lazy(() => import('./pages/Admin'))

function Loading() {
  return (
    <div className="flex h-96 items-center justify-center text-sm text-slate-500">
      Loading…
    </div>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<Dashboard />} />
          <Route
            path="eda"
            element={
              <Suspense fallback={<Loading />}>
                <EdaStudio />
              </Suspense>
            }
          />
          <Route
            path="models"
            element={
              <Suspense fallback={<Loading />}>
                <ModelLab />
              </Suspense>
            }
          />
          <Route
            path="admin"
            element={
              <Suspense fallback={<Loading />}>
                <Admin />
              </Suspense>
            }
          />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
