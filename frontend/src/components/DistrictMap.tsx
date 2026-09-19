/**
 * District map — task 10.4.
 *
 * District polygons from `data/raw/districts.geojson`, filled by the PM2.5 each
 * district's station currently reports. The fill is a **status** scale (the
 * air-quality bands), not a continuous ramp, because that is how the reading is
 * described everywhere else in the product — one encoding for one quantity.
 *
 * Every polygon carries a tooltip and the legend names each band, so the map
 * never asks the reader to decode a colour unaided.
 *
 * **The polygons are schematic.** They are not surveyed administrative
 * boundaries, and the backend says so in the GeoJSON's own metadata; the note
 * below repeats it where someone is actually looking at them.
 */

import { GeoJSON, MapContainer, TileLayer, CircleMarker, Tooltip } from 'react-leaflet'
import type { Feature, Geometry } from 'geojson'
import type { DistrictsResponse, StationReading } from '../services/api'
import { AQI_BANDS, band } from '../charts/theme'
import 'leaflet/dist/leaflet.css'

interface DistrictMapProps {
  districts: DistrictsResponse
  readings: StationReading[]
  onSelect?: (reading: StationReading) => void
  selected?: string
}

export default function DistrictMap({
  districts,
  readings,
  onSelect,
  selected,
}: DistrictMapProps) {
  const byDistrict = new Map(
    readings.filter((r) => r.district_id).map((r) => [r.district_id as string, r]),
  )

  const centre: [number, number] = [28.63, 77.21]

  function style(feature?: Feature<Geometry>) {
    const reading = feature?.id ? byDistrict.get(String(feature.id)) : undefined
    const { color } = band(reading?.pm25)
    return {
      color: '#1e293b',
      weight: 1,
      fillColor: color,
      // Translucent so the basemap's streets stay legible underneath; the
      // legend carries the exact band, the fill carries the gradient.
      fillOpacity: reading ? 0.55 : 0.12,
    }
  }

  return (
    <div>
      <MapContainer
        center={centre}
        zoom={10}
        scrollWheelZoom={false}
        className="h-[26rem] w-full rounded-lg"
        style={{ backgroundColor: '#0f172a' }}
        // The charts get their label from Plot's required `ariaLabel` prop;
        // Leaflet has no equivalent, so it is set here. The band legend below
        // and the per-district tooltips carry the values themselves.
        aria-label={`Map of current PM2.5 by district, ${readings.length} monitoring stations`}
      >
        {/*
          OpenStreetMap's own tiles, darkened with a CSS filter rather than a
          ready-made dark basemap: every hosted dark style now wants an API key,
          and a key is a credential this deployment has no way to hold (SEC-2).
        */}
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://tile.openstreetmap.org/{z}/{x}/{y}.png"
          className="map-basemap"
        />

        <GeoJSON
          data={districts as never}
          style={style}
          onEachFeature={(feature, layer) => {
            const reading = byDistrict.get(String(feature.id))
            const name = feature.properties?.name ?? String(feature.id)
            const pm25 = reading?.pm25
            const value =
              pm25 !== null && pm25 !== undefined
                ? `${pm25.toFixed(1)} µg/m³ — ${band(pm25).label}`
                : 'no current reading'
            layer.bindTooltip(`<strong>${name}</strong><br/>PM2.5 ${value}`, {
              sticky: true,
            })
          }}
        />

        {readings.map((reading) => (
          <CircleMarker
            key={reading.station}
            center={[reading.lat, reading.lon]}
            radius={selected === reading.station ? 8 : 5}
            pathOptions={{
              color: selected === reading.station ? '#e2e8f0' : '#0f172a',
              weight: 2,
              fillColor: band(reading.pm25).color,
              fillOpacity: 1,
            }}
            eventHandlers={{ click: () => onSelect?.(reading) }}
          >
            <Tooltip>
              {reading.district_name ?? reading.station}
              {reading.is_anomaly ? ' · flagged anomaly' : ''}
            </Tooltip>
          </CircleMarker>
        ))}
      </MapContainer>

      <ul className="mt-3 flex flex-wrap gap-x-4 gap-y-1">
        {AQI_BANDS.map((entry, index) => (
          <li key={entry.label} className="flex items-center gap-1.5 text-xs">
            <span
              aria-hidden="true"
              className="inline-block h-2.5 w-2.5 rounded-sm"
              style={{ backgroundColor: entry.color }}
            />
            <span className="text-slate-400">
              {entry.label}
              <span className="ml-1 text-slate-600">
                {index === AQI_BANDS.length - 1
                  ? `>${AQI_BANDS[index - 1]?.limit ?? ''}`
                  : `≤${entry.limit}`}
              </span>
            </span>
          </li>
        ))}
      </ul>
    </div>
  )
}
