// Standing Orders mission control: preview, orders, events, funnel. Plain ESM, no JSX, no build step.
// Only imports the SDK and React (catalog rule 8); all data comes from dashboard/plugin_api.py via ctx.rest.
import { PANES_AREA } from '@hermes/plugin-sdk'
import { useEffect, useState } from 'react'
import { jsx, jsxs } from 'react/jsx-runtime'

const h = (tag, props, ...children) => jsxs(tag, { ...props, children })
const dim = 'text-(--ui-text-tertiary)'

function usePoll(ctx, path, ms) {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  useEffect(() => {
    let live = true
    const load = () =>
      ctx.rest(path).then(
        d => live && (setData(d), setError(null)),
        e => live && setError(String(e?.message || e))
      )
    load()
    const t = setInterval(load, ms)
    return () => { live = false; clearInterval(t) }
  }, [ctx, path, ms])
  return [data, error]
}

function Button({ label, onClick }) {
  return jsx('button', {
    type: 'button', onClick,
    className: 'rounded border border-(--ui-border) px-1.5 py-0.5 text-xs hover:bg-(--ui-surface-hover)',
    children: label
  })
}

function Preview({ ctx }) {
  const [p] = usePoll(ctx, '/preview', 2000)
  if (!p?.image) return jsx('div', { className: `rounded border border-(--ui-border) p-6 text-center text-xs ${dim}`, children: 'No frames yet' })
  return jsx('img', { src: p.image, alt: 'Latest camera frame', className: 'w-full rounded border border-(--ui-border)' })
}

function Funnel({ f }) {
  const cells = [['frames seen', f.frames], ['changed', f.t0_passed], ['settled locally', f.t1_decided],
                 ['vision calls', f.t2_calls], ['firings', f.fires]]
  return h('div', { className: 'grid grid-cols-5 gap-1 text-center' },
    ...cells.map(([k, v]) => h('div', { key: k, className: 'rounded border border-(--ui-border) p-1' },
      jsx('div', { className: 'text-sm font-medium', children: String(v) }),
      jsx('div', { className: `text-[0.625rem] ${dim}`, children: k })))
}

function OrderCard({ o, act }) {
  const armed = o.status === 'armed'
  return h('div', { className: 'flex flex-col gap-1 rounded border border-(--ui-border) p-2' },
    h('div', { className: 'flex items-center justify-between gap-2' },
      jsx('span', { className: 'font-medium', children: o.status }),
      jsx(Button, { label: armed ? 'Pause' : 'Arm', onClick: () => act(o.id, armed ? 'paused' : 'armed') })),
    jsx('div', { children: o.text }),
    jsx('div', { className: `text-xs ${dim}`, children: `zones: ${o.zones.join(', ') || 'none'} · ${o.vision_calls_last_hour}/${o.vision_budget_per_hour} vision calls this hour · ${o.fires_last_hour} firings` }),
    jsx('pre', { className: `whitespace-pre-wrap text-xs ${dim}`, children: o.plan }))
}

function EventRow({ e, label }) {
  const when = new Date(e.fired_at * 1000).toLocaleTimeString()
  return h('div', { className: 'flex flex-col gap-1 rounded border border-(--ui-border) p-2' },
    jsx('div', { className: 'text-xs', children: `${when} · ${e.kind} · ${e.order_id} · ${Math.round((e.confidence || 0) * 100)}%` }),
    jsx('div', { children: e.reason || '' }),
    e.clip_path && jsx('div', { className: `break-all text-[0.625rem] ${dim}`, children: e.clip_path }),
    e.label
      ? jsx('div', { className: `text-xs ${dim}`, children: `labeled ${e.label}` })
      : h('div', { className: 'flex gap-1' },
          jsx(Button, { label: 'Right', onClick: () => label(e.id, 'correct') }),
          jsx(Button, { label: 'Wrong', onClick: () => label(e.id, 'wrong') })))
}

function Pane({ ctx }) {
  const [snap, error] = usePoll(ctx, '/snapshot', 3000)
  const [notice, setNotice] = useState('')
  const post = (path, body) =>
    ctx.rest(path, { method: 'POST', body }).then(r => setNotice(r.error || r.result || ''), e => setNotice(String(e?.message || e)))
  const act = (id, status) => post(`/orders/${encodeURIComponent(id)}/status`, { status })
  const label = (id, value) => post(`/events/${encodeURIComponent(id)}/label`, { label: value })

  if (error) return jsx('div', { className: `p-3 text-sm ${dim}`, children: `Standing Orders backend unavailable: ${error}. Enable the plugin in config.yaml (plugins.enabled).` })
  if (!snap) return jsx('div', { className: `p-3 text-sm ${dim}`, children: 'Loading…' })
  const cam = snap.camera
  const paused = cam.state !== 'running'
  return h('div', { className: 'flex h-full flex-col gap-3 overflow-y-auto p-3 text-sm' },
    jsx('div', { className: paused ? 'text-(--ui-warning, #b45309)' : dim,
      children: paused ? `Not watching: ${cam.state}${cam.detail ? ` (${cam.detail})` : ''}. A sleeping Mac stops watching.` : `Watching · ${cam.frames_seen ?? 0} frames` }),
    jsx(Preview, { ctx }),
    jsx(Funnel, { f: snap.funnel_last_hour }),
    notice && jsx('div', { className: 'text-xs', children: notice }),
    jsx('div', { className: 'font-medium', children: 'Orders' }),
    snap.orders.length ? snap.orders.map(o => jsx(OrderCard, { o, act }, o.id))
                       : jsx('div', { className: dim, children: 'No orders. Describe one in chat.' }),
    snap.list_items?.length ? h('div', { className: 'flex flex-col gap-1' },
      jsx('div', { className: 'font-medium', children: 'List' }),
      ...snap.list_items.map(i => jsx('div', { key: i.id, children: `• ${i.text}` }))) : null,
    jsx('div', { className: 'font-medium', children: 'Events' }),
    snap.events.length ? snap.events.map(e => jsx(EventRow, { e, label }, e.id))
                       : jsx('div', { className: dim, children: 'Nothing has fired.' }))
}

export default {
  id: 'standing-orders',
  name: 'Standing Orders',
  register(ctx) {
    ctx.register({
      id: 'pane', area: PANES_AREA, title: 'standing orders',
      data: { placement: 'right', width: '360px' },
      render: () => jsx(Pane, { ctx })
    })
  }
}
