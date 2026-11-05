const base = import.meta.env.VITE_API_URL || ''
export const url = path => `${base}/api${path}`
export async function api(path, options = {}) {
  const isForm = options.body instanceof FormData
  const response = await fetch(url(path), { ...options, headers: { ...(!isForm && options.body ? { 'Content-Type': 'application/json' } : {}), ...options.headers }, body: options.body && !isForm ? JSON.stringify(options.body) : options.body })
  if (!response.ok) {
    const error = await response.json().catch(() => ({}))
    throw new Error(typeof error.detail === 'string' ? error.detail : `Request failed (${response.status}).`)
  }
  return response.status === 204 ? null : response.json()
}
export async function downloadPost(path, body, filename) {
  const response = await fetch(url(path), { method: 'POST', body })
  if (!response.ok) { const e = await response.json(); throw new Error(e.detail || 'Download failed.') }
  const href = URL.createObjectURL(await response.blob())
  const a = document.createElement('a'); a.href = href; a.download = filename; a.click(); URL.revokeObjectURL(href)
}
