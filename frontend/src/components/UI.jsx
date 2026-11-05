import { AlertCircle, ArrowUpRight, Download as DownloadGlyph, LoaderCircle } from 'lucide-react'
import { Link } from 'react-router-dom'
import { url } from '../api'

export const fmt = value => value == null ? '—' : typeof value === 'number' ? Number.isInteger(value) ? value.toLocaleString() : value.toLocaleString(undefined, { maximumFractionDigits: 4 }) : String(value)
export function Button({ children, secondary, danger, busy, ...props }) { return <button className={`button ${secondary ? 'secondary' : ''} ${danger ? 'danger' : ''}`} disabled={busy || props.disabled} {...props}>{busy && <LoaderCircle size={15} className="spin" />}{children}</button> }
export function Badge({ children, tone = '' }) { return <span className={`badge ${tone}`}>{children}</span> }
export function Empty({ title = 'Your research starts with a dataset', children, to = '/datasets', action = 'Upload a dataset' }) { return <div className="empty"><div className="empty-orbit"><span>✳</span></div><h2>{title}</h2><p>{children || 'Bring your own CSV or Excel file. Every insight, repair, and result will be computed from your data.'}</p>{to && <Link className="button" to={to}>{action}<ArrowUpRight size={16}/></Link>}</div> }
export function ErrorBox({ error }) { return error && <div className="error" role="alert"><AlertCircle size={18}/><span>{error.message || String(error)}</span></div> }
export function Note({ children }) { return <div className="note"><AlertCircle size={16}/><div>{children}</div></div> }
export function Panel({ title, subtitle, action, children, className = '' }) { return <section className={`panel ${className}`}><div className="panel-head"><div><h3>{title}</h3>{subtitle && <p>{subtitle}</p>}</div>{action}</div>{children}</section> }
export function Stat({ title, value, detail, icon: Icon, accent }) { return <div className={`stat ${accent ? 'accent' : ''}`}><div className="stat-top"><span>{title}</span>{Icon && <Icon size={18}/>}</div><strong>{value ?? '—'}</strong><small>{detail}</small></div> }
export function Download({ path, children }) { return <a className="button secondary" href={url(path)} download><DownloadIcon/>{children}</a> }
function DownloadIcon() { return <DownloadGlyph size={15}/> }
export function Table({ rows = [], columns, rowIds, sort, onSort }) {
  columns = columns || Object.keys(rows[0] || {})
  if (!rows.length) return <p className="muted pad">No records to display.</p>
  return <div className="table-scroll"><table><thead><tr>{rowIds && <th>Row ID</th>}{columns.map(c => <th key={c}>{onSort ? <button className="sort" onClick={() => onSort(c)}>{c}{sort === c ? ' ↕' : ''}</button> : c}</th>)}</tr></thead><tbody>{rows.map((r, i) => <tr key={i}>{rowIds && <td className="mono">{rowIds[i]}</td>}{columns.map(c => <td key={c} title={fmt(r[c])}>{r[c] == null ? <span className="null">null</span> : fmt(r[c])}</td>)}</tr>)}</tbody></table></div>
}
export function Job({ job }) { if (!job) return null; return <div className="job"><div className="row spread"><h3>{job.status === 'running' || job.status === 'queued' ? <LoaderCircle size={18} className="spin"/> : null}{job.stage}</h3><Badge tone={job.status === 'failed' ? 'red' : 'green'}>{job.status}</Badge></div><p className="muted">Elapsed {fmt(job.elapsed_seconds || 0)} seconds · Real backend stages; no estimated percentage</p><ErrorBox error={job.error}/><div className="job-logs">{job.logs?.map((l,i) => <div key={i}><span>{l.seconds.toFixed(2)}s</span>{l.message}</div>)}</div></div> }
