import { useCallback, useEffect, useRef, useState } from 'react'
import { Activity, ArrowUpRight, Beaker, BookOpen, ChevronDown, Database, FlaskConical, History, LayoutDashboard, Menu, ScanLine, ShieldCheck, Sparkles, Stethoscope, WandSparkles, X } from 'lucide-react'
import { Link, NavLink, Route, Routes, useLocation } from 'react-router-dom'
import { api } from './api'
import { DataContext } from './context'
import { ErrorBox } from './components/UI'
import Overview from './pages/Overview'
import Datasets from './pages/Datasets'
import Inspector from './pages/Inspector'
import Repairs from './pages/Repairs'
import Training from './pages/Training'
import Results from './pages/Results'
import Predictions from './pages/Predictions'
import Research from './pages/Research'

const navigation = [ ['/', LayoutDashboard, 'Overview'], ['/datasets', Database, 'Datasets'], ['/inspector', ScanLine, 'Quality inspector'], ['/repairs', WandSparkles, 'Repair studio'], ['/training', FlaskConical, 'Train & evaluate'], ['/results', Activity, 'Results'], ['/predictions', Beaker, 'Predictions'], ['/research', History, 'History & reports'], ['/guide', BookOpen, 'How it works'] ]
export default function App() {
  const [datasets, setDatasets] = useState([]), [selected, select] = useState(localStorage.getItem('datadoctor.dataset') || '')
  const [dataset, setDataset] = useState(null), [profile, setProfile] = useState(null), [jobs,setJobs] = useState([]), [experiments,setExperiments] = useState([])
  const [error,setError] = useState(null), [online,setOnline] = useState(false), [mobile,setMobile] = useState(false)
  const [experimentId, setExperimentId] = useState(localStorage.getItem('datadoctor.experiment') || '')
  const location = useLocation(), fingerprint = useRef('')
  const setSelected = id => { select(id); localStorage.setItem('datadoctor.dataset',id) }
  const chooseExperiment = id => { setExperimentId(id); localStorage.setItem('datadoctor.experiment',id) }
  const refresh = useCallback(async () => {
    try {
      const [ds,ex] = await Promise.all([api('/datasets'),api('/experiments')]); setDatasets(ds); setExperiments(ex)
      if (selected && ds.some(d => d.id === selected)) {
        const detail = await api(`/datasets/${selected}`); setDataset(detail)
        const p = await api(`/datasets/${selected}/profile`).catch(() => null); setProfile(p)
      } else { setDataset(null); setProfile(null); if (ds.length) { select(ds[0].id); localStorage.setItem('datadoctor.dataset',ds[0].id) } }
      setError(null)
    } catch(e) { setError(e) }
  },[selected])
  useEffect(() => { setDataset(null); setProfile(null); refresh() },[refresh])
  useEffect(() => {
    let alive = true
    async function poll() {
      try {
        const [health,j] = await Promise.all([api('/health'),api('/jobs')]); if (!alive) return
        setOnline(health.status === 'ok'); setJobs(j)
        const stamp = j.map(x => `${x.id}:${x.status}`).join('|')
        if (stamp !== fingerprint.current) { fingerprint.current = stamp; refresh() }
      } catch { if (alive) setOnline(false) }
    }
    poll(); const timer = setInterval(poll,2000); return () => { alive=false; clearInterval(timer) }
  },[refresh])
  useEffect(() => setMobile(false),[location.pathname])
  const title = navigation.find(n => n[0] === location.pathname)?.[2] || 'Page not found'
  return <DataContext.Provider value={{datasets,dataset,profile,selected,setSelected,refresh,jobs,experiments,experimentId,chooseExperiment}}>
    <div className="app-shell">
      {mobile && <button className="backdrop" aria-label="Close navigation" onClick={() => setMobile(false)}/>}
      <aside className={`sidebar ${mobile ? 'open' : ''}`}>
        <Link className="brand" to="/"><span className="brand-icon"><Stethoscope size={25}/></span><span>DataDoctor<span className="brand-ai">AI</span><small>DATASET INTELLIGENCE</small></span></Link>
        <div className="workspace-tag"><span className="workspace-icon">D</span><div>Research workspace<small>Local · Single user</small></div><ChevronDown size={14}/></div>
        <p className="nav-label">WORKSPACE</p>
        <nav>{navigation.map(([path,Icon,label],i) => <NavLink end={path==='/'} key={path} to={path}><Icon size={18}/><span>{label}</span>{i===3 && <span className="tiny-ai" aria-hidden="true">AI</span>}</NavLink>)}</nav>
        <div className="sidebar-bottom"><ShieldCheck size={20}/><strong>Your data stays yours.</strong><p>Files and models are stored locally. Original data is preserved.</p><div className="connection"><i className={online?'online':''}/>{online?'Backend connected':'Backend unavailable'}</div></div>
      </aside>
      <div className="main-shell"><header className="topbar"><div className="row"><button className="icon-button mobile-menu" aria-label="Toggle navigation" onClick={() => setMobile(!mobile)}>{mobile?<X/>:<Menu/>}</button><span className="breadcrumb">Workspace <span>/</span> <b>{title}</b></span></div><div className="row"><span className="local-pill"><i/> LOCAL ENVIRONMENT</span><div className="avatar">DR</div></div></header>
        <main><div className="page-heading"><div><div className="eyebrow"><Sparkles size={13}/> EVIDENCE-BASED DATA REPAIR</div><h1>{title === 'Overview' ? 'A healthier dataset. Better evidence.' : title}</h1><p>{title==='Overview'?'Understand what’s wrong. Decide what to repair. Measure what changes.':'Inspect, investigate, and make every data decision explainable.'}</p></div><Link to="/datasets" className="button">Upload dataset <ArrowUpRight size={16}/></Link></div>
          {datasets.length>0 && <div className="dataset-switch"><Database size={16}/><label htmlFor="active-dataset">Active dataset</label><select id="active-dataset" value={selected} onChange={e=>setSelected(e.target.value)}>{datasets.map(d=><option key={d.id} value={d.id}>{d.name} · {d.rows.toLocaleString()} rows</option>)}</select>{dataset && <span className="muted mono">v{dataset.versions?.findIndex(v=>v.id===dataset.current_version)+1} · {dataset.current_version.slice(0,8)}</span>}</div>}
          <ErrorBox error={error}/><Routes><Route path="/" element={<Overview/>}/><Route path="/datasets" element={<Datasets/>}/><Route path="/inspector" element={<Inspector/>}/><Route path="/repairs" element={<Repairs/>}/><Route path="/training" element={<Training/>}/><Route path="/results" element={<Results/>}/><Route path="/predictions" element={<Predictions/>}/><Route path="/research" element={<Research/>}/><Route path="/guide" element={<Research guide/>}/><Route path="*" element={<p>Page not found. <Link to="/">Return to overview</Link></p>}/></Routes>
          <footer><span><Stethoscope size={14}/> DataDoctor AI</span><span>Explainable repairs. Reproducible experiments.</span><span>CPU · Local research platform</span></footer>
        </main>
      </div>
    </div>
  </DataContext.Provider>
}
