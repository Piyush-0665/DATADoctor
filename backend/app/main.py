import io
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response

from app.ml.profiler import MAX_BYTES, parse_upload, profile
from app.ml.repairs import repair_dataset
from app.ml.trainer import predict, prepare, run_experiment, split_data
from app.schemas import RepairRequest, Selection, TrainingConfig
from app.store import Store, safe_json, uid


def export_csv(frame):
    """Escape spreadsheet formulas in textual cells, including leading whitespace."""
    frame = frame.copy()
    for c in frame:
        frame[c] = frame[c].map(lambda v: "'"+v if isinstance(v,str) and v.lstrip().startswith(('=','+','-','@')) else v)
    frame.columns = ["'"+c if c.lstrip().startswith(('=','+','-','@')) else c for c in frame.columns]
    return frame.to_csv(index=False).encode('utf-8-sig')


class BodyLimit:
    """Also bound chunked request bodies before file spooling."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        count = 0
        async def limited_receive():
            nonlocal count
            message = await receive()
            count += len(message.get('body', b''))
            if count > MAX_BYTES + 1024 * 1024:
                raise HTTPException(413, 'Request exceeds the 21 MiB body limit.')
            return message
        await self.app(scope, limited_receive if scope['type'] == 'http' else receive, send)


def create_app(storage=None):
    store = Store(storage)
    executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix='datadoctor')

    @asynccontextmanager
    async def lifespan(app):
        store.recover_jobs()
        yield
        executor.shutdown(wait=True, cancel_futures=True)

    app = FastAPI(title='DataDoctor AI', version='1.0.0', lifespan=lifespan)
    app.state.store = store
    app.add_middleware(BodyLimit)
    app.add_middleware(CORSMiddleware, allow_origins=os.getenv('DATADOCTOR_CORS','http://localhost:5173,http://127.0.0.1:5173').split(','), allow_methods=['GET','POST','PUT','DELETE'], allow_headers=['Content-Type'])

    @app.middleware('http')
    async def request_size_limit(request, call_next):
        # Reject declared oversized bodies before multipart parsing can spool them.
        length = request.headers.get('content-length')
        if length:
            try:
                if int(length) > MAX_BYTES + 1024 * 1024:
                    return JSONResponse(status_code=413, content={'detail':'Request exceeds the 21 MiB body limit (20 MiB file plus multipart overhead).'})
            except ValueError:
                return JSONResponse(status_code=400, content={'detail':'Invalid Content-Length header.'})
        return await call_next(request)

    @app.exception_handler(KeyError)
    async def not_found(request, exc):
        return JSONResponse(status_code=404,content={'detail':str(exc).strip("'")})

    @app.exception_handler(ValueError)
    async def bad_request(request, exc):
        return JSONResponse(status_code=422,content={'detail':str(exc)})

    @app.exception_handler(RequestValidationError)
    async def invalid_schema(request, exc):
        return JSONResponse(status_code=422,content={'detail':'; '.join(f'{".".join(map(str,e["loc"]))}: {e["msg"]}' for e in exc.errors())})

    def submit(kind, dataset_id, config, work):
        job = store.put('job',dict(kind=kind,dataset_id=dataset_id,config=config,status='queued',stage='Queued',logs=[],error=None))

        def execute():
            start = time.monotonic()

            def update(stage):
                current = store.get('job',job['id'])
                current.update(status='running',stage=stage,elapsed_seconds=time.monotonic()-start)
                current['logs'].append(dict(seconds=round(time.monotonic()-start,2),message=stage))
                store.put('job',current)

            try:
                update('Starting worker')
                result = work(update,job['id'])
                current = store.get('job',job['id'])
                current.update(status='completed',stage='Completed',result=result,elapsed_seconds=time.monotonic()-start)
                store.put('job',current)
            except Exception as exc:
                current = store.get('job',job['id'])
                current.update(status='failed',stage='Failed',error=f'{type(exc).__name__}: {exc}',elapsed_seconds=time.monotonic()-start)
                store.put('job',current)
        executor.submit(execute)
        return job

    def get_dataset(identifier):
        return store.get('dataset',identifier)

    def version_for(dataset, version_id=None):
        version = store.get('version',version_id or dataset['current_version'])
        if version['dataset_id'] != dataset['id']:
            raise ValueError('Version does not belong to this dataset.')
        return version

    def start_analysis(dataset, version):
        selection = dataset.get('selection') or {}
        def work(update, job_id):
            update('Profiling columns, missingness, duplicates and distributions')
            result = profile(store.frame(version['id']),selection.get('target'),selection.get('task'))
            result.update(dataset_id=dataset['id'],version_id=version['id'])
            store.put('profile',dict(id=job_id,**result))
            return dict(profile_id=job_id)
        return submit('analysis',dataset['id'],dict(version_id=version['id'],selection=selection),work)

    @app.get('/api/health')
    def health():
        return dict(status='ok',device='CPU',mode='local-single-user',upload_limit_mib=20,max_rows=50000,max_columns=200)

    @app.post('/api/datasets',status_code=201)
    async def upload(file: UploadFile=File(...),sheet: str|None=None):
        filename = (file.filename or 'dataset').replace('\\','/').rsplit('/',1)[-1]
        content = await file.read(MAX_BYTES+1)
        frame,sheets = parse_upload(content,filename,sheet)
        if frame is None:
            return dict(requires_sheet=True,sheets=sheets)
        dataset_id = uid()
        store.path('datasets',dataset_id,'source').write_bytes(content)
        version = store.save_frame(dataset_id,frame,[],original=True)
        dataset = store.put('dataset',dict(id=dataset_id,name=filename,sheet=sheet or (sheets[0] if sheets else None),
                                           original_version=version['id'],current_version=version['id'],rows=len(frame),columns=list(frame.columns),selection=None))
        job = start_analysis(dataset,version)
        return dict(dataset=dataset,job=job)

    @app.get('/api/datasets')
    def datasets():
        return store.all('dataset')

    @app.get('/api/datasets/{dataset_id}')
    def dataset_detail(dataset_id: str):
        dataset = get_dataset(dataset_id)
        return dict(**dataset,versions=[v for v in store.all('version') if v['dataset_id']==dataset_id])

    @app.delete('/api/datasets/{dataset_id}',status_code=204)
    def delete_dataset(dataset_id: str):
        get_dataset(dataset_id)
        jobs = [j for j in store.all('job') if j['dataset_id']==dataset_id]
        if any(j['status'] in ('running','queued') for j in jobs):
            raise HTTPException(409,'Wait for active jobs before deleting this dataset.')
        for kind in ('version','profile','experiment','job'):
            for entry in store.all(kind):
                if entry.get('dataset_id')==dataset_id:
                    if kind=='version':
                        store.path('datasets',entry['id'],'pkl').unlink(missing_ok=True)
                    elif kind=='experiment':
                        store.path('models',entry['id'],'joblib').unlink(missing_ok=True)
                    store.delete(kind,entry['id'])
        store.path('datasets',dataset_id,'source').unlink(missing_ok=True)
        store.delete('dataset',dataset_id)
        return Response(status_code=204)

    @app.put('/api/datasets/{dataset_id}/selection')
    def selection(dataset_id: str, request: Selection):
        dataset = get_dataset(dataset_id)
        frame = store.frame(dataset['original_version'])
        if request.target and request.target not in frame:
            raise ValueError('Target does not exist.')
        if any(c not in frame for c in request.features) or request.target in request.features:
            raise ValueError('Choose existing input features excluding the target.')
        if request.identifier and request.identifier not in frame:
            raise ValueError('Identifier does not exist.')
        dataset['selection'] = request.model_dump()
        store.put('dataset',dataset)
        return dataset

    @app.get('/api/datasets/{dataset_id}/records')
    def records(dataset_id: str,version_id: str|None=None,offset: int=Query(0,ge=0),limit: int=Query(25,ge=1,le=100),search: str='',sort: str|None=None,descending: bool=False):
        dataset = get_dataset(dataset_id)
        frame = store.frame(version_for(dataset,version_id)['id'])
        if search:
            mask = frame.astype(str).apply(lambda col:col.str.contains(search,case=False,regex=False)).any(axis=1)
            frame = frame.loc[mask]
        if sort:
            if sort not in frame:
                raise ValueError('Unknown sort column.')
            frame = frame.sort_values(sort,ascending=not descending,key=lambda s:s.astype(str) if s.dtype=='object' else s)
        page = frame.iloc[offset:offset+limit]
        return safe_json(dict(total=len(frame),columns=list(frame),rows=page.to_dict('records'),row_ids=page.index.tolist()))

    @app.post('/api/datasets/{dataset_id}/analysis',status_code=202)
    def analyze(dataset_id: str,version_id: str|None=None):
        dataset = get_dataset(dataset_id)
        return start_analysis(dataset,version_for(dataset,version_id))

    @app.get('/api/datasets/{dataset_id}/profile')
    def get_profile(dataset_id: str,version_id: str|None=None):
        dataset = get_dataset(dataset_id)
        vid = version_for(dataset,version_id)['id']
        profiles = [p for p in store.all('profile') if p['dataset_id']==dataset_id and p['version_id']==vid]
        if not profiles:
            raise HTTPException(404,'Analysis is not complete for this version. Run analysis first.')
        return profiles[0]

    @app.get('/api/datasets/{dataset_id}/issues/{issue_id}/records')
    def issue_records(dataset_id: str,issue_id: str,version_id: str|None=None):
        p = get_profile(dataset_id,version_id)
        issue = next((i for i in p['issues'] if i['id']==issue_id),None)
        if not issue:
            raise KeyError('Issue not found')
        frame = store.frame(p['version_id'])
        return safe_json(dict(rows=frame.loc[issue['rows']].to_dict('records'),row_ids=issue['rows'],limit=100))

    @app.post('/api/datasets/{dataset_id}/repairs/preview')
    def preview_repairs(dataset_id: str,request: RepairRequest):
        dataset = get_dataset(dataset_id)
        original = store.frame(dataset['original_version'])
        target = (dataset.get('selection') or {}).get('target')
        repaired,changes = repair_dataset(original,[o.model_dump() for o in request.operations],target)
        return safe_json(dict(before=dict(rows=len(original),columns=len(original.columns),missing=int(original.isna().sum().sum())),
                              after=dict(rows=len(repaired),columns=len(repaired.columns),missing=int(repaired.isna().sum().sum())),changes=changes,preview=repaired.head(10).to_dict('records'),
                              note='Export preview learns from the full dataset. Experiments instead fit repairs on training data only. All operations are relative to the immutable original.'))

    @app.post('/api/datasets/{dataset_id}/repairs',status_code=201)
    def apply_repairs(dataset_id: str,request: RepairRequest):
        dataset = get_dataset(dataset_id)
        operations = [o.model_dump() for o in request.operations]
        repaired,changes = repair_dataset(store.frame(dataset['original_version']),operations,(dataset.get('selection') or {}).get('target'))
        version = store.save_frame(dataset_id,repaired,operations,parent=dataset['original_version'])
        dataset['current_version'] = version['id']
        store.put('dataset',dataset)
        job = start_analysis(dataset,version)
        return dict(version=version,changes=safe_json(changes),job=job)

    @app.post('/api/datasets/{dataset_id}/restore')
    def restore(dataset_id: str):
        dataset = get_dataset(dataset_id)
        dataset['current_version'] = dataset['original_version']
        return store.put('dataset',dataset)

    @app.get('/api/datasets/{dataset_id}/download')
    def download(dataset_id: str,version_id: str|None=None):
        dataset = get_dataset(dataset_id)
        version = version_for(dataset,version_id)
        return Response(export_csv(store.frame(version['id'])),media_type='text/csv',headers={'Content-Disposition':f'attachment; filename="datadoctor-{version["id"][:8]}.csv"'})

    @app.get('/api/datasets/{dataset_id}/source')
    def source(dataset_id: str):
        d = get_dataset(dataset_id)
        return FileResponse(store.path('datasets',dataset_id,'source'),filename=d['name'])

    @app.get('/api/datasets/{dataset_id}/report')
    def quality_report(dataset_id: str):
        dataset = get_dataset(dataset_id)
        return JSONResponse(content=safe_json(dict(dataset=dataset,profile=get_profile(dataset_id),versions=[v for v in store.all('version') if v['dataset_id']==dataset_id])),headers={'Content-Disposition':'attachment; filename="quality-and-repairs.json"'})

    @app.get('/api/jobs')
    def jobs(dataset_id: str|None=None):
        return [{k:v for k,v in j.items() if k!='result'} for j in store.all('job') if dataset_id is None or j['dataset_id']==dataset_id]

    @app.get('/api/jobs/{job_id}')
    def job_status(job_id: str):
        return store.get('job',job_id)

    @app.get('/api/algorithms')
    def algorithms():
        return dict(classification=['logistic','tree','forest','gradient'],regression=['linear','ridge','tree','forest','gradient'],baseline='DummyClassifier(prior) / DummyRegressor(mean)')

    def training_inputs(cfg):
        dataset = get_dataset(cfg['dataset_id'])
        version = version_for(dataset,cfg['version_id'])
        original = store.frame(dataset['original_version'])
        return original,version['operations']

    @app.post('/api/training/validate')
    def validate_training(request: TrainingConfig):
        cfg = request.model_dump()
        frame,operations = training_inputs(cfg)
        data = prepare(frame,cfg,operations)
        train,val,test,_,method = split_data(data,cfg,operations)
        return dict(valid=True,train=len(train),validation=len(val),test=len(test),method=method,approved_repairs=operations,config=cfg)

    @app.post('/api/training',status_code=202)
    def train(request: TrainingConfig):
        cfg = request.model_dump()
        frame,operations = training_inputs(cfg)
        prepare(frame,cfg,operations)
        def work(update,job_id):
            result = run_experiment(frame,cfg,operations,store.path('models',job_id,'joblib'),update)
            store.put('experiment',dict(id=job_id,dataset_id=cfg['dataset_id'],config=cfg,operations=operations,status='completed',result=result))
            return dict(experiment_id=job_id)
        with store.lock:
            if any(j['kind']=='training' and j['status'] in ('queued','running') for j in store.all('job')):
                raise HTTPException(409,'A training job is already active. Wait for it to finish.')
            return submit('training',cfg['dataset_id'],cfg,work)

    @app.get('/api/experiments')
    def experiments():
        return [dict(id=e['id'],dataset_id=e['dataset_id'],created=e['created'],config=e['config'],status=e['status'],comparison=e['result']['primary_comparison']) for e in store.all('experiment')]

    @app.get('/api/experiments/{experiment_id}')
    def experiment(experiment_id: str):
        return store.get('experiment',experiment_id)

    @app.get('/api/experiments/{experiment_id}/report')
    def experiment_report(experiment_id: str):
        return JSONResponse(content=experiment(experiment_id),headers={'Content-Disposition':'attachment; filename="experiment-report.json"'})

    @app.get('/api/experiments/{experiment_id}/model')
    def model(experiment_id: str):
        experiment(experiment_id)
        return FileResponse(store.path('models',experiment_id,'joblib'),filename=f'datadoctor-{experiment_id[:8]}.joblib',media_type='application/octet-stream')

    def bundle(experiment_id):
        experiment(experiment_id)
        return joblib.load(store.path('models',experiment_id,'joblib'))

    @app.post('/api/experiments/{experiment_id}/predict')
    def single_prediction(experiment_id: str,records: list[dict],arm: str='repaired'):
        return dict(predictions=predict(bundle(experiment_id),records,arm),note='Probabilities are uncalibrated estimates, not guaranteed confidence.')

    @app.post('/api/experiments/{experiment_id}/predict-batch')
    async def batch_prediction(experiment_id: str,file: UploadFile=File(...),arm: str='repaired'):
        if not (file.filename or '').lower().endswith('.csv'):
            raise ValueError('Batch predictions require CSV.')
        frame,_ = parse_upload(await file.read(MAX_BYTES+1),file.filename)
        records = safe_json(frame.to_dict('records'))
        predictions = predict(bundle(experiment_id),records,arm)
        output = frame.copy()
        output['datadoctor_prediction'] = [r['prediction'] for r in predictions]
        return Response(export_csv(output),media_type='text/csv',headers={'Content-Disposition':'attachment; filename="predictions.csv"'})

    return app


app = create_app()
