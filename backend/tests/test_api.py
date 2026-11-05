import io
import time

import pandas as pd
import pytest

from app.store import Store


def wait(client,job_id):
    deadline=time.monotonic()+30
    while time.monotonic()<deadline:
        response=client.get(f'/api/jobs/{job_id}')
        assert response.status_code==200,response.text
        result=response.json()
        if result['status'] in ('completed','failed'):
            assert result['status']=='completed',result
            return result
        time.sleep(.02)
    raise AssertionError('Job did not complete in 30 seconds')


def upload(client,frame):
    response=client.post('/api/datasets',files={'file':('fixture.csv',frame.to_csv(index=False).encode(),'text/csv')})
    assert response.status_code==201,response.text
    value=response.json();wait(client,value['job']['id']);return value['dataset']


@pytest.mark.parametrize('task',['classification','regression'])
def test_end_to_end_api(client,frame,cfg,task):
    if task=='regression':frame['label']=frame.age*3+frame.signal
    dataset=upload(client,frame);id=dataset['id']
    assert client.get('/api/health').json()['status']=='ok'
    assert len(client.get('/api/datasets').json())==1
    response=client.get(f'/api/datasets/{id}/profile')
    assert response.status_code==200,response.text
    assert response.json()['missing_total']>0
    selection=dict(target='label',task=task,features=cfg['features'])
    assert client.put(f'/api/datasets/{id}/selection',json=selection).status_code==200
    operation=dict(kind='impute',column='income',method='mean')
    plan=dict(operations=[operation,dict(kind='drop_duplicates')])
    response=client.post(f'/api/datasets/{id}/repairs/preview',json=plan)
    assert response.status_code==200 and response.json()['after']['missing']==0
    version=client.post(f'/api/datasets/{id}/repairs',json=plan).json();wait(client,version['job']['id'])
    assert version['version']['id']!=dataset['original_version']
    cfg.update(dataset_id=id,version_id=version['version']['id'],task=task,algorithm='forest')
    assert client.post('/api/training/validate',json=cfg).json()['valid']
    response=client.post('/api/training',json=cfg)
    assert response.status_code==202,response.text
    job=wait(client,response.json()['id'])
    eid=job['result']['experiment_id']
    result=client.get(f'/api/experiments/{eid}').json()
    assert len(result['result']['arms'])==3
    assert client.get('/api/experiments').json()[0]['id']==eid
    assert client.get(f'/api/experiments/{eid}/model').status_code==200
    saved=client.get(f'/api/experiments/{eid}/report').json()['operations']
    assert [{k:v for k,v in op.items() if v is not None} for op in saved]==plan['operations']
    row=frame[cfg['features']].iloc[1].to_dict()
    assert client.post(f'/api/experiments/{eid}/predict',json=[row]).status_code==200
    assert client.post(f'/api/experiments/{eid}/predict',json=[{'wrong':1}]).status_code==422
    csv=frame[cfg['features']].iloc[:3].to_csv(index=False).encode()
    prediction=client.post(f'/api/experiments/{eid}/predict-batch',files={'file':('batch.csv',csv,'text/csv')})
    assert prediction.status_code==200 and 'datadoctor_prediction' in prediction.text
    assert client.get(f'/api/datasets/{id}/report').status_code==200
    source=client.get(f'/api/datasets/{id}/source').content
    assert source==frame.to_csv(index=False).encode()
    assert client.post(f'/api/datasets/{id}/restore').json()['current_version']==dataset['original_version']
    assert len(client.get(f'/api/datasets/{id}').json()['versions'])==2
    assert client.delete(f'/api/datasets/{id}').status_code==204
    assert client.get(f'/api/experiments/{eid}').status_code==404
    assert client.get('/api/datasets').json()==[]


def test_errors_and_isolation(client,frame,cfg):
    assert client.post('/api/datasets',files={'file':('evil.joblib',b'unsafe')}).status_code==422
    a=upload(client,frame);b=upload(client,frame)
    assert client.get(f'/api/datasets/{a["id"]}/records?version_id={b["current_version"]}').status_code==422
    assert client.get('/api/datasets/not-found').status_code==404
    cfg.update(dataset_id=a['id'],version_id=b['current_version'])
    assert client.post('/api/training',json=cfg).status_code==422
    assert client.get(f'/api/datasets/{a["id"]}/records?sort=unknown').status_code==422
    records=client.get(f'/api/datasets/{a["id"]}/records?limit=10&search=north&sort=age').json()
    assert len(records['rows'])<=10 and records['total']>0


def test_recover_interrupted_jobs(tmp_path):
    store=Store(tmp_path)
    j=store.put('job',dict(status='running',stage='fitting'))
    store.recover_jobs()
    assert store.get('job',j['id'])['status']=='failed'


def test_oversized_body_rejected_before_parsing(client):
    response=client.post('/api/datasets',content=b'a',headers={'Content-Length':str(22*1024*1024)})
    assert response.status_code==413
