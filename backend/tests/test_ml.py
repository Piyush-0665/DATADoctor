import joblib
import numpy as np
import pandas as pd
import pytest

from app.ml.trainer import build_pipeline,predict,prepare,run_experiment,split_data


def test_splits_are_reproducible_and_duplicate_safe(frame,cfg):
    data=prepare(frame,cfg,[])
    a,b,c,groups,_=split_data(data,cfg)
    second=split_data(data,cfg)
    assert all(np.array_equal(x,y) for x,y in zip((a,b,c),second[:3]))
    assert not (set(groups[a])&set(groups[b]) or set(groups[a])&set(groups[c]) or set(groups[b])&set(groups[c]))
    assert len(a)+len(b)+len(c)==len(frame)


def test_groups_and_exclusions_do_not_leak(frame,cfg):
    frame=frame.copy();frame['entity']=np.arange(len(frame))//4
    cfg=dict(cfg,split='group',group_column='entity')
    a,b,c,_,_=split_data(prepare(frame,cfg,[]),cfg)
    assert not set(frame.iloc[a].entity)&set(frame.iloc[c].entity)
    # A unique ID must not make true duplicate inputs appear independent.
    frame['unique_id']=np.arange(len(frame));cfg['features']+=['unique_id']
    ops=[dict(kind='drop_column',column='unique_id')]
    a,b,c,g,_=split_data(prepare(frame,cfg,ops),cfg,ops)
    assert not set(g[a])&set(g[c])


def test_chronological_split_and_overlap_error(frame,cfg):
    frame=frame.iloc[:140].copy();frame['time']=pd.date_range('2020-01-01',periods=len(frame))
    cfg=dict(cfg,split='chronological',time_column='time')
    a,b,c,_,_=split_data(prepare(frame,cfg,[]),cfg)
    assert max(a)<min(b) and max(b)<min(c)
    frame.loc[len(frame)-1,cfg['features']]=frame.loc[0,cfg['features']].values
    with pytest.raises(ValueError,match='overlap'):split_data(prepare(frame,cfg,[]),cfg)


def test_imputer_fit_uses_training_only(cfg):
    X=pd.DataFrame({'age':[1.,2.,np.nan,3.],'segment':['a','b',None,'a']})
    pipe=build_pipeline(X,cfg)
    pipe.fit(X,pd.Series(['yes','no','yes','no']))
    imputer=pipe.named_steps['preprocess'].named_transformers_['c0'].named_steps['imputer']
    assert imputer.statistics_[0]==2
    pipe.predict(pd.DataFrame({'age':[1e12,np.nan],'segment':['unseen',None]}))
    assert imputer.statistics_[0]==2


@pytest.mark.parametrize('task,algorithm',[('classification','logistic'),('classification','tree'),('classification','forest'),('classification','gradient'),('regression','linear'),('regression','ridge'),('regression','tree'),('regression','forest'),('regression','gradient')])
def test_training_and_saved_inference(frame,cfg,tmp_path,task,algorithm):
    if task=='regression':frame['label']=2*frame.age+frame.signal
    cfg=dict(cfg,task=task,algorithm=algorithm)
    ops=[dict(kind='impute',column='income',method='mean'),dict(kind='trim',column='segment'),dict(kind='clip',column='income'),dict(kind='drop_duplicates')]
    path=tmp_path/f'{task}-{algorithm}.joblib'
    result=run_experiment(frame,cfg,ops,path)
    assert set(result['arms'])=={'baseline','standard','repaired'}
    assert result['split']['duplicate_overlap']==0
    assert result['arms']['repaired']['removed_training_rows']>=0
    metric='f1_macro' if task=='classification' else 'rmse'
    assert all(np.isfinite(arm['test']['metrics'][metric]) for arm in result['arms'].values())
    bundle=joblib.load(path)
    row=frame[cfg['features']].iloc[:2].replace({np.nan:None}).to_dict('records')
    response=predict(bundle,row)
    assert len(response)==2
    np.testing.assert_array_equal([r['prediction'] for r in response],bundle['pipelines']['repaired'].predict(frame[cfg['features']].iloc[:2]))
    with pytest.raises(ValueError):predict(bundle,[{'wrong':1}])
    row[0]['age']='not numeric'
    with pytest.raises(ValueError,match='numeric'):predict(bundle,row)


def test_full_experiment_reproducibility(frame,cfg,tmp_path):
    first=run_experiment(frame,cfg,[],tmp_path/'one.joblib')
    second=run_experiment(frame,cfg,[],tmp_path/'two.joblib')
    assert first['split']==second['split']
    assert first['arms']==second['arms']
    assert first['primary_comparison']['interpretation']=='unchanged'
