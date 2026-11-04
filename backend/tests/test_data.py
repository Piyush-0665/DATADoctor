import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from app.main import export_csv
from app.ml.profiler import parse_upload,profile
from app.ml.repairs import FrameRepair,repair_dataset
from app.store import Store


@pytest.mark.parametrize('filename,content',[('bad.exe',b'abc'),('empty.csv',b''),('bad.csv',b'\xff\xfe'),('bin.csv',b'hello\x00'),('header.csv',b'a,b\n'),('broken.xlsx',b'not a zip')])
def test_invalid_uploads(filename,content):
    with pytest.raises(ValueError):parse_upload(content,filename)


def test_csv_and_excel(frame):
    actual,_=parse_upload(frame.to_csv(index=False).encode(),'sample.csv')
    assert len(actual)==len(frame)
    output=io.BytesIO()
    with pd.ExcelWriter(output,engine='openpyxl') as writer:
        frame.to_excel(writer,sheet_name='data',index=False)
        pd.DataFrame({'other':[1]}).to_excel(writer,sheet_name='notes',index=False)
    parsed,sheets=parse_upload(output.getvalue(),'data.xlsx')
    assert parsed is None and sheets==['data','notes']
    parsed,_=parse_upload(output.getvalue(),'data.xlsx','data')
    assert parsed.shape==frame.shape
    with pytest.raises(ValueError):parse_upload(output.getvalue(),'data.xlsx','absent')


@pytest.mark.parametrize('content',[b'a,a\n1,2\n',b'a,\n1,2\n'])
def test_ambiguous_headers_rejected(content):
    with pytest.raises(ValueError,match='unique, nonempty'):parse_upload(content,'data.csv')


def test_size_and_expansion_limits(monkeypatch):
    import app.ml.profiler as module
    monkeypatch.setattr(module,'MAX_BYTES',5)
    with pytest.raises(ValueError,match='20 MiB'):parse_upload(b'a\n12345','x.csv')
    monkeypatch.setattr(module,'MAX_BYTES',20*1024*1024)
    monkeypatch.setattr(module,'MAX_ROWS',2)
    with pytest.raises(ValueError,match='50,000'):parse_upload(b'a\n1\n2\n3','x.csv')


def test_profile_real_evidence(frame):
    frame.loc[3,'income']=1e8
    frame['target_copy']=frame.label
    p=profile(frame,'label','classification')
    assert p['missing_total']==frame.isna().sum().sum()
    assert p['duplicate_count']==frame.duplicated().sum()
    assert any(i['category']=='anomaly' and i['column']=='income' for i in p['issues'])
    assert any(i['title']=='Feature copies the target' for i in p['issues'])
    assert any(i['category']=='consistency' for i in p['issues'])
    assert len(p['anomaly_scores'])>0
    assert not any(r['operation']['column']=='label' for r in p['recommendations'])
    assert 0<=p['quality']['score']<=100


def test_conflicting_duplicates_are_review_only(frame):
    original=frame.iloc[0].copy()
    original['label']='different'
    data=pd.concat([frame,pd.DataFrame([original])],ignore_index=True)
    p=profile(data,'label','classification')
    assert any(i['title']=='Conflicting identical feature rows' for i in p['issues'])
    repaired,_=repair_dataset(data,[dict(kind='drop_duplicates')],'label')
    assert 'different' in repaired.label.values


def test_versioning_and_immutable_original(frame,tmp_path):
    store=Store(tmp_path)
    original=store.save_frame('d'*32,frame,[],original=True)
    operations=[dict(kind='impute',column='income',method='mean'),dict(kind='trim',column='segment'),dict(kind='drop_duplicates')]
    cleaned,changes=repair_dataset(frame,operations,'label')
    version=store.save_frame('d'*32,cleaned,operations,parent=original['id'])
    assert cleaned.income.isna().sum()==0
    assert store.frame(original['id']).equals(frame)
    assert store.get('version',version['id'])['parent']==original['id']
    assert store.frame(version['id']).equals(cleaned)
    assert changes[0]['filled_cells']>0


def test_target_repair_rejected(frame):
    with pytest.raises(ValueError,match='Target repair'):repair_dataset(frame,[dict(kind='drop_column',column='label')],'label')


def test_training_fences_do_not_learn_evaluation():
    train=pd.DataFrame({'x':[1.,2.,3.,4.]})
    repair=FrameRepair([dict(kind='clip',column='x')]).fit(train)
    bounds=repair.bounds_.copy()
    result=repair.transform(pd.DataFrame({'x':[10000.,np.nan]}))
    assert repair.bounds_==bounds
    assert result.x.iloc[0]==bounds['x'][1] and len(result)==2


def test_formula_escape_and_path_protection(tmp_path):
    csv=export_csv(pd.DataFrame({'=header':['=SUM(A1)',' @evil','+formula','normal'], 'number':[-1,2,3,4]})).decode('utf-8-sig')
    assert "'=header" in csv and "'=SUM(A1)" in csv and "' @evil" in csv and "'+formula" in csv
    assert '-1' in csv
    with pytest.raises(ValueError):Store(tmp_path).path('models','../../etc/passwd','joblib')
