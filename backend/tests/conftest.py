import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import make_classification

from app.main import create_app
from app.schemas import TrainingConfig


@pytest.fixture
def frame():
    X,y=make_classification(n_samples=140,n_features=4,n_informative=3,n_redundant=0,random_state=11)
    df=pd.DataFrame(X,columns=['age','income','signal','noise'])
    df['segment']=np.where(df.age>0,'north','south')
    df.loc[::9,'income']=np.nan
    df.loc[::7,'segment']=' north '
    df['label']=np.where(y,'yes','no')
    return pd.concat([df,df.iloc[:8]],ignore_index=True)


@pytest.fixture
def cfg(frame):
    return TrainingConfig(dataset_id='a'*32,version_id='b'*32,target='label',task='classification',features=[c for c in frame if c!='label'],n_estimators=10).model_dump()


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(tmp_path/'api-storage')) as client:
        yield client
