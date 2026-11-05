import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def validate_operations(frame, operations, target=None):
    for op in operations:
        kind, c = op['kind'], op.get('column')
        if kind != 'drop_duplicates' and c not in frame:
            raise ValueError(f'Unknown repair column: {c}')
        if c and c == target:
            raise ValueError('Target repair is not allowed. Correct labels only with verified domain evidence outside this workflow.')
        if kind in ('clip', 'impute') and c:
            numeric = pd.api.types.is_numeric_dtype(frame[c])
            if kind == 'clip' and not numeric:
                raise ValueError('IQR clipping requires a numeric column.')
            if kind == 'impute':
                if op.get('method') not in ('median', 'mean', 'most_frequent', 'missing_category'):
                    raise ValueError('Choose an imputation method.')
                if numeric != (op['method'] in ('median', 'mean')):
                    raise ValueError('Use mean/median for numeric data; categorical methods for other data.')
        if kind == 'trim' and pd.api.types.is_numeric_dtype(frame[c]):
            raise ValueError('Whitespace normalization requires a categorical column.')
    dropped = {o['column'] for o in operations if o['kind'] == 'drop_column'}
    if len(dropped) == len(frame.columns):
        raise ValueError('At least one column must remain.')


class FrameRepair(BaseEstimator, TransformerMixin):
    """Fit only numeric fences on training X; all transforms preserve row count."""
    def __init__(self, operations=None):
        self.operations = operations

    def fit(self, X, y=None):
        self.bounds_ = {}
        for op in self.operations or []:
            c = op.get('column')
            if op['kind'] == 'clip' and c in X:
                q1, q3 = X[c].quantile([.25, .75])
                if pd.notna(q1) and q3 > q1:
                    self.bounds_[c] = (q1 - 1.5*(q3-q1), q3 + 1.5*(q3-q1))
        return self

    def transform(self, X):
        result = X.copy()
        for op in self.operations or []:
            c = op.get('column')
            if c not in result:
                continue
            if op['kind'] == 'drop_column':
                result = result.drop(columns=c)
            elif op['kind'] == 'trim':
                result[c] = result[c].map(lambda v: v.strip() if isinstance(v, str) else v)
            elif op['kind'] == 'clip' and c in self.bounds_:
                result[c] = result[c].clip(*self.bounds_[c])
        return result


def repair_dataset(frame, operations, target=None):
    """Descriptive/export version. Its learned values are NEVER used to train models."""
    validate_operations(frame, operations, target)
    result = FrameRepair(operations).fit_transform(frame)
    changes = []
    for op in operations:
        c, kind = op.get('column'), op['kind']
        before = len(result)
        if kind == 'drop_duplicates':
            result = result.drop_duplicates()
            changes.append(dict(operation=op, removed_rows=before-len(result)))
        elif kind == 'impute' and c in result:
            s = result[c]
            if op['method'] == 'median':
                value = s.median() if s.notna().any() else 0
            elif op['method'] == 'mean':
                value = s.mean() if s.notna().any() else 0
            elif op['method'] == 'most_frequent':
                value = s.mode().iloc[0] if s.notna().any() else '__MISSING__'
            else:
                value = '__MISSING__'
            changes.append(dict(operation=op, filled_cells=int(s.isna().sum()), export_value=value))
            result[c] = s.fillna(value)
        else:
            changes.append(dict(operation=op, affected_column=c))
    return result, changes
