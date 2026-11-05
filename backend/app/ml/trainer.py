"""Fixed three-arm experiments. Test results never select a repair or estimator."""
import platform
import time
from importlib.metadata import version

import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error, precision_score, r2_score,
                             recall_score, roc_auc_score, roc_curve, precision_recall_curve)
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from threadpoolctl import threadpool_limits

from app.ml.repairs import FrameRepair, repair_dataset, validate_operations
from app.store import safe_json


class FeatureTypes(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.numeric_ = X.select_dtypes(include='number').columns.tolist()
        return self

    def transform(self, X):
        result = X.copy()
        for c in result:
            if c in self.numeric_:
                result[c] = pd.to_numeric(result[c], errors='coerce').replace([np.inf, -np.inf], np.nan)
            else:
                result[c] = result[c].map(lambda v: str(v) if pd.notna(v) else np.nan)
        return result


def prepare(frame, cfg, operations):
    target = cfg['target']
    if target not in frame:
        raise ValueError('Choose an existing target column.')
    features = cfg['features']
    if len(set(features)) != len(features) or any(c not in frame for c in features):
        raise ValueError('Features must be unique existing columns.')
    if len(features) > 60:
        raise ValueError('Select at most 60 features for this local training workflow.')
    if target in features:
        raise ValueError('Target cannot be an input feature.')
    for special in ('identifier', 'group_column', 'time_column'):
        if cfg.get(special) and cfg[special] not in frame:
            raise ValueError(f'Unknown {special}.')
    if cfg.get('identifier') in features:
        raise ValueError('Exclude the selected record identifier from input features.')
    if cfg['split'] == 'group' and not cfg.get('group_column'):
        raise ValueError('Group-aware splitting requires a group column.')
    if cfg['split'] == 'chronological' and not cfg.get('time_column'):
        raise ValueError('Chronological splitting requires a timestamp column.')
    validate_operations(frame, operations, target)
    mask = frame[target].notna()
    data = frame.loc[mask].copy()
    if len(data) < 30:
        raise ValueError('At least 30 rows with an observed target are needed for a three-way comparison.')
    if cfg['task'] == 'classification':
        data[target] = data[target].astype(str)
        counts = data[target].value_counts()
        if not 2 <= len(counts) <= 50 or counts.min() < 5:
            raise ValueError('Classification requires 2–50 classes with at least 5 rows per class. Check the selected task and target.')
    else:
        converted = pd.to_numeric(data[target], errors='coerce')
        if converted.isna().any() or not np.isfinite(converted).all():
            raise ValueError('Regression requires a finite numeric target.')
        if converted.nunique() < 2:
            raise ValueError('Regression target must vary.')
        data[target] = converted
    dropped = {o.get('column') for o in operations if o['kind'] == 'drop_column'}
    if not set(features) - dropped:
        raise ValueError('Approved exclusions remove every selected feature.')
    return data


def split_data(data, cfg, operations=None):
    """Union entity and identical-feature groups, preventing duplicate overlap."""
    n = len(data)
    indices = np.arange(n)
    # Normalize whitespace/case for conservative grouping only; do not change X.
    excluded = {o.get('column') for o in operations or [] if o['kind'] == 'drop_column'}
    grouping = data[[c for c in cfg['features'] if c not in excluded]].copy()
    for c in grouping.select_dtypes(exclude='number'):
        grouping[c] = grouping[c].map(lambda v: v.strip().casefold() if isinstance(v, str) else v)
    hashes = pd.util.hash_pandas_object(grouping, index=False).to_numpy()
    parent = list(range(n))

    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union_keys(values):
        seen = {}
        for i, key in enumerate(values):
            if key in seen:
                parent[root(i)] = root(seen[key])
            else:
                seen[key] = i

    union_keys(hashes)
    if cfg['split'] == 'group':
        if data[cfg['group_column']].isna().any():
            raise ValueError('Group identifiers must not be missing.')
        union_keys(data[cfg['group_column']].astype(str).tolist())
    groups = np.array([root(i) for i in indices])
    grouped = len(np.unique(groups)) < n or cfg['split'] == 'group'
    test_size, val_size, seed = cfg['test_size'], cfg['validation_size'], cfg['seed']
    y = data[cfg['target']]
    stratified = cfg['task'] == 'classification' and cfg['stratify'] and not grouped
    if cfg['split'] == 'chronological':
        times = pd.to_datetime(data[cfg['time_column']], errors='coerce', utc=True)
        if times.isna().any():
            raise ValueError('Chronological timestamps must all be parseable and non-missing.')
        order = np.argsort(times.to_numpy(), kind='stable')
        a, b = int(n*(1-test_size-val_size)), int(n*(1-test_size))
        train, val, test = order[:a], order[a:b], order[b:]
        if times.iloc[train].max() >= times.iloc[val].min() or times.iloc[val].max() >= times.iloc[test].min():
            raise ValueError('Timestamps tie across chronological boundaries. Choose another split or resolve time groups.')
        method = 'Chronological; strictly ordered timestamps'
    elif grouped:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        rest, test = next(splitter.split(indices, groups=groups))
        splitter = GroupShuffleSplit(n_splits=1, test_size=val_size/(1-test_size), random_state=seed+1)
        a, b = next(splitter.split(rest, groups=groups[rest]))
        train, val = rest[a], rest[b]
        method = 'Group shuffle of union(entity groups, normalized identical-feature groups); not stratified'
    else:
        rest, test = train_test_split(indices, test_size=test_size, random_state=seed, stratify=y if stratified else None)
        train, val = train_test_split(rest, test_size=val_size/(1-test_size), random_state=seed+1, stratify=y.iloc[rest] if stratified else None)
        method = 'Stratified random' if stratified else 'Random'
    if min(map(len, (train, val, test))) < 2:
        raise ValueError('Every partition must contain at least two rows. Check groups and split fractions.')
    for left, right in ((train,val),(train,test),(val,test)):
        if set(groups[left]) & set(groups[right]):
            raise ValueError('Duplicate feature groups overlap chronological partitions. Choose group-aware/random splitting or review duplicates.')
    if cfg['task'] == 'classification':
        if y.iloc[train].nunique() < 2:
            raise ValueError('Training partition contains fewer than two classes.')
        if set(y) - set(y.iloc[train]):
            raise ValueError('Some evaluation classes are absent from training. Revise the split or collect more data.')
    return train, val, test, groups, method


def estimator(cfg, dummy=False):
    classification = cfg['task'] == 'classification'
    if dummy:
        return DummyClassifier(strategy='prior') if classification else DummyRegressor(strategy='mean')
    algorithm, seed = cfg['algorithm'], cfg['seed']
    weight = None if cfg['class_weight'] == 'none' else cfg['class_weight']
    if algorithm == 'logistic':
        return LogisticRegression(max_iter=1000, random_state=seed, class_weight=weight)
    if algorithm == 'linear':
        return LinearRegression()
    if algorithm == 'ridge':
        return Ridge(alpha=1)
    if algorithm == 'tree':
        return DecisionTreeClassifier(max_depth=cfg['max_depth'], random_state=seed, class_weight=weight) if classification else DecisionTreeRegressor(max_depth=cfg['max_depth'], random_state=seed)
    if algorithm == 'forest':
        args = dict(n_estimators=cfg['n_estimators'], max_depth=cfg['max_depth'], random_state=seed, n_jobs=1)
        return RandomForestClassifier(**args, class_weight=weight) if classification else RandomForestRegressor(**args)
    cls = GradientBoostingClassifier if classification else GradientBoostingRegressor
    return cls(n_estimators=cfg['n_estimators'], max_depth=cfg['max_depth'] or 3, random_state=seed)


def build_pipeline(X, cfg, operations=None, dummy=False):
    operations = operations or []
    repaired = FrameRepair(operations).fit_transform(X)
    transformers = []
    overrides = {o['column']: o.get('method') for o in operations if o['kind'] == 'impute'}
    for i, c in enumerate(repaired):
        numeric = pd.api.types.is_numeric_dtype(repaired[c])
        if numeric:
            strategy = 'constant' if dummy else overrides.get(c, cfg['imputation'])
            steps = [('imputer', SimpleImputer(strategy=strategy, fill_value=0, keep_empty_features=True, add_indicator=not dummy)), ('scale', StandardScaler())]
        else:
            strategy = 'constant' if dummy or overrides.get(c) == 'missing_category' else 'most_frequent'
            steps = [('imputer', SimpleImputer(strategy=strategy, fill_value='__MISSING__', keep_empty_features=True)),
                     ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False, max_categories=20, dtype=np.float32))]
        transformers.append((f'c{i}', Pipeline(steps), [c]))
    return Pipeline([('repair', FrameRepair(operations)), ('types', FeatureTypes()),
                     ('preprocess', ColumnTransformer(transformers, sparse_threshold=0)), ('model', estimator(cfg, dummy))])


def evaluate(pipeline, X, y, task):
    predicted = pipeline.predict(X)
    if task == 'regression':
        return safe_json(dict(metrics=dict(mae=mean_absolute_error(y, predicted), rmse=np.sqrt(mean_squared_error(y, predicted)), r2=r2_score(y, predicted)),
                              predictions=[dict(row=int(i), actual=float(a), predicted=float(b), residual=float(a-b)) for i,a,b in zip(X.index[:200], y.iloc[:200], predicted[:200])]))
    classes = list(pipeline.classes_)
    probs = pipeline.predict_proba(X)
    metrics = dict(accuracy=accuracy_score(y,predicted), precision_macro=precision_score(y,predicted,labels=classes,average='macro',zero_division=0),
                   recall_macro=recall_score(y,predicted,labels=classes,average='macro',zero_division=0), f1_macro=f1_score(y,predicted,labels=classes,average='macro',zero_division=0))
    curves = {}
    if len(classes) == 2 and y.nunique() == 2:
        binary = (y == classes[1]).astype(int)
        metrics['roc_auc'] = roc_auc_score(binary, probs[:,1])
        metrics['average_precision'] = average_precision_score(binary, probs[:,1])
        fpr,tpr,_ = roc_curve(binary, probs[:,1])
        precision,recall,_ = precision_recall_curve(binary, probs[:,1])
        step = max(1, len(precision)//200)
        curves = dict(positive_class=classes[1], roc=[dict(x=float(a),y=float(b)) for a,b in zip(fpr,tpr)], pr=[dict(x=float(a),y=float(b)) for a,b in zip(recall[::step],precision[::step])])
    elif len(classes) > 2 and set(y) == set(classes):
        metrics['roc_auc_ovr_macro'] = roc_auc_score(y, probs, labels=classes, multi_class='ovr', average='macro')
    return safe_json(dict(metrics=metrics, labels=classes, confusion_matrix=confusion_matrix(y,predicted,labels=classes).tolist(), curves=curves))


def training_diagnostics(X, y, cfg, groups):
    result = dict(label_candidates=[], label_analysis='Not applicable to regression.', leakage=[], distribution_shift=[])
    for c in X.select_dtypes(include='number'):
        if cfg['task'] == 'regression':
            corr = X[c].corr(y)
            if pd.notna(corr) and abs(corr) > .98:
                result['leakage'].append(dict(column=c, evidence=f'Training-only Pearson correlation {corr:.4f}; high correlation alone does not prove leakage.'))
        if X[c].astype(str).equals(y.astype(str)):
            result['leakage'].append(dict(column=c, evidence='Training values copy the target; review availability at prediction time.'))
    if cfg['task'] == 'classification':
        counts = y.value_counts()
        if len(X) > 5000:
            result['label_analysis'] = 'Skipped: out-of-fold label review is limited to 5,000 training rows.'
        elif cfg['split'] == 'chronological' or len(set(groups)) < len(groups):
            result['label_analysis'] = 'Skipped: ordinary stratified OOF would violate group or chronological structure. Conflicting-input checks remain available.'
        elif counts.min() >= 3:
            diagnostic = dict(cfg, algorithm='logistic', class_weight='none')
            cv = StratifiedKFold(n_splits=min(5, int(counts.min())), shuffle=True, random_state=cfg['seed'])
            pipeline = build_pipeline(X, diagnostic)
            probabilities = cross_val_predict(pipeline, X, y, cv=cv, method='predict_proba', n_jobs=1)
            labels = np.sort(y.unique())
            predictions = labels[probabilities.argmax(axis=1)]
            mask = (predictions != y.to_numpy()) & (probabilities.max(axis=1) >= .8)
            result['label_candidates'] = [dict(row=int(X.index[i]), observed=str(y.iloc[i]), suggested=str(predictions[i]), probability=float(probabilities[i].max())) for i in np.where(mask)[0][:100]]
            result['label_analysis'] = f'{cv.n_splits}-fold training-only logistic OOF disagreement, probability ≥ 0.8. Suggestions are NOT verified label corrections; probabilities are uncalibrated estimates.'
        else:
            result['label_analysis'] = 'Skipped: insufficient per-class training samples for OOF analysis.'
    return result


def run_experiment(frame, cfg, operations, model_path, update=lambda stage: None):
    started = time.monotonic()
    update('Validating dataset and configuration')
    data = prepare(frame, cfg, operations)
    update('Creating fixed train, validation and held-out test partitions')
    train, val, test, groups, method = split_data(data, cfg, operations)
    X, y = data[cfg['features']], data[cfg['target']]
    train_X, train_y = X.iloc[train], y.iloc[train]
    arms, pipelines = {}, {}
    with threadpool_limits(limits=1):
        update('Training-only label and leakage diagnostics')
        diagnostics = training_diagnostics(train_X, train_y, cfg, groups[train])
        for c in train_X.select_dtypes(include='number'):
            a,b = train_X[c].dropna(), X.iloc[val][c].dropna()
            if len(a) >= 5 and len(b) >= 5:
                stat,pvalue = ks_2samp(a,b)
                diagnostics['distribution_shift'].append(dict(column=c, ks_statistic=stat, p_value=pvalue, note='Train vs validation only; uncorrected exploratory test. Small-sample power and multiple testing limit interpretation.'))
        for arm, title, ops, dummy in [('baseline','A · Baseline',[],True),('standard','B · Standard preprocessing',[],False),('repaired','C · Approved DataDoctor repairs',operations,False)]:
            update(f'Fitting {title}; preprocessing learns from training rows only')
            fit_X,fit_y = train_X,train_y
            if any(o['kind'] == 'drop_duplicates' for o in ops):
                # Dedupe exact input+target rows only. Conflicting labels are retained.
                normalized = FrameRepair([o for o in ops if o['kind'] == 'trim']).fit_transform(fit_X)
                combined = normalized.copy()
                combined[cfg['target']] = fit_y.to_numpy()
                keep = ~combined.duplicated()
                fit_X,fit_y = fit_X.loc[keep],fit_y.loc[keep]
            pipe = build_pipeline(fit_X,cfg,ops,dummy)
            pipe.fit(fit_X,fit_y)
            pipelines[arm] = pipe
            update(f'Evaluating {title} on validation partition')
            arms[arm] = dict(title=title, training_rows=len(fit_X), removed_training_rows=len(train_X)-len(fit_X),
                             features=list(pipe.named_steps['repair'].transform(fit_X).columns),
                             validation=evaluate(pipe,X.iloc[val],y.iloc[val],cfg['task']))
        update('Computing validation permutation importance')
        importance = permutation_importance(pipelines['repaired'], X.iloc[val].head(1000), y.iloc[val].head(1000),
                                           n_repeats=3, random_state=cfg['seed'], scoring='f1_macro' if cfg['task']=='classification' else 'neg_root_mean_squared_error', n_jobs=1)
        diagnostics['importance'] = sorted([dict(feature=c, mean=m, std=s) for c,m,s in zip(X.columns,importance.importances_mean,importance.importances_std)],key=lambda r:r['mean'],reverse=True)
        # No ranking/selection occurs using these test outcomes.
        update('Final held-out test evaluation of all pre-specified arms')
        for arm in arms:
            arms[arm]['test'] = evaluate(pipelines[arm],X.iloc[test],y.iloc[test],cfg['task'])
    update('Saving fitted pipelines and reproducibility evidence')
    schema = [dict(name=c, kind='numeric' if pd.api.types.is_numeric_dtype(frame[c]) else 'categorical', nullable=True,
                   categories=frame.loc[train_X.index,c].dropna().astype(str).unique()[:100].tolist() if not pd.api.types.is_numeric_dtype(frame[c]) else []) for c in cfg['features']]
    joblib.dump(dict(pipelines=pipelines,schema=schema,config=cfg), model_path)
    key = 'f1_macro' if cfg['task']=='classification' else 'rmse'
    before,after = arms['standard']['test']['metrics'][key],arms['repaired']['test']['metrics'][key]
    delta = after-before
    exported, _ = repair_dataset(frame, operations, cfg['target'])
    def quality_summary(df):
        return dict(rows=len(df), columns=len(df.columns), missing_cells=int(df.isna().sum().sum()), exact_duplicates=int(df.duplicated().sum()))
    return safe_json(dict(arms=arms, schema=schema, diagnostics=diagnostics,
                          dataset_quality=dict(original=quality_summary(frame),descriptive_repaired_export=quality_summary(exported),
                                               note='Descriptive whole-dataset export comparison only. Experiments do not learn preprocessing from this export.'),
                          split=dict(method=method,train_rows=data.index[train].tolist(),validation_rows=data.index[val].tolist(),test_rows=data.index[test].tolist(),
                                     train_count=len(train),validation_count=len(val),test_count=len(test),excluded_missing_target=len(frame)-len(data),duplicate_overlap=0),
                          primary_comparison=dict(metric=key, absolute_change=delta,relative_change_pct=100*delta/abs(before) if before else None,
                                                  interpretation=('improved' if (delta>0 if key=='f1_macro' else delta<0) else 'reduced' if delta else 'unchanged')),
                          versions={p:version(p) for p in ('scikit-learn','pandas','numpy','scipy','joblib')},python=platform.python_version(),device='CPU',elapsed_seconds=time.monotonic()-started,
                          limitations=['Single split is preliminary evidence; no general improvement claim is justified.',
                                       'A is a dummy estimator. B and C use the same selected estimator and hyperparameters. C may have explicitly approved feature exclusions or fewer training duplicates; counts are disclosed.',
                                       'Evaluation rows and targets remain unchanged; C applies its fitted feature transformations at inference.',
                                       'Exploratory whole-dataset profiling and repeated human choices can influence test results. Use a new untouched dataset to confirm conclusions.',
                                       'Model/repair selection must use validation results. Do not tune against the displayed test scores.',
                                       'Categorical encoding retains at most 20 categories per feature and groups infrequent categories. No automatic relabeling, resampling or anomaly deletion is performed.']))


def prediction_frame(records, schema):
    if not records or len(records) > 5000:
        raise ValueError('Provide between 1 and 5,000 prediction rows.')
    expected = {s['name'] for s in schema}
    for record in records:
        if set(record) != expected:
            raise ValueError(f'Prediction columns must exactly match: {", ".join(sorted(expected))}. Use null for missing values.')
    frame = pd.DataFrame(records,columns=[s['name'] for s in schema])
    for item in schema:
        c = item['name']
        if item['kind'] == 'numeric':
            result = pd.to_numeric(frame[c],errors='coerce')
            if (frame[c].notna() & result.isna()).any() or np.isinf(result.dropna()).any():
                raise ValueError(f'{c} requires finite numeric values or null.')
            frame[c] = result.astype(float)
        else:
            if frame[c].map(lambda v:isinstance(v,(dict,list))).any():
                raise ValueError(f'{c} must be a scalar category or null.')
    return frame


def predict(bundle, records, arm='repaired'):
    if arm not in bundle['pipelines']:
        raise ValueError('Unknown experiment arm.')
    X = prediction_frame(records,bundle['schema'])
    pipe = bundle['pipelines'][arm]
    values = pipe.predict(X)
    result = [dict(prediction=v) for v in values]
    if hasattr(pipe,'predict_proba'):
        probabilities = pipe.predict_proba(X)
        for r,p in zip(result,probabilities):
            r['probabilities'] = dict(zip(map(str,pipe.classes_),p))
    return safe_json(result)
