"""Exploratory diagnostics, never interpreted as verified data errors."""
import csv
import io
import zipfile

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer

from app.store import safe_json

MAX_ROWS = 50_000
MAX_COLUMNS = 200
MAX_BYTES = 20 * 1024 * 1024


def parse_upload(content, filename, sheet=None):
    suffix = filename.lower().rsplit('.', 1)[-1]
    if not content or len(content) > MAX_BYTES:
        raise ValueError('Upload a nonempty file no larger than 20 MiB.')
    sheets = []
    try:
        if suffix == 'csv':
            if b'\x00' in content:
                raise ValueError('CSV contains binary/null bytes. Export a UTF-8 CSV instead.')
            try:
                text = content.decode('utf-8-sig')
            except UnicodeDecodeError as exc:
                raise ValueError('CSV must use UTF-8 encoding. Re-export the file as UTF-8 CSV.') from exc
            header = next(csv.reader(io.StringIO(text)), [])
            if not header or any(not name.strip() for name in header) or len(set(header)) != len(header):
                raise ValueError('CSV must have unique, nonempty column names.')
            if len(header) > MAX_COLUMNS:
                raise ValueError('Local application limit: 200 columns.')
            frame = pd.read_csv(io.StringIO(text), nrows=MAX_ROWS + 1)
        elif suffix == 'xlsx':
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                if sum(i.file_size for i in archive.infolist()) > 100 * 1024 * 1024:
                    raise ValueError('Expanded workbook exceeds the 100 MiB safety limit.')
            book = pd.ExcelFile(io.BytesIO(content), engine='openpyxl')
            sheets = book.sheet_names
            if sheet is not None and sheet not in sheets:
                raise ValueError('Selected worksheet does not exist.')
            if sheet is None and len(sheets) > 1:
                return None, sheets
            worksheet = book.book[sheet or sheets[0]]
            if worksheet.max_column and worksheet.max_column > MAX_COLUMNS:
                raise ValueError('Local application limit: 200 columns per worksheet.')
            frame = pd.read_excel(book, sheet_name=sheet or sheets[0], nrows=MAX_ROWS + 1)
        else:
            raise ValueError('Supported file types are .csv and .xlsx. Legacy .xls is not supported.')
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f'Unable to parse {suffix.upper()} file. Check its format and delimiters.') from exc
    if frame.empty or not len(frame.columns):
        raise ValueError('Dataset must contain a header and at least one data row.')
    if len(frame) > MAX_ROWS or len(frame.columns) > MAX_COLUMNS:
        raise ValueError('Local application limit: 50,000 rows and 200 columns.')
    frame.columns = [str(c) for c in frame.columns]
    if len(set(frame.columns)) != len(frame.columns):
        raise ValueError('Column names must be unique.')
    # Infinity cannot be represented as a finite feature; expose it as missing.
    frame = frame.replace([np.inf, -np.inf], np.nan).reset_index(drop=True)
    return frame, sheets


def profile(frame, target=None, task=None):
    rows, width = frame.shape
    issues, recommendations, columns = [], [], []

    def issue(kind, title, column, count, method, evidence, action, severity='review', indices=None):
        issues.append(dict(id=f'{kind}:{column or "records"}:{len(issues)}', category=kind, title=title,
                           column=column, count=int(count), technique=method, evidence=evidence,
                           action=action, severity=severity, rows=list(indices or [])[:100]))

    def recommend(kind, column, reason, risk, method=None, alternatives=None):
        recommendations.append(dict(id=f'{kind}:{column or "records"}', operation=dict(kind=kind, column=column, method=method),
                                    title=f'{kind.replace("_", " ").title()}' + (f' · {column}' if column else ''),
                                    reason=reason, benefit='More consistent model inputs; improvement must be measured on validation data.',
                                    risk=risk, alternatives=alternatives or [], reversible=True))

    numeric = [c for c in frame.select_dtypes(include='number').columns if c != target]
    for c in frame.columns:
        s = frame[c]
        num = pd.api.types.is_numeric_dtype(s)
        missing, unique = int(s.isna().sum()), int(s.nunique())
        info = dict(name=c, dtype=str(s.dtype), kind='numeric' if num else 'categorical', missing=missing,
                    missing_pct=round(100 * missing / rows, 2), unique=unique,
                    examples=s.dropna().astype(str).head(5).tolist())
        counts = s.value_counts(dropna=True).head(12)
        info['distribution'] = [dict(value=str(k), count=int(v)) for k, v in counts.items()]
        if num:
            info['statistics'] = safe_json(s.describe().to_dict())
            if s.notna().any():
                try:
                    counts_hist, bins = np.histogram(s.dropna(), bins=min(15, max(1, unique)))
                    info['histogram'] = [dict(value=f'{bins[i]:.3g}–{bins[i+1]:.3g}', count=int(v)) for i, v in enumerate(counts_hist)]
                except (ValueError, OverflowError, IndexError):
                    info['histogram_note'] = 'Numeric range is too extreme for a stable histogram; review units and values.'
        columns.append(info)
        if missing:
            issue('missing', 'Missing observations', c, missing, 'Null count', f'{missing / rows:.1%} missing; {info["kind"]} feature.',
                  'Review missingness; fit imputation on training data only.', 'high' if missing / rows > .4 else 'review', s.index[s.isna()].tolist())
            if c != target:
                recommend('impute', c, 'Median is robust to extremes.' if num else 'An explicit missing category preserves missingness.',
                          'Imputation adds assumptions and may hide informative missingness.', 'median' if num else 'missing_category',
                          ['median', 'mean'] if num else ['missing_category', 'most_frequent'])
        if unique <= 1:
            issue('distribution', 'Constant or empty feature', c, rows, 'Distinct-value count', f'{unique} non-null distinct values.', 'Consider excluding after review.')
            if c != target:
                recommend('drop_column', c, 'A constant feature carries no observed variation.', 'Future unseen variation will be unavailable to the model.')
        elif len(s.dropna()) and s.value_counts(normalize=True).iloc[0] >= .98:
            issue('distribution', 'Near-constant feature', c, rows, 'Dominant-value frequency', 'At least 98% of observed values are identical.', 'Review whether rare values matter.')
        if not num:
            dirty = s.notna() & s.astype(str).ne(s.astype(str).str.strip())
            if dirty.any():
                issue('consistency', 'Surrounding whitespace', c, dirty.sum(), 'String comparison', 'Some strings change when leading/trailing whitespace is stripped.', 'Preview normalization before approving.', indices=s.index[dirty].tolist())
                if c != target:
                    recommend('trim', c, 'Leading/trailing whitespace can create accidental categories.', 'Distinct intentional values may merge. Case is preserved.')
            if unique > 50 and unique / rows > .5:
                issue('distribution', 'High-cardinality categories', c, unique, 'Unique/row ratio', f'{unique} categories; {unique / rows:.1%} of row count.', 'Review identifiers and sparse category support.')
        if c != target and unique / rows > .95 and (not num or any(t in c.lower() for t in ('id', 'index', 'key'))):
            issue('leakage', 'Possible record identifier', c, unique, 'Cardinality and name heuristic', 'Nearly unique values may identify records rather than generalize.', 'Exclude only with domain confirmation.')
            recommend('drop_column', c, 'Potential identifier; confirm its meaning before excluding.', 'Unique continuous or meaningful features can still be predictive.')
        if num and c != target and unique > 4:
            q1, q3 = s.quantile([.25, .75])
            iqr = q3 - q1
            mask = ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)) if iqr > 0 else pd.Series(False, index=s.index)
            if mask.any():
                issue('anomaly', 'Univariate outlier candidates', c, mask.sum(), '1.5 × IQR fences', f'Exploratory fences: {q1 - 1.5 * iqr:.4g} to {q3 + 1.5 * iqr:.4g}.', 'Inspect records. Extremes are not necessarily errors.', indices=s.index[mask].tolist())
                recommend('clip', c, 'Optional training-fitted IQR clipping limits extreme leverage without deleting rows.', 'May erase real tail behavior; approve only after review.')
            if abs(s.skew()) > 2:
                issue('distribution', 'Skewed numeric distribution', c, s.notna().sum(), 'Sample skewness', f'Sample skewness {s.skew():.3g}.', 'Inspect the histogram; skewness alone is not an error.')

    duplicates = frame.duplicated(keep='first')
    if duplicates.any():
        issue('duplicates', 'Exact duplicate records', None, duplicates.sum(), 'All-column equality', 'Counts exclude the first occurrence in each identical group.', 'Duplicate feature groups are kept in the same random split.', indices=frame.index[duplicates].tolist())
        recommend('drop_duplicates', None, 'Remove repeated identical records from training after splitting; evaluation rows are retained.', 'Repeated observations may be intentional. Conflicting labels are never silently removed.')
    normalized = frame.copy()
    for c in frame.select_dtypes(exclude='number').columns:
        normalized[c] = frame[c].map(lambda v: v.strip().casefold() if isinstance(v, str) else v)
    near = normalized.duplicated(keep=False) & ~frame.duplicated(keep=False)
    if near.any():
        issue('duplicates', 'Normalized duplicate candidates', None, near.sum(), 'Whitespace/case-normalized equality', 'Records become equal only after string normalization; this is not fuzzy matching.', 'Review manually; no automatic deletion.', indices=frame.index[near].tolist())

    anomaly_scores = []
    usable = [c for c in numeric if frame[c].notna().any()][:40]
    if rows >= 20 and usable:
        x = SimpleImputer(strategy='median').fit_transform(frame[usable])
        detector = IsolationForest(n_estimators=60, random_state=42, n_jobs=1, contamination='auto')
        detector.fit(x[:10000])
        scores = detector.decision_function(x)
        order = np.argsort(scores)[:min(100, rows)]
        anomaly_scores = [dict(row=int(i), score=float(scores[i]), flagged=bool(scores[i] < 0)) for i in order]
        issue('anomaly', 'Multivariate anomaly candidates', None, (scores < 0).sum(), 'Isolation Forest',
              f'Exploratory detector fitted on first {min(rows, 10000)} rows and up to 40 numeric features. Lower scores are more unusual.',
              'Inspect; score < 0 is a model threshold, not evidence of a data error.', indices=[int(i) for i in order if scores[i] < 0])

    target_summary = None
    if target in frame.columns:
        y = frame[target]
        target_summary = dict(name=target, missing=int(y.isna().sum()), unique=int(y.nunique()),
                              distribution=[dict(value=str(k), count=int(v)) for k, v in y.value_counts().head(30).items()])
        if task == 'classification' and y.nunique() > 1:
            ratio = y.value_counts().max() / y.value_counts().min()
            if ratio >= 3:
                issue('distribution', 'Class imbalance', target, y.notna().sum(), 'Largest/smallest class ratio', f'Class ratio: {ratio:.2f}.', 'Consider macro F1, average precision and training-only class weights.')
        features = [c for c in frame if c != target]
        if features:
            groups = frame.groupby(features, dropna=False, observed=True)[target].transform('nunique')
            conflicts = groups > 1
            if conflicts.any():
                issue('labels', 'Conflicting identical feature rows', target, conflicts.sum(), 'Feature-group target disagreement', 'Identical inputs have different observed targets.', 'Verify labels with domain evidence. No automatic relabeling.', indices=frame.index[conflicts].tolist())
        for c in features:
            if frame[c].astype(str).equals(y.astype(str)):
                issue('leakage', 'Feature copies the target', c, rows, 'Exact target equality', 'Values equal the selected target; possible target-derived feature.', 'Confirm availability at inference and explicitly exclude if invalid.', 'high')
                recommend('drop_column', c, 'This feature copies the selected target.', 'Confirm domain semantics; correlation alone is not used for automatic exclusion.')

    missing_total = int(frame.isna().sum().sum())
    completeness = 100 * (1 - missing_total / (rows * width))
    uniqueness = 100 * (1 - duplicates.sum() / rows)
    patterns = frame.isna().apply(lambda row: ', '.join(frame.columns[row]) or 'Complete', axis=1).value_counts().head(10)
    corr_columns = numeric[:30]
    correlations = []
    if len(corr_columns) >= 2:
        corr = frame[corr_columns].corr()
        correlations = [dict(x=a, y=b, value=safe_json(corr.loc[a, b])) for i, a in enumerate(corr_columns) for b in corr_columns[i+1:] if pd.notna(corr.loc[a,b])]
        correlations.sort(key=lambda a: abs(a['value']), reverse=True)
    # Dedupe recommendations that have overlapping evidence.
    recommendations = list({r['id']: r for r in recommendations}.values())
    return safe_json(dict(rows=rows, columns=columns, preview=frame.head(50).to_dict('records'),
                          issues=issues, recommendations=recommendations, anomaly_scores=anomaly_scores,
                          duplicate_count=int(duplicates.sum()), missing_total=missing_total, target=target_summary,
                          correlations=correlations[:100], missing_patterns=[dict(pattern=k, count=int(v)) for k,v in patterns.items()],
                          quality=dict(score=round(.7 * completeness + .3 * uniqueness, 1), completeness=round(completeness, 1), uniqueness=round(uniqueness, 1),
                                       formula='0.70 × completeness + 0.30 × exact-record uniqueness (percent).',
                                       limitation='Heuristic coverage score, not validated predictive quality. Validity, consistency, label and leakage warnings are review counts, not verified error rates.'),
                          limitations=['Whole-dataset diagnostics are exploratory. Selecting repairs after inspecting all data weakens confirmatory claims; use a new untouched dataset for confirmation.',
                                       'Label-model disagreement and target association diagnostics are computed on training data only during experiments.',
                                       'Domain validity, future-information leakage and entity semantics require human review. No generic detector proves these.']))
