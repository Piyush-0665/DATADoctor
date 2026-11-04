# DataDoctor AI

**Dataset debugging, human-approved repair, and evidence-based ML evaluation.**

DataDoctor is a local, single-user research application for inspecting tabular data, investigating quality warnings, reviewing reversible repairs, and measuring whether those repairs improve, preserve, or reduce predictive performance. It includes a working React dashboard, FastAPI API, SQLite history, background jobs, real scikit-learn training, saved-pipeline inference, and downloadable artifacts.

No dataset was supplied during development. Automated tests use explicitly synthetic fixtures; there are no preloaded datasets, invented metrics, or claimed findings about your data.

## Why this project

Missing values, accidental categories, duplicates and leakage can undermine experiments. A cleaning script cannot establish that cleaning helps. DataDoctor makes the decisions inspectable and compares three fixed experiment arms under shared evaluation conditions.

The research question is: **Can data-aware repair recommendations improve downstream predictive performance compared with conventional preprocessing?** The contribution is an integrated, explainable workflow rather than a new learned repair-ranking algorithm.

## Features

- CSV and multi-sheet Excel `.xlsx` uploads; immutable source downloads and bounded, searchable, sortable previews.
- Column types, cardinality, descriptive statistics, histograms, missingness patterns, correlations, and class distributions after explicit target selection.
- Exact duplicates, case/whitespace-normalized duplicate candidates, conflicting labels, IQR and Isolation Forest anomaly candidates, identifier heuristics, constants, skewness, imbalance, and exact target-copy warnings.
- Rule-based repair explanations, alternatives, risk disclosures, explicit approval, change previews, version history and restoration. No automatic label corrections or anomaly deletion.
- Numeric mean/median imputation, categorical mode/missing-category imputation, whitespace normalization, approved feature exclusions, training-only exact deduplication and training-fitted IQR clipping.
- Classification: dummy, logistic regression, decision tree, random forest and gradient boosting. Regression: dummy, linear, ridge, tree, forest and gradient boosting.
- Random/stratified, entity-group and chronological splits; original row assignments saved for every run; conservative duplicate grouping.
- Real background stages/logs, interrupted-job recovery, SQLite experiment history, validation/test metrics, confusion matrices, binary ROC/PR curves, regression residuals, validation permutation importance and train–validation KS diagnostics.
- Training-only out-of-fold label-review suggestions where ordinary stratified folds are valid. Explicitly skipped for grouped/chronological data and >5,000 training rows.
- Individual and batch CSV prediction using the exact saved pipeline. Download CSV versions, JSON quality/repair/evaluation reports and trusted Joblib bundles.
- Responsive, accessible dashboard with empty, busy, failed, completed, and validation states, plus a beginner-friendly methodology guide.

## Architecture

```text
frontend/                  React 19 + Vite + Tailwind 4 + Router + Recharts + Lucide
  src/pages/               Overview, datasets, inspector, repairs, training,
                           results, prediction, history/reports and guide
  tests/                   Playwright browser integration tests
backend/
  app/main.py              Typed FastAPI endpoints and bounded worker executor
  app/schemas.py           Pydantic request validation
  app/store.py             SQLite metadata and controlled artifact storage
  app/ml/profiler.py        Parsing, quality diagnostics and recommendations
  app/ml/repairs.py         Export repairs and training-fitted transformations
  app/ml/trainer.py         Splits, pipelines, fitting, evaluation and inference
  tests/                   Parsing, diagnostics, ML methodology and API tests
  storage/                 Ignored local SQLite database, source files and models
scripts/dev.sh             Start both development services (macOS/Linux Bash)
docs/                      Browser-generated screenshots and research notes
```

The browser calls `/api`; Vite proxies requests to FastAPI. Pandas/NumPy handle tables, scikit-learn handles estimators and pipelines, SciPy supplies KS comparisons, OpenPyXL reads workbooks, SQLite stores metadata, and Joblib stores application-produced models. No external AI service, Redis, GPU, Docker, or credentials are required. HTTPX/pytest and Playwright are test dependencies.

## Install and run

Use **Python 3.12** (tested: 3.12.14), **Node 22.12+** (tested: 24.22.1), and npm. Run from the repository root. The Python lock records the tested transitive dependencies; the frontend has an npm lockfile.

### Linux and macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r backend/requirements-lock.txt
cd frontend
npm ci
cd ..
bash scripts/dev.sh
```

`dev.sh` uses Bash `wait -n` (Bash 4.3+). On macOS with the default Bash 3.2, use the two-terminal commands below or a current Bash.

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend/requirements-lock.txt
cd frontend
npm ci
cd ..
```

If activation is restricted, use `.\.venv\Scripts\python.exe` directly. No execution-policy change is necessary.

### Start services in two terminals (all platforms)

Backend, from the root after activating the virtual environment:

```bash
python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

Frontend, in a second terminal:

```bash
cd frontend
npm run dev -- --port 5173 --strictPort
```

Open `http://localhost:5173` in your own browser. API documentation is at `http://127.0.0.1:8000/docs` and health at `/api/health`. Stop services with Ctrl+C. Start one backend process: multiple workers would conflict with local job recovery and scheduling.

### Configuration

`backend/.env.example` documents configuration; it is **not automatically loaded**. Export variables in the shell that starts the backend:

| Variable | Default | Purpose |
|---|---|---|
| `DATADOCTOR_STORAGE` | `backend/storage` | SQLite, parsed versions, immutable sources and models |
| `DATADOCTOR_CORS` | `http://localhost:5173,http://127.0.0.1:5173` | Comma-separated explicit allowed browser origins |
| `DATADOCTOR_API_PROXY` | `http://127.0.0.1:8000` | Vite development proxy destination |
| `VITE_API_URL` | empty (same origin) | Optional backend origin for a separately served production frontend |

For PowerShell: `$env:DATADOCTOR_STORAGE = 'C:\DataDoctor\storage'`. For Bash: `export DATADOCTOR_STORAGE=/path/to/storage`. Never commit dataset files or local credentials.

`npm run build` produces `frontend/dist`. `npm run preview` is a static build preview and does not inherit Vite's development proxy: set `VITE_API_URL` when building and add its origin to `DATADOCTOR_CORS`, or configure a same-origin reverse proxy for `/api`. This project has no authentication; use it locally, not as a public multi-user service.

## Using the application

1. **Datasets:** drop a UTF-8 CSV or `.xlsx`; choose a worksheet when prompted. Analysis runs in the background.
2. Inspect records and column profiles. In **Target & features**, explicitly select the target, classification/regression, input features and optional record identifier; save the selection. Without a target, profiling and repairs still work.
3. **Quality inspector:** filter warnings, inspect affected rows, review missing patterns, anomaly scores and distributions. Rows are candidates for investigation, not proven errors.
4. **Repair studio:** check each approved recommendation, choose alternatives, preview the changes, then apply that exact plan. All plans start from the immutable original; previous versions remain downloadable. To start a different plan from original recommendations, restore the original version first.
5. **Train & evaluate:** choose the algorithm, features, split strategy, hyperparameters, seed and fractions. Validate and inspect the exact configuration, then start real three-arm training. Use entity grouping when rows share an entity and chronological splits for time-dependent prediction. Exclude unavailable/future or target-derived inputs with domain knowledge.
6. Follow actual worker stages. **Results** separates validation and test outcomes and discloses changes in training data. Inspect poor or unchanged results as carefully as improvements.
7. **Predictions:** enter schema-derived inputs or upload an exact-schema CSV (without the target). Missing fields in the form use fitted imputation; unseen categories are supported. The batch API requires every expected column and at most 5,000 rows.
8. **History & reports:** compare runs, download JSON research reports, datasets and trusted model bundles. Dataset deletion removes its local versions, jobs, reports and models after active jobs finish. Files otherwise remain until deletion; local filesystem backups are your responsibility.

### Loading your own exported model

Only load Joblib from your trusted instance, with the recorded dependency versions. Unpickling third-party models can execute code. The API does not accept model uploads.

```python
# Run with backend on PYTHONPATH so app.ml classes can be imported.
import joblib
from app.ml.trainer import predict

bundle = joblib.load('datadoctor-YOUR_RUN.joblib')
print(bundle['schema'])
# records must exactly match schema names; use None for missing inputs.
predictions = predict(bundle, records, arm='repaired')
```

## Experimental methodology

**A — Baseline:** dummy prior classifier or mean regressor with minimal constant imputation/encoding. **B — Standard:** selected estimator with training-fitted conventional imputation, encoding and numeric scaling. **C — DataDoctor:** the same estimator/hyperparameters with the approved operations, followed by training-fitted preprocessing. With no approved operations B and C are identical and should produce identical metrics.

All arms start from the original parsed data, excluding the same missing-target rows. A repaired export is never used as model-training input. Split assignments and evaluation targets are identical across arms. Approved duplicate removal affects only C's training rows; evaluation records are retained. Feature exclusions and training row counts are disclosed. C's fitted feature transformations run at inference too. Target imputation/relabeling, automatic outlier deletion and SMOTE are not implemented.

Random splitting unions normalized identical-feature groups (after approved exclusions) with user-selected entity groups. This conservatively prevents exact/case-whitespace duplicate overlap. Group splits are not stratified; inadequate class coverage fails with an error. Chronological splitting requires strictly increasing partition boundaries, rejects duplicate overlap and skips ordinary OOF label analysis. It does not infer an entity key or timestamp automatically.

Categorical imputation handles unseen values and preserves empty training columns. One-hot encoding retains at most 20 categories per feature and groups infrequent categories. Numeric imputers, scalers and IQR thresholds fit only on training rows. Optional deduplication preserves identical-input rows with conflicting targets. A label disagreement from training-only OOF logistic predictions is a review suggestion, never a verified correction.

Classification reports accuracy, macro precision/recall/F1, confusion matrices and ROC-AUC when class coverage allows it. Binary runs also report average precision and ROC/PR curves with the positive class named. Regression reports MAE, RMSE, R², actual/predicted values and residuals. Importance is measured by validation permutation, not test labels. KS tests compare training with validation only; uncorrected p-values are exploratory.

The fixed arms are evaluated on held-out test data after fitting and validation; there is no automatic test-driven selection. Use validation to investigate and pre-register choices before a final experiment. Full-dataset exploratory profiling and repeated human inspection can contaminate confirmatory claims even when transformations are fit safely. Confirm conclusions on a new untouched dataset. Repeated-seed runs can be compared in history; automated nested CV and confidence intervals are future work.

**Quality score:** `0.70 × completeness + 0.30 × exact-record uniqueness`, both in percent. This is a heuristic summary of two properties, not a scientifically validated measure of predictive quality. Validity, consistency, anomalies, label candidates and leakage warnings are reported as review evidence, not verified error rates.

## API overview

Interactive schemas and validation details are available in `/docs`.

| Endpoint | Purpose |
|---|---|
| `POST /api/datasets` | Multipart CSV/XLSX upload; `?sheet=` selects workbook sheet |
| `GET /api/datasets`, `GET/DELETE /api/datasets/{id}` | Dataset metadata and retention |
| `PUT /api/datasets/{id}/selection` | Explicit target, task, features and identifier |
| `GET /api/datasets/{id}/records` | `offset`, `limit`, `search`, `sort`, `descending`, `version_id` |
| `POST /api/datasets/{id}/analysis` | Schedule actual profiling |
| `GET /api/datasets/{id}/profile` | Warnings, feature stats and repair recommendations |
| `GET /api/datasets/{id}/issues/{issue_id}/records` | Bounded affected-record inspection |
| `POST /api/datasets/{id}/repairs/preview` | Preview approved operation list |
| `POST /api/datasets/{id}/repairs`, `POST /api/datasets/{id}/restore` | Versioned repair/restore |
| `GET /api/datasets/{id}/download`, `/source`, `/report` | CSV version, immutable source, quality/repair JSON |
| `GET /api/jobs`, `GET /api/jobs/{id}` | Persisted stages, logs, status and failures |
| `GET /api/algorithms`, `POST /api/training/validate`, `POST /api/training` | Configure and launch experiments |
| `GET /api/experiments`, `GET /api/experiments/{id}` | History and detailed measured results |
| `GET /api/experiments/{id}/report`, `/model` | Research JSON and trusted pipelines |
| `POST /api/experiments/{id}/predict`, `/predict-batch` | JSON rows / multipart CSV; optional `?arm=` |

## Testing

From an activated virtual environment:

```bash
cd backend
python -m pytest -q
cd ../frontend
npm run build
npm run test:e2e
```

Browser tests use system Chromium on Linux when available; otherwise run `npx playwright install chromium`, or set `PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH`. They start isolated services on ports 8100/5174 with a separate temporary storage directory, create a labelled synthetic fixture, exercise the real dashboard/API/worker/model pipeline, and shut down those services. No user data is touched. Generated traces and reports are ignored.

Backend tests cover CSV/Excel selection, malformed uploads, limits, missingness, duplicates/conflicts, anomalies, leakage warnings, reversible versions, formula escaping, path/ownership restrictions, train-only imputation/clipping, group/time splitting, every supported estimator family, reproducibility, saved inference, prediction validation, API exports and deletion, and interrupted-job recovery.

See [verified checks and explicit testing boundaries](docs/VALIDATION.md) and the [research protocol](docs/RESEARCH.md).

## Screenshots

The empty-dashboard screenshot is generated by the browser integration test. It deliberately contains no fabricated metrics.

![DataDoctor empty dashboard](docs/dashboard-empty.png)

## Limits and future work

- Local single-user CPU application: 20 MiB upload, 100 MiB expanded workbook, 50,000 rows, 200 columns; training limits input features to 60, classes to 50, and each class needs at least 5 samples. At least 30 labelled rows are required. Statistical diagnostics are bounded and their limits are disclosed.
- The two-thread background executor keeps request handlers responsive; only one training run is allowed at a time. No distributed queue or checkpoint/resume; interrupted jobs become failed on restart and must be resubmitted.
- Future-information leakage, domain validity, causal relationships, true label errors, and fuzzy semantic duplicates cannot be established generically. IDs and groups must be selected using domain knowledge.
- No automatic repair search, learned ranking, nested CV, calibration, PDF reports, advanced resampling, authentication, or deep learning. Optional advanced techniques are used only where valid; other limitations are displayed explicitly.
- Reports are machine-readable JSON and spreadsheet-safe CSV exports. Original source downloads preserve uploaded bytes and are not formula-sanitized; open untrusted original spreadsheets cautiously. Models are trusted local artifacts only.
- Confirmatory research requires additional untouched datasets, pre-specified repairs, repeated seeds/CV respecting entities and time, and uncertainty analysis. A successful demonstration is not evidence of general predictive improvement.
