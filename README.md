# DataDoctor AI 🩺

**Dataset debugging, human-approved repair, and evidence-based ML evaluation.**

DataDoctor is a local, single-user research application for inspecting tabular data, investigating quality warnings, reviewing reversible repairs, and measuring whether those repairs improve, preserve, or reduce predictive performance. It includes a working React dashboard, FastAPI API, SQLite history, background jobs, real scikit-learn training, saved-pipeline inference, and downloadable artifacts.

No dataset was supplied during development. Automated tests use explicitly synthetic fixtures; there are no preloaded datasets, invented metrics, or claimed findings about your data.

---

## 📋 Table of Contents
- [Why this project](#why-this-project)
- [Features](#features)
- [Architecture](#architecture)
  - [Backend Architecture](#backend-architecture-detailed)
  - [Frontend Architecture](#frontend-architecture-detailed)
- [Tech Stack](#tech-stack)
- [Installation & Setup](#install-and-run)
- [Usage Guide](#using-the-application)
- [API Documentation](#api-overview)
- [Testing](#testing)
- [Methodology](#experimental-methodology)
- [Limitations](#limits-and-future-work)

---

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

### 🏗️ Project Structure

```text
DATADoctor/
├── frontend/                       # React 19 SPA with Vite
│   ├── src/
│   │   ├── pages/                 # Route components
│   │   │   ├── Overview.jsx       # Dashboard landing page
│   │   │   ├── Datasets.jsx       # Upload & manage datasets
│   │   │   ├── Inspector.jsx      # Quality warnings viewer
│   │   │   ├── Repairs.jsx        # Data repair studio (AI-powered recommendations)
│   │   │   ├── Training.jsx       # ML model configuration
│   │   │   ├── Results.jsx        # Experiment results & metrics
│   │   │   ├── Predictions.jsx    # Single/batch inference
│   │   │   └── Research.jsx       # History, reports, methodology guide
│   │   ├── components/
│   │   │   └── UI.jsx             # Reusable UI components
│   │   ├── App.jsx                # Main app shell with navigation
│   │   ├── api.js                 # Axios/Fetch wrapper for backend
│   │   ├── context.jsx            # React Context for global state
│   │   ├── main.jsx               # App entry point
│   │   └── style.css              # Tailwind + custom styles
│   ├── tests/
│   │   └── workflow.spec.js       # End-to-end Playwright tests
│   ├── index.html
│   ├── vite.config.js             # Vite config with proxy to backend
│   └── package.json
│
├── backend/                        # FastAPI REST API
│   ├── app/
│   │   ├── main.py                # API routes, job executor, middleware
│   │   ├── schemas.py             # Pydantic models for validation
│   │   ├── store.py               # SQLite database & file storage
│   │   └── ml/                    # Machine learning core
│   │       ├── profiler.py        # Data profiling & quality diagnostics
│   │       ├── repairs.py         # Data repair transformations
│   │       └── trainer.py         # ML training, evaluation, inference
│   ├── tests/                      # Pytest unit & integration tests
│   │   ├── conftest.py
│   │   ├── test_api.py
│   │   ├── test_data.py
│   │   ├── test_ml.py
│   │   └── test_store.py
│   ├── storage/                    # Runtime data (gitignored)
│   │   ├── db.json                # SQLite metadata store
│   │   ├── datasets/              # Uploaded files & parsed versions
│   │   └── models/                # Trained model artifacts (.joblib)
│   ├── requirements.txt
│   ├── requirements-lock.txt       # Pinned dependencies
│   └── pytest.ini
│
├── scripts/
│   ├── dev.sh                     # Development server launcher (Unix)
│   └── install.sh                 # Dependency installer (Unix)
│
├── docs/                          # Documentation & research notes
│   ├── RESEARCH.md
│   ├── VALIDATION.md
│   └── *.png                      # Screenshots
│
└── README.md
```

---

### 🔧 Backend Architecture (Detailed)

The backend is built with **FastAPI** and provides a REST API for data processing, ML training, and model inference.

#### **Core Components**

##### 1️⃣ **`app/main.py`** - API Server & Job Orchestration
```python
# Key responsibilities:
# • Define all REST API endpoints (/api/datasets, /api/training, etc.)
# • Background job executor (ThreadPoolExecutor with 2 workers)
# • Request validation & error handling middleware
# • CORS configuration for frontend communication
# • File upload handling with size limits (20 MiB)

Key Features:
- Lifespan events for job recovery on restart
- Body size limiting middleware (prevents DOS)
- Custom exception handlers for 404/422/500 errors
- Formula-escaping CSV export (prevents spreadsheet injection)
- Multipart file upload with streaming validation
```

##### 2️⃣ **`app/store.py`** - Persistence Layer
```python
# Handles all data storage with hybrid approach:
# • SQLite for metadata (datasets, jobs, experiments, profiles)
# • Filesystem for large artifacts (uploaded files, parsed DataFrames, models)

Key Operations:
- save_frame(): Serialize pandas DataFrame to .pkl with versioning
- get/put/delete(): CRUD operations on JSON documents
- Job recovery: Rehydrate interrupted jobs on server restart
- Atomic writes with file locking for concurrent safety
```

##### 3️⃣ **`app/schemas.py`** - Request/Response Validation
```python
# Pydantic models for type-safe API contracts:
# • Selection: Dataset target & feature configuration
# • Operation: Data repair operations (impute, clip, drop)
# • RepairRequest: List of approved repair operations
# • TrainingConfig: ML hyperparameters, split strategy, algorithm selection

Validation Examples:
- Max 200 operations per repair request
- Train + validation + test size must leave ≥35% for training
- Algorithm compatibility checks (e.g., logistic requires classification)
```

##### 4️⃣ **`app/ml/profiler.py`** - Data Quality Analysis
```python
# Core data profiling engine that generates:

Quality Diagnostics:
- Missing value patterns (per column, per row)
- Duplicate detection (exact & normalized)
- Conflicting labels (same inputs, different targets)
- Anomaly detection (IQR + Isolation Forest)
- Data leakage warnings (ID columns, target copies)
- Cardinality, skewness, class imbalance
- Correlation matrices & histograms

Repair Recommendations:
- Automatic suggestions with alternatives
- Risk assessment for each operation
- Explanation text for each recommendation
```

##### 5️⃣ **`app/ml/repairs.py`** - Data Transformation Pipeline
```python
# Implements approved data repair operations:

Supported Operations:
- impute: Fill missing values (mean/median/mode/missing_category)
- trim: Remove whitespace & normalize text
- drop_duplicates: Remove exact duplicates (training only)
- drop_column: Exclude features
- clip: IQR-based outlier capping (fitted on training data)

Key Design Principle:
- All repairs start from immutable original dataset
- Training-fitted transformations (no data leakage)
- Version history with rollback support
```

##### 6️⃣ **`app/ml/trainer.py`** - ML Experiment Engine
```python
# Three-arm experimental design:

Arm A (Baseline):
- DummyClassifier (prior) or DummyRegressor (mean)
- Minimal preprocessing

Arm B (Standard):
- Selected algorithm with default scikit-learn preprocessing
- Imputation → Encoding → Scaling

Arm C (DataDoctor):
- Same algorithm as B, but with approved repairs applied first
- Training-fitted repair operations + standard preprocessing

Supported Algorithms:
Classification: Logistic, Decision Tree, Random Forest, Gradient Boosting
Regression: Linear, Ridge, Decision Tree, Random Forest, Gradient Boosting

Split Strategies:
- Random/Stratified (with duplicate group handling)
- Entity-grouped (prevents data leakage across entities)
- Chronological (for time-series prediction)

Evaluation Metrics:
Classification: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Confusion Matrix
Regression: MAE, RMSE, R², Residuals, Actual vs Predicted plots
Feature Importance: Permutation importance on validation set
Distribution Shift: KS test (train vs validation)
```

#### **Backend API Flow Example**

```
User uploads CSV → POST /api/datasets
                ↓
        Parse & validate (profiler.py)
                ↓
        Save to storage (store.py)
                ↓
        Submit analysis job (main.py)
                ↓
        Background worker runs profiling
                ↓
        Generate quality warnings & repair recommendations
                ↓
        Store profile results
                ↓
        Frontend polls /api/jobs/{job_id} for status
```

---

### 🎨 Frontend Architecture (Detailed)

Modern React 19 SPA with Vite for blazing-fast HMR and optimized builds.

#### **Core Technologies**

| Technology | Purpose |
|------------|---------|
| **React 19** | UI library with concurrent rendering |
| **React Router 7** | Client-side routing |
| **Vite 7** | Dev server, HMR, bundler |
| **Tailwind CSS 4** | Utility-first styling |
| **Recharts** | Data visualization (charts, ROC curves) |
| **Lucide React** | Icon library |
| **Playwright** | E2E testing |

#### **State Management**

```javascript
// Global state via React Context (context.jsx)
const DataContext = {
  datasets,         // All uploaded datasets
  dataset,          // Currently selected dataset
  profile,          // Quality analysis results
  selected,         // Active dataset ID
  jobs,             // Background job statuses
  experiments,      // Training run history
  refresh,          // Refetch all data
  experimentId,     // Selected experiment for results page
}
```

#### **Key Components**

##### 1️⃣ **`App.jsx`** - Application Shell
```javascript
// Main responsibilities:
// • Sidebar navigation with workspace switcher
// • Dataset selector dropdown
// • Health check polling (backend connectivity indicator)
// • Job status polling (auto-refresh on status changes)
// • Mobile-responsive drawer navigation
// • Error boundary for API failures

Features:
- 2-second polling interval for job updates
- Local storage persistence for selected dataset/experiment
- Fingerprinting to detect job state changes
- Responsive design with mobile menu
```

##### 2️⃣ **`pages/Datasets.jsx`** - Dataset Management
```javascript
// Features:
// • Drag & drop file upload (CSV, XLSX)
// • Multi-sheet Excel support with sheet selector
// • Dataset list with metadata (rows, columns, versions)
// • Version history viewer
// • Delete datasets with cascade cleanup
```

##### 3️⃣ **`pages/Inspector.jsx`** - Quality Diagnostics UI
```javascript
// Displays profiling results:
// • Filterable warning list (missing, duplicates, anomalies)
// • Column statistics table
// • Distribution histograms
// • Correlation heatmap
// • Click-through to affected records
```

##### 4️⃣ **`pages/Repairs.jsx`** - Interactive Repair Studio ✨
```javascript
// AI-powered repair workflow:
// • Checkbox selection for recommended operations
// • Alternative operation choices (mean vs median imputation)
// • Risk disclosure for each repair
// • Live preview of changes before applying
// • One-click restore to original version

Design Philosophy:
- Explicit approval required (no auto-apply)
- Show impact metrics (rows/columns/missing values)
- Reversible operations with version control
```

##### 5️⃣ **`pages/Training.jsx`** - ML Configuration
```javascript
// Comprehensive training setup:
// • Algorithm selection (classification/regression)
// • Feature selection with checkboxes
// • Split strategy (random/grouped/chronological)
// • Hyperparameter tuning (n_estimators, max_depth, etc.)
// • Validation preview before training
// • Real-time job progress with stage logs

Smart Defaults:
- Auto-detect numeric vs categorical features
- Suggested hyperparameters per algorithm
- Validation warnings for common mistakes
```

##### 6️⃣ **`pages/Results.jsx`** - Experiment Dashboard
```javascript
// Rich result visualization:
// • Three-arm comparison table (Baseline vs Standard vs DataDoctor)
// • Performance metrics with delta calculations
// • Confusion matrices & ROC/PR curves
// • Feature importance bar charts
// • Distribution shift diagnostics (KS tests)
// • Training logs & timing breakdown
// • Download JSON report

Interpretation Guidance:
- Color-coded performance changes (green=improvement, red=degradation)
- Statistical significance indicators
- Disclaimers for exploratory vs confirmatory results
```

##### 7️⃣ **`pages/Predictions.jsx`** - Inference Interface
```javascript
// Two inference modes:

1. Single Prediction:
   - Dynamic form based on model schema
   - Missing value handling (uses fitted imputation)
   - Unseen category support

2. Batch Prediction:
   - Upload CSV matching exact schema
   - Process up to 5,000 rows
   - Download predictions as CSV
   - Optional arm selection (baseline/standard/repaired)
```

##### 8️⃣ **`pages/Research.jsx`** - History & Methodology
```javascript
// Features:
// • Experiment comparison table
// • Download research reports (JSON)
// • Download trained models (.joblib bundles)
// • Download datasets (all versions)
// • Methodology guide (explains three-arm design)

Artifact Downloads:
- Quality reports with full profiling data
- Experiment reports with reproducibility info
- Trusted model bundles (with security warning)
```

#### **Frontend Data Flow**

```
User Action → Component Event Handler
                    ↓
            API call via api.js
                    ↓
            Backend processes request
                    ↓
            Frontend receives response
                    ↓
        Update React state/context
                    ↓
        Re-render affected components
                    ↓
    Poll /api/jobs if background job started
```

#### **Responsive Design**

```css
/* Breakpoints handled via Tailwind */
Mobile:  < 768px   - Drawer navigation, stacked layouts
Tablet:  768-1024px - Sidebar visible, 2-column grids
Desktop: > 1024px   - Full sidebar, multi-column layouts
```

---

### 🔄 Backend ↔ Frontend Communication

```
┌─────────────────┐         HTTP/JSON          ┌─────────────────┐
│   React App     │ ◄────────────────────────► │   FastAPI       │
│  (Port 5173)    │                             │  (Port 8000)    │
└─────────────────┘                             └─────────────────┘
        │                                               │
        │  GET /api/health                             │
        │  POST /api/datasets (multipart)              │
        │  GET /api/datasets/{id}/profile              │
        │  POST /api/training                          │
        │  GET /api/jobs/{id} (polling)                │
        │                                               │
        └─────────────► Vite Proxy ◄──────────────────┘
                      (Dev mode only)
```

**Production Deployment:**
- `npm run build` generates static files in `frontend/dist`
- Serve via CDN/Nginx with reverse proxy to backend
- Set `VITE_API_URL` for cross-origin requests
- Update `DATADOCTOR_CORS` with frontend origin

---

## 🛠️ Tech Stack

### Backend
- **Python 3.12** - Core language
- **FastAPI 0.141** - Modern async web framework
- **Uvicorn 0.52** - ASGI server
- **Pandas 2.2** - DataFrame operations
- **NumPy 2.3** - Numerical computing
- **scikit-learn 1.8** - Machine learning algorithms
- **SciPy 1.17** - Statistical tests
- **OpenPyXL 3.1** - Excel file parsing
- **Joblib 1.5** - Model serialization
- **Pydantic 2.14** - Data validation
- **SQLite** - Embedded database
- **pytest 8.4** - Testing framework

### Frontend
- **React 19.1** - UI library
- **Vite 7.1** - Build tool & dev server
- **React Router 7.9** - Client-side routing
- **Tailwind CSS 4.1** - Utility-first CSS
- **Recharts 3.3** - Chart library
- **Lucide React 0.548** - Icon components
- **Playwright 1.56** - E2E testing

---

## 🚀 Install and Run

### Prerequisites
- **Python 3.12** (tested: 3.12.10) - [Download](https://www.python.org/downloads/)
- **Node.js 22.12+** (tested: 24.22.1) - [Download](https://nodejs.org/)
- **npm** (comes with Node.js)

### Quick Start (All Platforms)

#### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Piyush-0665/DATADoctor.git
cd DATADoctor
```

#### 2️⃣ Install Dependencies

**Linux/macOS:**
```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install backend dependencies
python -m pip install -r backend/requirements-lock.txt

# Install frontend dependencies
cd frontend
npm ci
cd ..

# Start both servers (single command)
bash scripts/dev.sh
```

**Windows PowerShell:**
```powershell
# Option 1: Use Python launcher (recommended)
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Option 2: If py -3.12 not found, install Python 3.12 first:
# winget install Python.Python.3.12
# Then retry above commands

# Install backend dependencies
python -m pip install -r backend\requirements-lock.txt

# Install frontend dependencies
cd frontend
npm ci
cd ..

# See "Two Terminal Method" below to start services
```

> **Note for Windows:** If activation is restricted by execution policy, use `.\.venv\Scripts\python.exe` directly instead of activating.

---

### 🖥️ Starting the Application

#### Method 1: Two Terminal Method (Recommended for Windows)

**Terminal 1 - Backend:**
```bash
# From project root, with venv activated
python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

**Terminal 2 - Frontend:**
```bash
# From project root
cd frontend
npm run dev -- --port 5173 --strictPort
```

#### Method 2: Single Command (Linux/macOS)
```bash
bash scripts/dev.sh
```
> Uses Bash `wait -n` (requires Bash 4.3+). On macOS with default Bash 3.2, use the two-terminal method.

---

### ✅ Verify Installation

1. **Backend health check:**
   - Open: `http://127.0.0.1:8000/api/health`
   - Expected response: `{"status":"ok", "device":"CPU", ...}`

2. **Backend API docs:**
   - Open: `http://127.0.0.1:8000/docs`
   - Interactive Swagger UI

3. **Frontend app:**
   - Open: `http://localhost:5173`
   - Should see DataDoctor dashboard

**Stop services:** Press `Ctrl+C` in each terminal

> ⚠️ **Important:** Run only one backend instance. Multiple workers will conflict with job recovery and scheduling.

---

### ⚙️ Configuration

`backend/.env.example` documents available environment variables. They are **not automatically loaded** — export them in your shell before starting the backend.

#### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `DATADOCTOR_STORAGE` | `backend/storage` | Directory for SQLite DB, datasets, and models |
| `DATADOCTOR_CORS` | `http://localhost:5173,http://127.0.0.1:5173` | Comma-separated allowed CORS origins |
| `DATADOCTOR_API_PROXY` | `http://127.0.0.1:8000` | Backend URL for Vite dev proxy |
| `VITE_API_URL` | *(empty, same origin)* | Backend URL for production frontend build |

#### Platform-Specific Examples

**Windows PowerShell:**
```powershell
$env:DATADOCTOR_STORAGE = 'C:\DataDoctor\storage'
$env:DATADOCTOR_CORS = 'http://localhost:3000'
# Then start backend
python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

**Linux/macOS Bash:**
```bash
export DATADOCTOR_STORAGE=/path/to/storage
export DATADOCTOR_CORS=http://localhost:3000
# Then start backend
python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

#### Production Build

```bash
# Build frontend
cd frontend
npm run build  # Outputs to frontend/dist

# Preview build locally
npm run preview  # Serves static files on port 4173
```

> ⚠️ **Production Notes:**
> - `npm run preview` does **not** inherit Vite's dev proxy
> - Set `VITE_API_URL` when building for deployment
> - Add your production origin to `DATADOCTOR_CORS`
> - Or configure a reverse proxy (Nginx/Caddy) to serve both on same origin
> - **No authentication** — use locally only, not as public service

---

## 📖 Using the Application

### 1️⃣ Upload Dataset
- Go to **Datasets** page
- Drag & drop a CSV or Excel (`.xlsx`) file
- For Excel files with multiple sheets, select the target sheet
- Maximum: 20 MiB upload, 50,000 rows, 200 columns

### 2️⃣ Inspect Data Quality
- Navigate to **Quality Inspector**
- Review warnings:
  - Missing values
  - Duplicate records
  - Anomalies (IQR + Isolation Forest)
  - Data leakage risks
  - Class imbalance
- Click on warnings to see affected rows
- Filter by warning type

### 3️⃣ Configure Target & Features
- In **Target & Features** section:
  - Select prediction target column
  - Choose task type (classification/regression)
  - Select input features
  - Optional: Set identifier column (for entity grouping)

### 4️⃣ Apply Data Repairs ✨
- Go to **Repair Studio**
- Review AI-powered recommendations
- Check boxes to approve operations:
  - **Impute**: Fill missing values (mean/median/mode)
  - **Trim**: Normalize whitespace
  - **Drop duplicates**: Remove exact duplicates
  - **Drop column**: Exclude features
  - **Clip**: Cap outliers (IQR-based)
- Choose alternative methods when available
- Click **Preview Changes** to see impact
- Apply repairs (creates new version)
- Restore to original anytime

> 💡 **Tip:** All repairs start from immutable original. Previous versions remain downloadable.

### 5️⃣ Train ML Models
- Navigate to **Train & Evaluate**
- Configure experiment:
  - **Algorithm**: Logistic, Tree, Forest, Gradient Boosting, Linear, Ridge
  - **Split strategy**: Random, Entity-grouped, Chronological
  - **Hyperparameters**: n_estimators, max_depth, class_weight, etc.
  - **Validation/test sizes**: Default 20% each
- Click **Validate Configuration** to preview split sizes
- Start training (runs in background)

**Three-Arm Design:**
- **Arm A (Baseline)**: Dummy estimator (prior/mean)
- **Arm B (Standard)**: Your algorithm with default preprocessing
- **Arm C (DataDoctor)**: Same algorithm with approved repairs

### 6️⃣ View Results
- Go to **Results** page
- Select experiment from dropdown
- Compare three arms:
  - Performance metrics (accuracy, RMSE, F1, etc.)
  - Confusion matrices
  - ROC/PR curves (classification)
  - Feature importance
  - Distribution shift diagnostics (KS tests)
- Download JSON report

### 7️⃣ Make Predictions
- Navigate to **Predictions**
- **Single prediction**: Fill form with feature values
- **Batch prediction**: Upload CSV matching model schema
  - Max 5,000 rows
  - Must include all features (not target)
  - Download results as CSV
- Choose prediction arm (baseline/standard/repaired)

### 8️⃣ Review History & Download Artifacts
- Go to **History & Reports**
- View all experiments
- Download:
  - Quality reports (JSON with profiling data)
  - Experiment reports (JSON with metrics)
  - Trained models (`.joblib` bundles)
  - Dataset versions (CSV)
- Compare runs side-by-side
- Read methodology guide

---

### 🔐 Loading Exported Models (Python)

```python
# ⚠️ Only load models from your own trusted instance!
# Unpickling third-party models can execute arbitrary code.

import joblib
from app.ml.trainer import predict

# Load model bundle
bundle = joblib.load('datadoctor-12ab34cd.joblib')

# Inspect schema
print(bundle['schema'])
# {'features': ['age', 'income', ...], 'target': 'outcome', ...}

# Make predictions
records = [
    {'age': 30, 'income': 50000},
    {'age': 45, 'income': None}  # Missing values OK
]

predictions = predict(bundle, records, arm='repaired')
print(predictions)
# [{'prediction': 1, 'probability': 0.78}, ...]
```

---

## 🔬 Experimental Methodology

DataDoctor implements a rigorous **three-arm comparative design** to evaluate whether data repairs improve ML performance.

### Research Question

> **Can data-aware repair recommendations improve downstream predictive performance compared with conventional preprocessing?**

### Three-Arm Design

```
┌──────────────────────────────────────────────────────────────┐
│                    Original Dataset                          │
│              (Immutable, all arms start here)                │
└──────────────────────────────────────────────────────────────┘
                           │
           ┌───────────────┼───────────────┐
           │               │               │
           ▼               ▼               ▼
    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │  ARM A   │    │  ARM B   │    │  ARM C   │
    │ Baseline │    │ Standard │    │DataDoctor│
    └──────────┘    └──────────┘    └──────────┘
           │               │               │
           ▼               ▼               ▼
    DummyClassifier  Selected Algo   Selected Algo
    or DummyRegr.   + Std Preproc.  + Approved Repairs
                                     + Std Preproc.
           │               │               │
           └───────────────┼───────────────┘
                           │
                           ▼
                 ┌─────────────────┐
                 │  Same Test Set  │
                 │  (Held-out)     │
                 └─────────────────┘
```

#### **Arm A: Baseline**
- **Estimator**: `DummyClassifier(strategy='prior')` or `DummyRegressor(strategy='mean')`
- **Preprocessing**: Minimal (constant imputation for missing values)
- **Purpose**: Establish lower bound (prior probability or mean prediction)

#### **Arm B: Standard**
- **Estimator**: Your selected algorithm (logistic, tree, forest, gradient boosting, etc.)
- **Preprocessing**: Training-fitted conventional pipeline
  - Numeric imputation (mean/median)
  - One-hot encoding (max 20 categories per feature)
  - Standard scaling
- **Purpose**: Typical ML workflow without domain-aware repairs

#### **Arm C: DataDoctor** ✨
- **Estimator**: Same algorithm as Arm B (identical hyperparameters)
- **Preprocessing**: Approved repairs → Standard pipeline
  - **First**: Apply approved operations (drop duplicates, impute with domain method, clip outliers, exclude features)
  - **Then**: Same preprocessing as Arm B
- **Purpose**: Test whether data-aware repairs improve performance

### Key Principles

1. **All arms use identical evaluation data**
   - Same train/validation/test split
   - Same evaluation metrics
   - Same random seed for reproducibility

2. **No data leakage**
   - Imputers, scalers, and IQR thresholds fit **only** on training data
   - Test set never seen during any preprocessing/training
   - Duplicate removal affects only C's training rows (evaluation retained)

3. **Fair comparison**
   - Arms B and C use identical algorithm and hyperparameters
   - Only difference: approved repairs in C
   - If no repairs approved, B and C produce identical results

4. **Human-in-the-loop**
   - No automatic repair application
   - Each operation requires explicit approval
   - Alternative methods available (mean vs median imputation)

### Data Splitting Strategies

#### 1. Random/Stratified Split
- **When**: Standard ML problems with IID data
- **Stratification**: Preserves target distribution (classification only)
- **Duplicate handling**: Conservative grouping to prevent overlap

#### 2. Entity-Grouped Split
- **When**: Multiple rows per entity (e.g., user, customer, hospital)
- **Purpose**: Prevents entity leakage across train/test
- **Requirement**: Specify `group_column` (e.g., `user_id`)
- **Note**: Not stratified (may have unbalanced classes)

#### 3. Chronological Split
- **When**: Time-series or temporal prediction
- **Purpose**: Respects temporal ordering (train on past, test on future)
- **Requirement**: Specify `time_column` with increasing values
- **Note**: Skips OOF label review (not valid for time-based splits)

### Evaluation Metrics

#### Classification
| Metric | Description | Range |
|--------|-------------|-------|
| **Accuracy** | Correct predictions / Total | 0-1 (higher better) |
| **Precision** | True positives / Predicted positives | 0-1 (higher better) |
| **Recall** | True positives / Actual positives | 0-1 (higher better) |
| **F1 Score** | Harmonic mean of precision & recall | 0-1 (higher better) |
| **ROC-AUC** | Area under ROC curve | 0-1 (higher better) |
| **PR-AUC** | Area under Precision-Recall curve (binary) | 0-1 (higher better) |
| **Confusion Matrix** | Actual vs Predicted class distribution | Visual |

#### Regression
| Metric | Description | Range |
|--------|-------------|-------|
| **MAE** | Mean Absolute Error | 0-∞ (lower better) |
| **RMSE** | Root Mean Squared Error | 0-∞ (lower better) |
| **R²** | Coefficient of determination | -∞ to 1 (higher better) |
| **Residuals** | Predicted - Actual | Visual distribution |

#### Feature Importance
- **Method**: Permutation importance on **validation** set
- **Note**: Not measured on test set (would be peeking)
- **Purpose**: Identify influential features

#### Distribution Shift Diagnostics
- **Test**: Kolmogorov-Smirnov (train vs validation)
- **Purpose**: Detect distribution drift
- **Note**: Uncorrected p-values (exploratory only)

### What DataDoctor Does NOT Do

❌ **Automatic target correction** - Labels are never modified automatically  
❌ **Automatic outlier deletion** - Only capping (clipping) is supported  
❌ **SMOTE or oversampling** - Class imbalance handled via `class_weight`  
❌ **Test-driven hyperparameter tuning** - No automated GridSearch on test set  
❌ **Nested cross-validation** - Single train/val/test split per run  

### Out-of-Fold Label Review

For classification tasks with **stratified random splits** and **≤5,000 training rows**:
- DataDoctor trains an auxiliary logistic model on OOF folds
- Flags training samples where predicted ≠ actual label
- These are **suggestions for human review**, not automatic corrections
- Skipped for grouped/chronological splits (not valid)

### Quality Score

```
Quality Score = 0.70 × Completeness + 0.30 × Uniqueness
```

Where:
- **Completeness** = % of non-missing values
- **Uniqueness** = % of unique rows (exact duplicates)

> ⚠️ This is a **heuristic summary**, not a scientifically validated predictor of model performance.

### Interpretation Guidelines

✅ **Do:**
- Compare Arm C vs Arm B (repaired vs standard)
- Check validation metrics before test metrics
- Inspect unchanged/degraded results carefully
- Use multiple seeds for robust conclusions
- Pre-register hypotheses before final test evaluation

❌ **Don't:**
- Trust single-run results for high-stakes decisions
- Ignore validation warnings (distribution shift, class imbalance)
- Assume causation from correlation
- Apply conclusions to new datasets without validation
- Use test metrics to select hyperparameters (data leakage)

### Reproducibility

- All experiments record:
  - Exact row assignments (train/val/test IDs)
  - Random seed
  - Algorithm & hyperparameters
  - Approved repair operations
  - Dependency versions
- Download JSON reports for full reproducibility
- Load saved models with `joblib.load()` for inference

---

## 📡 API Overview

All endpoints are documented interactively at `http://127.0.0.1:8000/docs` (Swagger UI) when the backend is running.

### Base URL
- Development: `http://127.0.0.1:8000/api`
- Proxied via Vite: `http://localhost:5173/api`

### Endpoints

#### 📊 Datasets

| Method | Endpoint | Purpose | Request Body | Response |
|--------|----------|---------|--------------|----------|
| `POST` | `/api/datasets` | Upload CSV/XLSX | `multipart/form-data` (file) | Dataset metadata + job |
| `GET` | `/api/datasets` | List all datasets | - | Array of datasets |
| `GET` | `/api/datasets/{id}` | Get dataset details | - | Dataset + versions |
| `DELETE` | `/api/datasets/{id}` | Delete dataset | - | 204 No Content |
| `PUT` | `/api/datasets/{id}/selection` | Set target & features | `Selection` schema | Updated dataset |
| `GET` | `/api/datasets/{id}/records` | Paginated records | Query: `offset`, `limit`, `search`, `sort` | Records + pagination |
| `GET` | `/api/datasets/{id}/download` | Export as CSV | Query: `version_id` (optional) | CSV file |
| `GET` | `/api/datasets/{id}/source` | Download original | - | Original file |
| `GET` | `/api/datasets/{id}/report` | Quality report JSON | - | JSON report |

#### 🔍 Profiling

| Method | Endpoint | Purpose | Response |
|--------|----------|---------|----------|
| `POST` | `/api/datasets/{id}/analysis` | Trigger profiling | Job metadata |
| `GET` | `/api/datasets/{id}/profile` | Get profile results | Profile with warnings |
| `GET` | `/api/datasets/{id}/issues/{issue_id}/records` | View affected rows | Records for issue |

#### 🛠️ Repairs

| Method | Endpoint | Purpose | Request Body | Response |
|--------|----------|---------|--------------|----------|
| `POST` | `/api/datasets/{id}/repairs/preview` | Preview repairs | `RepairRequest` | Before/after stats |
| `POST` | `/api/datasets/{id}/repairs` | Apply repairs | `RepairRequest` | New version + job |
| `POST` | `/api/datasets/{id}/restore` | Restore to original | - | Updated dataset |

#### 🤖 Training

| Method | Endpoint | Purpose | Request Body | Response |
|--------|----------|---------|--------------|----------|
| `GET` | `/api/algorithms` | List algorithms | - | Available algorithms |
| `POST` | `/api/training/validate` | Validate config | `TrainingConfig` | Split sizes + validation |
| `POST` | `/api/training` | Start training | `TrainingConfig` | Job metadata |

#### 📈 Experiments

| Method | Endpoint | Purpose | Response |
|--------|----------|---------|----------|
| `GET` | `/api/experiments` | List experiments | Array of experiments |
| `GET` | `/api/experiments/{id}` | Get experiment details | Full results |
| `GET` | `/api/experiments/{id}/report` | Download JSON report | JSON file |
| `GET` | `/api/experiments/{id}/model` | Download model | `.joblib` file |

#### 🔮 Predictions

| Method | Endpoint | Purpose | Request Body | Response |
|--------|----------|---------|--------------|----------|
| `POST` | `/api/experiments/{id}/predict` | Single/batch predict | Array of records | Predictions |
| `POST` | `/api/experiments/{id}/predict-batch` | CSV batch predict | `multipart/form-data` (CSV) | CSV with predictions |

#### ⚙️ Jobs & Health

| Method | Endpoint | Purpose | Response |
|--------|----------|---------|----------|
| `GET` | `/api/health` | Health check | Status + limits |
| `GET` | `/api/jobs` | List all jobs | Array of jobs |
| `GET` | `/api/jobs/{id}` | Job status + logs | Job details |

### Request Schemas (Pydantic)

#### `Selection`
```json
{
  "target": "outcome",
  "task": "classification",
  "features": ["age", "income", "education"],
  "identifier": "user_id"
}
```

#### `Operation`
```json
{
  "kind": "impute",
  "column": "age",
  "method": "median"
}
```

#### `RepairRequest`
```json
{
  "operations": [
    {"kind": "impute", "column": "age", "method": "median"},
    {"kind": "trim", "column": "name", "method": null},
    {"kind": "drop_duplicates", "column": null, "method": null}
  ]
}
```

#### `TrainingConfig`
```json
{
  "dataset_id": "abc123",
  "version_id": "xyz789",
  "target": "outcome",
  "task": "classification",
  "features": ["age", "income"],
  "algorithm": "forest",
  "seed": 42,
  "validation_size": 0.2,
  "test_size": 0.2,
  "split": "random",
  "stratify": true,
  "n_estimators": 100,
  "max_depth": null,
  "class_weight": "none",
  "imputation": "median"
}
```

### Response Examples

#### Dataset Upload
```json
{
  "dataset": {
    "id": "abc123",
    "name": "customers.csv",
    "rows": 1500,
    "columns": ["id", "age", "income", "outcome"],
    "current_version": "v1"
  },
  "job": {
    "id": "job_789",
    "status": "queued",
    "stage": "Queued"
  }
}
```

#### Profile Results
```json
{
  "dataset_id": "abc123",
  "version_id": "v1",
  "quality_score": 87.5,
  "issues": [
    {
      "id": "missing_age",
      "severity": "high",
      "category": "missing",
      "message": "Column 'age' has 15% missing values",
      "rows": [3, 7, 12, 45, ...]
    }
  ],
  "recommendations": [
    {
      "operation": {
        "kind": "impute",
        "column": "age",
        "method": "median"
      },
      "alternatives": ["mean"],
      "explanation": "Fill missing ages with median (robust to outliers)",
      "risk": "May introduce bias if data is not MCAR"
    }
  ],
  "column_stats": { ... },
  "distributions": { ... }
}
```

#### Training Job
```json
{
  "id": "job_123",
  "kind": "training",
  "dataset_id": "abc123",
  "status": "running",
  "stage": "Fitting Arm C (DataDoctor)",
  "logs": [
    {"seconds": 0.5, "message": "Starting worker"},
    {"seconds": 1.2, "message": "Preparing data"},
    {"seconds": 3.4, "message": "Splitting data (random)"},
    {"seconds": 5.6, "message": "Fitting Arm A (Baseline)"}
  ],
  "elapsed_seconds": 5.6
}
```

#### Experiment Results
```json
{
  "id": "exp_456",
  "dataset_id": "abc123",
  "status": "completed",
  "config": { ... },
  "result": {
    "primary_comparison": {
      "baseline_accuracy": 0.72,
      "standard_accuracy": 0.85,
      "repaired_accuracy": 0.88,
      "improvement": 0.03
    },
    "arms": {
      "baseline": { "metrics": {...}, "confusion_matrix": [...] },
      "standard": { "metrics": {...}, "roc_curve": [...] },
      "repaired": { "metrics": {...}, "feature_importance": [...] }
    },
    "diagnostics": {
      "ks_tests": [...],
      "training_time_seconds": 12.3
    }
  }
}
```

### Error Responses

```json
// 404 Not Found
{
  "detail": "Dataset not found"
}

// 422 Validation Error
{
  "detail": "body.operations: ensure this value has at most 200 items"
}

// 413 Payload Too Large
{
  "detail": "Request exceeds the 21 MiB body limit"
}

// 409 Conflict
{
  "detail": "A training job is already active. Wait for it to finish."
}
```

---

## 🧪 Testing

DataDoctor includes comprehensive test coverage for both backend and frontend.

### Backend Tests (pytest)

```bash
# Activate virtual environment first
cd backend
python -m pytest -q

# Run with coverage
python -m pytest --cov=app --cov-report=html

# Run specific test file
python -m pytest tests/test_ml.py -v
```

**Test Coverage:**
- ✅ CSV/Excel parsing & validation
- ✅ Malformed upload handling
- ✅ Size & row/column limits
- ✅ Missing value analysis
- ✅ Duplicate & conflict detection
- ✅ Anomaly detection (IQR, Isolation Forest)
- ✅ Data leakage warnings
- ✅ Repair operations (impute, clip, drop)
- ✅ Version history & rollback
- ✅ Formula escaping (CSV injection prevention)
- ✅ Path traversal protection
- ✅ Train/test splitting strategies
- ✅ All algorithm families (logistic, tree, forest, gradient boosting, linear, ridge)
- ✅ Reproducibility (seed-based)
- ✅ Model serialization & inference
- ✅ Prediction validation
- ✅ Job recovery after interruption
- ✅ API error handling

### Frontend E2E Tests (Playwright)

```bash
cd frontend

# Build frontend first
npm run build

# Run E2E tests
npm run test:e2e

# Run with UI
npm run test:e2e -- --ui

# Debug mode
npm run test:e2e -- --debug
```

**Test Workflow:**
1. Starts isolated backend (port 8100) + frontend (port 5174)
2. Creates synthetic labeled dataset
3. Uploads dataset via drag & drop
4. Navigates through quality inspector
5. Applies data repairs
6. Configures and runs ML training
7. Views results and metrics
8. Makes predictions
9. Downloads artifacts
10. Cleans up services

**Browser Setup:**
```bash
# Install Chromium (if not using system browser)
npx playwright install chromium

# Or set custom browser path
$env:PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH = 'C:\Path\To\Chrome.exe'
```

> 🔐 **Isolation:** E2E tests use a temporary storage directory. No user data is accessed.

### Test Files Structure

```
backend/tests/
├── conftest.py           # Pytest fixtures & test client
├── test_api.py           # API endpoint tests
├── test_data.py          # Data parsing & validation
├── test_ml.py            # ML algorithms & evaluation
└── test_store.py         # Storage & persistence

frontend/tests/
└── workflow.spec.js      # Full workflow E2E test
```

### Running All Tests

```bash
# From project root
cd backend && python -m pytest -q && cd ../frontend && npm run build && npm run test:e2e
```

---

## 📸 Screenshots

The empty-dashboard screenshot is generated by the browser integration test. It deliberately contains no fabricated metrics.

![DataDoctor empty dashboard](docs/dashboard-empty.png)

*Mobile-responsive design with drawer navigation*

![DataDoctor mobile view](docs/dashboard-mobile.png)

---

## ⚠️ Limits and Future Work

### Current Limitations

#### System Limits
| Resource | Limit | Rationale |
|----------|-------|-----------|
| **Upload size** | 20 MiB | Local CPU processing |
| **Expanded workbook** | 100 MiB | Memory constraints |
| **Rows** | 50,000 | Interactive responsiveness |
| **Columns** | 200 | UI/UX practicality |
| **Training features** | 60 | Prevent overfitting |
| **Classes** | 50 | One-hot encoding limit |
| **Min samples per class** | 5 | Statistical validity |
| **Min labeled rows** | 30 | Model training requirement |

#### Concurrency
- **Single training job** at a time (prevents resource contention)
- **2-thread executor** (request handlers remain responsive)
- **No distributed queue** (SQLite-based job recovery)
- **No checkpoint/resume** (interrupted jobs must be resubmitted)

#### Methodology
- **No automatic feature engineering** - Use domain knowledge to create features
- **No automatic hyperparameter search** - Manual tuning or use defaults
- **No nested cross-validation** - Single train/val/test split
- **No calibration** - Probabilities are uncalibrated estimates
- **No SHAP or LIME** - Feature importance via permutation only

#### Data Quality Detection
- ❌ Cannot detect **future information leakage** - Requires domain knowledge
- ❌ Cannot detect **causal relationships** - Correlation ≠ causation
- ❌ Cannot verify **true label errors** - Only suggests candidates
- ❌ Cannot detect **semantic duplicates** - Only exact/normalized matches
- ❌ Cannot infer **entity IDs** automatically - Must specify manually

#### Security
- 🔒 **No authentication** - Local single-user only
- 🔒 **No encryption** - Data stored in plaintext SQLite
- 🔒 **No audit logs** - Actions not recorded
- 🚨 **Unpickle risk** - Only load trusted `.joblib` models

### Future Enhancements

#### Planned Features
- [ ] **Multi-user support** with authentication
- [ ] **Advanced resampling** (SMOTE, ADASYN)
- [ ] **Automatic hyperparameter tuning** (Optuna, GridSearchCV)
- [ ] **Nested cross-validation** for robust estimates
- [ ] **Confidence intervals** via bootstrap
- [ ] **Model calibration** (Platt scaling, isotonic regression)
- [ ] **PDF reports** with visualizations
- [ ] **Deep learning** support (PyTorch, TensorFlow)
- [ ] **Time-series forecasting** models
- [ ] **Natural language features** (TF-IDF, embeddings)
- [ ] **Advanced anomaly detection** (LOF, DBSCAN)
- [ ] **Fairness metrics** (disparate impact, equalized odds)

#### Research Extensions
- [ ] **Learned repair ranking** - Train a meta-model to predict repair effectiveness
- [ ] **Multi-dataset validation** - Test repairs on multiple domains
- [ ] **Causal inference** - Identify confounders and mediators
- [ ] **Active learning** - Suggest which samples to label next

### Important Disclaimers

> ⚠️ **This is a research platform for local exploration, not production inference.**

- **No real-time serving** - Models are not production-ready
- **No data encryption** - Do not use with sensitive/regulated data (PHI, PII)
- **No authentication** - Anyone with network access can use it
- **Confirmatory research requires:**
  - Additional untouched datasets
  - Pre-specified repair strategies
  - Multiple seeds/CV folds
  - Uncertainty quantification
  - Domain expert validation

> 🧪 A successful demonstration ≠ Evidence of general predictive improvement

### Reports & Artifacts

#### What DataDoctor Exports

| Artifact | Format | Contains |
|----------|--------|----------|
| **Quality report** | JSON | Profile, warnings, repair history |
| **Experiment report** | JSON | Config, metrics, row assignments |
| **Dataset version** | CSV | Repaired data (formula-escaped) |
| **Original source** | CSV/XLSX | Raw uploaded bytes (not sanitized) |
| **Trained model** | `.joblib` | Scikit-learn pipeline bundle |

> ⚠️ **Security Note:** Original source files preserve uploaded bytes **without formula sanitization**. Open untrusted spreadsheets cautiously (Excel may execute formulas).

### Model Loading Safety

```python
# ⚠️ SECURITY WARNING ⚠️
# Only load .joblib files from YOUR trusted DataDoctor instance
# Unpickling malicious models can execute arbitrary code

# Safe usage:
bundle = joblib.load('datadoctor-YOUR_RUN.joblib')

# Unsafe usage:
bundle = joblib.load('random_internet_model.joblib')  # ❌ DON'T DO THIS
```

---

## 📚 Additional Documentation

- **[Research Protocol](docs/RESEARCH.md)** - Detailed methodology and rationale
- **[Validation Boundaries](docs/VALIDATION.md)** - What's tested vs what's not
- **Methodology Guide** - In-app guide (click "How it works" in sidebar)

---

## 🤝 Contributing

This is a research demonstration project. Contributions are welcome via:
- Bug reports (open an issue)
- Feature requests (open an issue)
- Pull requests (fork → branch → PR)

### Development Setup

```bash
# Install dev dependencies
cd backend
pip install pytest pytest-cov black ruff

cd ../frontend
npm install --save-dev prettier eslint

# Format code
cd backend && black app tests
cd ../frontend && npm run format

# Lint
cd backend && ruff check app
cd ../frontend && npm run lint
```

---

## 📝 License

See [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

Built with:
- [FastAPI](https://fastapi.tiangolo.com/) - Modern Python web framework
- [React](https://react.dev/) - UI library
- [scikit-learn](https://scikit-learn.org/) - Machine learning toolkit
- [Pandas](https://pandas.pydata.org/) - Data analysis library
- [Vite](https://vitejs.dev/) - Frontend tooling
- [Tailwind CSS](https://tailwindcss.com/) - CSS framework
- [Recharts](https://recharts.org/) - Chart library

---

## 📧 Contact

- **GitHub**: [@Piyush-0665](https://github.com/Piyush-0665)
- **Repository**: [DATADoctor](https://github.com/Piyush-0665/DATADoctor)

---

<div align="center">

**Made with ❤️ for better data science**

[⬆ Back to top](#datadoctor-ai-)

</div>
