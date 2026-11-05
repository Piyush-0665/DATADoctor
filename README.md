# DataDoctor AI

**Dataset debugging, human-approved repair, and evidence-based ML evaluation.**

DataDoctor is a local, single-user research application for inspecting tabular data, investigating quality warnings, reviewing reversible repairs, and measuring whether those repairs improve, preserve, or reduce predictive performance.

## Features

- CSV and Excel `.xlsx` uploads with data profiling
- Quality warnings: missing values, duplicates, anomalies, data leakage detection
- AI-powered repair recommendations with human approval
- Three-arm ML experiment design (Baseline vs Standard vs Repaired)
- Multiple algorithms: Logistic, Tree, Forest, Gradient Boosting, Linear, Ridge
- Interactive dashboard with results visualization
- Model export and batch predictions

## Tech Stack

**Backend:** Python 3.12, FastAPI, Pandas, scikit-learn, SQLite  
**Frontend:** React 19, Vite, Tailwind CSS, Recharts

## Quick Start

### Prerequisites
- Python 3.12
- Node.js 22+
- npm

### Installation & Run

**Windows:**
```powershell
# Backend
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend\requirements-lock.txt

# Frontend
cd frontend
npm ci
cd ..
```

**Linux/macOS:**
```bash
# Backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r backend/requirements-lock.txt

# Frontend
cd frontend
npm ci
cd ..
```

### Start Services

**Terminal 1 - Backend:**
```bash
python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev -- --port 5173 --strictPort
```

### Access Application

- **Frontend:** http://localhost:5173
- **Backend API Docs:** http://127.0.0.1:8000/docs
- **Health Check:** http://127.0.0.1:8000/api/health

Stop with `Ctrl+C` in each terminal.

## Usage

1. **Upload** CSV or Excel dataset (max 20 MiB, 50K rows, 200 columns)
2. **Inspect** quality warnings and data statistics
3. **Select** target variable and features
4. **Review & approve** data repair recommendations
5. **Configure** ML algorithm and hyperparameters
6. **Train** three-arm experiment (Baseline, Standard, Repaired)
7. **Compare** results and download models/reports
8. **Predict** on new data using trained models

## Architecture

```
frontend/          React dashboard with Vite
  src/pages/       8 route components
  tests/           Playwright E2E tests

backend/
  app/main.py      FastAPI endpoints
  app/ml/          ML modules (profiling, repairs, training)
  tests/           Pytest unit tests
  storage/         SQLite + file storage
```

## Three-Arm Methodology

- **Arm A (Baseline):** Dummy estimator (prior/mean)
- **Arm B (Standard):** Selected algorithm + standard preprocessing
- **Arm C (DataDoctor):** Same algorithm + approved repairs + preprocessing

All arms use identical train/test splits for fair comparison.

## Testing

**Backend:**
```bash
cd backend
python -m pytest -q
```

**Frontend:**
```bash
cd frontend
npm run build
npm run test:e2e
```

## Limits

- 20 MiB upload, 50K rows, 200 columns max
- Local single-user only (no authentication)
- One training job at a time
- Requires Python 3.12 specifically

## Documentation

- [Research Protocol](docs/RESEARCH.md)
- [Validation Details](docs/VALIDATION.md)
- API docs at `/docs` when backend is running

## License

See [LICENSE](LICENSE) file.

## Contact

- GitHub: [@Piyush-0665](https://github.com/Piyush-0665)
- Repository: [DATADoctor](https://github.com/Piyush-0665/DATADoctor)
