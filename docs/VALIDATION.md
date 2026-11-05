# Verified development behavior

Verified on 4 November 2026 in the Linux cloud workspace using Python 3.12.14, Node 24.22.1 and system Chromium. Tests use synthetic fixtures only. No actual user dataset was supplied or evaluated.

| Check | Observed outcome |
|---|---|
| `cd backend && ../.venv/bin/python -m pytest -q` | **37 passed**, one upstream Starlette/HTTPX deprecation warning |
| `cd frontend && npm run build` | **Passed**, production assets generated and split into application, React and chart bundles |
| `cd frontend && npm run test:e2e` | **2 passed** using real FastAPI, SQLite, Vite, Chromium and actual scikit-learn training |
| `bash scripts/install.sh` | **Passed** on repeat invocation, Python lock installation, dependency check, npm frozen install and frontend build |
| `bash scripts/dev.sh` | **Started** backend and frontend in the documented working directory |
| Real HTTP requests through Vite | `/api/health`, `/api/datasets`, `/api/experiments` returned **200**; root returned the correct application HTML |
| Normal application storage | Dataset and experiment lists are empty; no test fixture was inserted into the user's workspace |
| `python -m pip check` | **No broken requirements** |
| `git diff --check`, `bash -n scripts/install.sh scripts/dev.sh` | **Passed** |

Backend coverage includes parsing CSV and multi-sheet Excel; invalid files, ambiguous headers, size limits and oversized requests; missingness, duplicate conflicts, anomaly candidates and target-copy warnings; export formula escaping; artifact path and dataset-version ownership validation; immutable originals and repair versions; fixed splits with duplicate/entity-group protection and chronological rejection of overlap; training-only imputation and clipping; all four classification and all five regression estimator choices; identical-arm reproducibility; exported pipeline inference; prediction validation; complete classification/regression API workflows; report/download/delete behavior; interrupted-job recovery; and SQLite record-type isolation/migration.

The browser workflow exercises upload, explicit target selection, diagnosis, human approval, preview/application, training validation, actual job completion, real comparison metrics, saved-pipeline prediction and report download. A second test covers mobile navigation, invalid-upload messages and horizontal overflow. Browser screenshots in this directory are generated from that run; the empty dashboard has no example metrics.

## Boundaries

- macOS/Windows installation commands are documented but have not been executed on those operating systems.
- Browser end-to-end training covers classification. Regression is executed through backend integration tests, including inference and artifact export.
- No performance, fairness, validity or repair-improvement claim about a real research dataset is established.
- Large files at every documented upper bound, all possible domain-specific schemas, multiple concurrent users, and Firefox/Safari are unverified.
- Cloud `install_script` and `start_skill` were saved as a configuration draft. Publication and restoration into a fresh cloud task have not been tested; the user must review/save settings and publish the environment.
