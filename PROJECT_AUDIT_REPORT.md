# DataForge — Project Audit Report (Step 1)

**Audit date:** 2026-10-24 · **Scope:** inspection and diagnostics only. No application code, database rows, or data files were changed.
**Evidence labels:** **[V]** verified by running or reading the code in this audit · **[A]** assumption or inference · **[U]** unverified.

---

## 1. Executive summary

DataForge is a two-part app: a FastAPI/SQLite backend with a **real, hand-written NumPy VAE**, and a React/TypeScript frontend. The two halves are **not connected**.

| Area | State |
|---|---|
| Backend pipeline (upload → preprocess → train → generate → evaluate → export → history) | **Works end to end [V]** on a fresh upload, run against an isolated copy of the backend. All 12 router routes plus `/api/health` responded correctly. |
| Backend with the data committed in the repo | **Broken [V].** The DB stores absolute paths from another machine (`C:\Users\divya\...`). Preprocess, train, and export fail on the existing records. Generation from the existing models also fails because the preprocessor file is looked up in the wrong folder. |
| Frontend | **Builds [V] but is a mock-driven prototype.** Training, generation, quality charts, and export are simulated in the browser. Only a health ping and a CSV upload touch the backend, and the upload result is mishandled. `tsc` reports 1 error, so `npm run build` fails [V]. |
| Tests | `test_backend_api.py` is a manual script, not a pytest suite. It needs a live server on port 8000, and it writes data permanently. It stopped at Test 4 here, caused by the path problem above. |
| Model quality | Weak [V]. Fresh 20-epoch models score about 70.7% fidelity with a mean KS statistic of about 0.39. Every numeric column is rated "Failed" by the evaluator's own thresholds. KL is about 0.0004, which points to posterior collapse. |
| Documentation | Substantially out of date. It claims every endpoint passes, PyTorch, a 1000-row dataset, and 13 endpoints with different paths. |

**Biggest risks:** (1) the cross-machine absolute paths and the split `data/` folders (P0/P1); (2) a frontend that looks complete but is simulated, so the core workflow cannot run from the UI (P1); (3) the VAE collapses to roughly the mean, so generated data is poor (P1/P2); (4) git hygiene: `node_modules`, `.env`, the DB, and weights are committed (P0/P3).

---

## 2. Actual project directory structure

```text
dataforge/                         (git repo, branch main, 1 commit; no .gitignore)
├── BACKEND_DEVELOPMENT.md, DEVELOPMENT_SETUP.md, FRONTEND_DEVELOPMENT.md
├── PROJECT_REQUIREMENTS_DOCUMENT.md, SYSTEM_DESIGN_DOCUMENT.md
├── data/                          EMPTY subfolders: uploads/ models/ synthetic/   <-- the folders the API code uses
├── backend/
│   ├── .env                       (tracked; keys: PROJECT_NAME, ENVIRONMENT, HOST, PORT, DATABASE_URL, UPLOAD_DIR, MODEL_DIR, SYNTHETIC_DIR, CORS_ORIGINS. Values not read into this report. Nothing in app/ loads it [V])
│   ├── .venv/                     UNTRACKED, created today 10:31 (not by this audit; provenance unclear). Contains fastapi, numpy, pandas, scipy, scikit-learn. Does not contain pytest or requests.
│   ├── dataforge.db               (tracked) SQLite: 2 datasets, 2 models, 2 experiments, 2 evaluations
│   ├── requirements.txt           8 unpinned (>=) packages
│   ├── test_backend_api.py        manual integration script (requests-based)
│   ├── __pycache__/ + app/**/__pycache__/   (tracked .pyc, cpython-312 and cpython-315)
│   ├── app/
│   │   ├── main.py                FastAPI app, CORS, /, /api/health, DB init and seeding
│   │   ├── api/endpoints.py       11 router routes, prefix /api
│   │   ├── core/database.py       sqlite3 helpers, 4 tables
│   │   ├── models/vae.py          NumPy VAE (encoder/decoder, Adam, hand-derived backprop)
│   │   ├── schemas/api_models.py  Pydantic models
│   │   └── services/{preprocessor,evaluator}.py
│   └── data/                      <-- the folders the committed DB points at
│       ├── uploads/   student_performance.csv (250 rows), ds_5acee684_test_upload.csv (3 rows)
│       ├── models/    preprocessor_ds_student_01.json, vae_5e5e011c_weights.npz, vae_56639f23_weights.npz
│       └── synthetic/ 2 generated CSVs
└── frontend/
    ├── .env                       VITE_API_BASE_URL (empty value), VITE_APP_TITLE (empty value)
    ├── dist/                      (tracked) stale build output
    ├── node_modules/              (tracked: about 9,995 files)
    ├── package.json, tsconfig.json, vite.config.ts, tailwind/postcss configs
    └── src/ App.tsx, main.tsx, index.css
        ├── components/  Navbar.tsx, Sidebar.tsx, DataGenerator.tsx (orphan)
        ├── pages/       Dashboard, DatasetStudio, ModelLab, DataGenerator, QualityComparison, ExportHistory
        ├── services/    api.ts, mockData.ts
        └── types/index.ts
```

**Generated or non-source content [V]:** `dist/`, `node_modules/`, `__pycache__/`, `dataforge.db`, `*.npz`, `data/**`, `backend/.venv/` (untracked).
**Not present [V]:** a root `data/` with content (it is empty), pytest tests, ESLint config, frontend tests, a `.gitignore`, `backend/app/__init__.py` files (the packages work as namespace packages).

---

## 3. Technology stack and application entry points

| Layer | Actual [V] | Notes |
|---|---|---|
| Backend | Python, FastAPI, Uvicorn, Pydantic v2, sqlite3 (stdlib, no ORM), pandas, NumPy, SciPy | `scikit-learn` is in `requirements.txt` but **never imported** [V: grep]. |
| ML | **NumPy-only VAE** (`backend/app/models/vae.py`) | The docs say PyTorch/TensorFlow. |
| Frontend | React 18.3, TypeScript, Vite 5, Tailwind 3, **Recharts**, lucide-react | The task brief says Chart.js. `package.json` has **no Chart.js**, no router, and no HTTP client. |
| Navigation | `useState` tab switch in `App.tsx`; **no React Router** | The docs list URL routes (`/dashboard`, `/generator`, …) that do not exist. |
| Tests | `requests`-based script | **No pytest tests** and no `pytest` in requirements. A stale `*-pytest-9.1.1.pyc` is committed. |

**Entry points**
- Backend: `backend/app/main.py` → `app.main:app` (imports as `app.*`, so it must be launched from `backend/`).
- Frontend: `frontend/index.html` → `src/main.tsx` → `App.tsx`.

---

## 4. Backend API audit

The router has **11 routes**, defined in `backend/app/api/endpoints.py`. `main.py` adds `GET /api/health` and `GET /`. Status below comes from calls against an isolated copy of the backend (fresh DB and data folders), plus read-only calls against the real DB.

| # | Method + route | Function (endpoints.py) | Input → output | Status | Evidence, risks |
|---|---|---|---|---|---|
| 0 | `GET /api/health` (main.py) | `health_check` | → `{status, message, version, deep_learning_engine}` | **Complete [V]** | The engine string is a hardcoded literal and does not check anything. `GET /` is the same. |
| 1 | `POST /api/datasets/upload` | `upload_dataset` | multipart CSV → `DatasetResponse` | **Complete [V]** | `.txt` is rejected with 400. `uploadDate` is the literal `"Just now"`. The extension check is case-sensitive (`.CSV` is rejected). There is no size limit, although the SDD says 25 MB. The same-named file is saved under a new id each time. The `id`-column heuristic in `inspect_csv` is `'id' in col.lower()`, which also matches columns like `paid`, `midterm`, or `provider` [A]. |
| 2 | `GET /api/datasets` | `get_datasets` | → list with a 15-row preview | **Complete, with caveat [V]** | The preview is silently empty when the file path is missing. This happens for every record in the committed DB. |
| 3 | `GET /api/datasets/{id}` | `get_dataset_details` | → dataset with a 50-row preview | **Complete, with caveat [V]** | The SDD documents `limit/offset` query params. They are not implemented. |
| 4 | `POST /api/preprocess` | `preprocess_dataset` | `{dataset_id, columns[]}` → dims, feature names | **Works on fresh data [V]; broken on committed data [V]** | Returned 11 dims on a fresh upload. On the committed DB it fails with a 500 `FileNotFoundError` (`pd.read_csv` at line 132 sits outside the `try`). It saves `preprocessor_<ds>.json`. Train re-fits from the stored schema and **ignores** any custom column config sent here. |
| 5 | `POST /api/models/train` | `train_model` | hyperparameters → full result | **Works [V]**, with design problems | Training is **synchronous**, so the request blocks until it finishes. 20 epochs took 0.2 s. The status is always `completed` when the response arrives, and the status route never shows `training`. There is no stop route. `hidden_layers` with fewer than 2 entries → **500** [V: `[64]` and `[]`], because the VAE hardcodes two hidden layers. A failure after `insert_model` leaves a model row stuck as `initialized` with no cleanup. `activation` is not supported. |
| 6 | `GET /api/models/status/{id}` | `get_model_status` | → `ModelStatusResponse` | **Complete [V]** | The SDD path is `/models/{id}/status`. |
| 7 | `GET /api/models` | `get_models` | → list | **Complete [V]** | Not in the original PRD table but implemented. |
| 8 | `POST /api/generate` | `generate_records` | `{model_id, dataset_id, num_records, temperature, random_seed}` → preview and fidelity | **Works on fresh data [V]; broken on committed data [V]** | 200 records produced and reproducible with the same seed [V]. On the committed models it returns **400 "Preprocessor pipeline not found"** [V], because the code looks in root `data/models/` and the file is in `backend/data/models/`. It does not check that `model.dataset_id == req.dataset_id`. It uses the global `np.random.seed`, which is not thread-safe [A]. It rebuilds the VAE with the default `beta` (harmless for inference). |
| 9 | `GET /api/evaluate/{exp_id}` | `get_evaluation_metrics` | → metrics | **Works [V]** | Evaluation is computed and stored at generation time. The fallback `0.04` for the average KS is a hardcoded placeholder. The SDD documents `POST /api/evaluate`. |
| 10 | `GET /api/export/{exp_id}` | `download_synthetic_csv` | → `text/csv` | **Works on fresh data [V]; broken on committed data [V]** | 404 for `gen_8e9beadc`, because the stored path is on the other machine. |
| 11 | `GET /api/experiments` | `get_experiment_history` | → list with dataset and model names | **Works [V]** | Response model `created_at: str`. No pagination. |

**Unimplemented items from the SDD [V: absent in code]:** `/datasets/{id}/preprocess` path style, `/models/{id}/stop`, `/generate/{id}/preview`, `POST /evaluate`, `/history`, `/export/{id}/csv`, `/export/{id}/report`.

**Cross-cutting backend issues**
- No authentication or authorization. CORS is `allow_origins=["*"]` combined with `allow_credentials=True`. `.env` values (including `CORS_ORIGINS`) are **never loaded** [V].
- SQLite foreign keys are not enforced: there is no `PRAGMA foreign_keys=ON`, so `ON DELETE CASCADE` does nothing [V: pragma returned 0]. No delete routes exist anyway.
- Pydantic v2 deprecations: `.dict()` is used at `endpoints.py:133` and `:181`.
- Each helper opens and closes its own connection, with no context managers or WAL mode.

---

## 5. Machine-learning pipeline audit

| Stage | Real or placeholder | Evidence |
|---|---|---|
| 1. Dataset loading and validation | **Real** | `DataPreprocessor.inspect_csv` infers types, nulls, and stats. Validation is minimal: any parseable `.csv` is accepted, with no size, row-count, or empty-file guard. |
| 2. Preprocessing | **Real** | Median/mean/mode imputation, min-max scaling, one-hot encoding, and `save`/`load` to JSON. The `drop` imputation strategy is accepted by the schema but not implemented (it falls through to median). Rare categories are not handled. |
| 3. Training | **Real** | Hand-derived backprop, Adam, gradient clipping, ELBO. Loss falls from 0.233 to 0.133 over 20 epochs [V]. |
| 4. Persistence and loading | **Real, fragile** | Weights go to `.npz` and architecture params to the DB. Reload needs `latent_dim` and `hidden_layers` from `params_json` plus the preprocessor JSON. The stored `weights_path` is absolute. |
| 5. Generation | **Real** | `z ~ N(0, τ²I)` → decoder → inverse transform. Numerics are rounded to 1 decimal, so integer columns come back as floats. Categorical output is the argmax of sigmoid outputs, so temperature does not reduce categorical diversity [A]. |
| 6. Evaluation | **Mostly real** | KS, Wasserstein, a correlation Frobenius delta, and domain validity. Placeholder defaults: correlation delta `0.04` when fewer than 2 numeric columns, and average KS `0.05` / `0.04` when none. Categorical fidelity is only a "category in real set" check, which a one-hot argmax output always passes, so it is nearly meaningless [A]. |
| 7. History and persistence | **Real** | `experiments` and `evaluations` rows are written at generation time. |

**Model quality finding [V]:** a fresh 20-epoch model gave `kl ≈ 0.0004` and a fidelity score of 70.7%. The mean KS was 0.392, with every numeric column between 0.376 and 0.417, so all are "Failed" (≥ 0.25). The committed models gave the same result (74.8% and 74.9%, mean KS 0.41). The doc summary calls this an "initial" score. Cause **[A, not diagnosed]:** posterior collapse. The decoder is barely using z, and the MSE plateau of about 0.133 is consistent with predicting column means. Checking this was out of scope.

**Hardcoded or mock values in the UI path** are listed in section 7.

---

## 6. Database and file-storage audit

**Schema (`backend/app/core/database.py`)** [V]: `datasets`, `models`, `experiments`, `evaluations`. Tables are created automatically at import time (`init_db()` in `main.py`). Metadata survives restarts because it lives in the SQLite file.

**Current contents (read-only inspection):**

| Table | Rows | Notes |
|---|---|---|
| datasets | 2 | `ds_student_01` (250 rows), `ds_5acee684` (3 rows, from Test 11 of the test script) |
| models | 2 | Both for `ds_student_01`, completed |
| experiments | 2 | `gen_8e9beadc`, `gen_772407a6` |
| evaluations | 2 | Match the experiments |

**P0 finding: path portability [V]**
1. `DB_PATH` is derived from `__file__`, so the DB itself is found regardless of working directory. Good. It resolves to `backend/dataforge.db`.
2. Every file path **stored in the DB is absolute and belongs to another machine**: `C:\Users\divya\OneDrive\Desktop\dl project\backend\data\...` (datasets, model weights, and experiment outputs).
3. `endpoints.py:32` computes `BASE_DIR` with **three** `..` from `api/`, which gives the repo root, so new files go to **root `data/`**. But `main.py:14` seeds from **`backend/data/uploads/`**, and the committed files are in `backend/data/`. Result: two data trees, with the root `data/` empty.
4. Consequence [V]: with the committed DB, `GET /datasets/*` returns empty previews, and preprocess, train, and export fail. Generation fails because `preprocessor_ds_student_01.json` is in `backend/data/models/` while the code looks in `data/models/`.
5. The seeding block in `main.py` only runs when `ds_student_01` is absent. It is skipped here, so the stale path is never refreshed.

**Other findings**
- No enforced foreign keys [V]. IDs are consistent across the workflow in the isolated run: dataset id → model id → `gen_` id → evaluation key [V].
- `INSERT OR REPLACE` is used throughout. With an `evaluations.id AUTOINCREMENT`, a replace changes the row id [A, minor].
- No migrations and no schema version.
- Uploaded filenames are stored as `{id}_{original}`. The original name is not sanitized for path separators [A].
- Nothing is cleaned up on failed training, and there are no delete routes.
- Running the app writes to tracked files (`dataforge.db`, `data/`, `.pyc`), which pollutes `git status`.

---

## 7. Frontend page and component audit

Navigation is `useState` in `App.tsx`. **All state starts from `mockData.ts` and is never loaded from the backend** (no `getDatasets`, `getExperiments`, or `getModels` call on load).

| File | Purpose | Backend integration | Findings |
|---|---|---|---|
| `App.tsx` | Shell, global state | Health ping only | `apiBaseUrl` comes from `VITE_API_BASE_URL`, which is **empty in `.env`**, so the `||` fallback is used. Initial state = mock dataset (`id ds_student_01`, claims **1000 rows**; real DB has **250**), mock models, and mock experiments. |
| `components/Sidebar.tsx`, `Navbar.tsx` | Navigation, status | `onRefresh` → health | Fine. Sidebar shows the base URL. |
| `pages/Dashboard.tsx` | KPIs | None | Computed from mock state. |
| `pages/DatasetStudio.tsx` | Upload and profile | `uploadDataset` | Parses the CSV in the browser (naive `split(',')`, no quoted-field handling, 200 rows, `id` heuristic `includes('id')`). **Bug [V, from `tsc`]:** `.then(apiDataset => apiDataset.id)` treats the `ApiResponse` wrapper as the dataset (TS2339, `DatasetStudio.tsx:50`). It is always `undefined`. Because `request()` never throws, the `.catch` error display **cannot fire**. The upload result, backend schema, and column edits are never sent to `/preprocess`. |
| `pages/ModelLab.tsx` | Training UI | **None** | Training is a `setInterval` **simulation** with a formula-based loss curve (`exp(-epoch/15)` + noise). No backend call. The model ID is `vae_${Date.now()}`, not a backend id. The UI's hidden-layer choices are fixed two-layer options (`128, 64` / `256, 128` / `64, 32`), epochs max 200 (backend allows 500), and the `activation` choice (LeakyReLU/ReLU/ELU) has no backend support. |
| `pages/DataGenerator.tsx` | Generation UI | **None** | `setTimeout` mock. Rows are drawn from per-column mean/std in the browser, **capped at 50 rows**. The fidelity score is `92.5 + Math.random()*3.5`. Validity `99.8` and KS `0.041` are hardcoded. |
| `components/DataGenerator.tsx` | A second generator that calls `generateSynthetic` | Yes | **Orphan [V]:** not imported anywhere. Dataset and model IDs are typed in as free text. Shows errors and loading state. Never rendered. |
| `pages/QualityComparison.tsx` | Fidelity charts | **None** | `getDistributionData` returns **hardcoded bins** (same chart for every feature); `statComparisons` is imported from `mockData`; fallbacks `93.8` and `99.8`. `selectedFeature` defaults to `math_score`. |
| `pages/ExportHistory.tsx` | History and export | **None** | CSV is built **client-side** from at most 50 preview rows (not the backend export). The JSON report is built client-side, with a hardcoded "Distributions match at 95% confidence (p > 0.05)" conclusion regardless of results. Subject text `AI23531` is hardcoded. |
| `services/api.ts` | API client | — | URLs are **hardcoded `http://127.0.0.1:8000`** in every function (ignores `VITE_API_BASE_URL` and the Vite proxy). `credentials: 'include'` is combined with a wildcard CORS origin on the server. See the contract mismatches below. |
| `services/mockData.ts`, `types/index.ts` | Mock data, types | — | Types are camelCase UI shapes that do not match backend responses (`DatasetInfo.name/data` vs `filename/preview`; `ModelRecord` vs `ModelStatusResponse`; `ExperimentRun.records` vs `preview_records`). |

**Frontend/backend contract mismatches [V by reading both sides]**
1. `startTraining` sends `{dataset_id, hyperparameters:{...}}`. The backend expects flat fields (`latent_dim`, `hidden_layers`, `learning_rate`, …). The extra key is ignored and **defaults would be used silently**.
2. `uploadDataset` consumer reads `.id` from the wrapper (see above).
3. `hiddenLayers` is a string (`"128, 64"`) in the UI and a `List[int]` in the API. `betaWeight`/`learningRate` vs `beta_weight`/`learning_rate`, `latentDim` vs `latent_dim` all differ.
4. No API function exists for: `GET /models`, `POST /preprocess`, or the dataset detail preview. `getDataset` exists but is unused.
5. `downloadExport` returns a Blob but nothing calls it.
6. The doc'd/mock `uploadDate` is a formatted string; the API returns `"Just now"` or a UTC timestamp.

**UX gaps:** no loading or error states anywhere except `components/DataGenerator.tsx`. No empty states (always mock data). No form validation. The backend-offline state is shown but does not change behavior.

---

## 8. Testing and dependency status

### Checks that actually ran

| # | Check | Result |
|---|---|---|
| 1 | `test_backend_api.py` against the **real** DB (server on port 8001; port 8000 is occupied by an unrelated "CreditWise AI" app, so a patched copy of the script pointed at 8001 was used) | **Tests 1–3 passed. Test 4 failed** (`POST /api/preprocess` → 500, `FileNotFoundError` on the `divya` path). Tests 5–11 did not run. |
| 2 | Full workflow on an **isolated copy** of `backend/` (fresh DB in the scratchpad, port 8002) | Upload, preprocess, train (20 epochs), status, models, generate, evaluate, export, and experiments all returned 200 with sensible payloads. Seed reproducibility passed. Validation errors returned 400/404/422 as expected. `hidden_layers=[64]` and `[]` returned **500**. |
| 3 | Read-only probes of the real DB via the real server | `GET /models`, `/datasets/ds_student_01`, `/evaluate/gen_8e9beadc`, `/experiments` → 200. `GET /export/gen_8e9beadc` → **404**. `POST /generate` with a committed model → **400**. |
| 4 | `npx tsc --noEmit` in `frontend/` | **1 error:** `src/pages/DatasetStudio.tsx(50,63): TS2339 Property 'id' does not exist on type 'ApiResponse<any>'`. Hence **`npm run build` (`tsc && vite build`) fails** [V by inference from the tsc exit code 2]. |
| 5 | `npx vite build --outDir <scratchpad>` (skips `tsc`; the committed `dist/` was not touched) | **Succeeds** (2,384 modules; JS 604.7 kB, with a chunk-size warning). The committed `dist/` has different asset hashes, so it is stale relative to the source. |
| 6 | Python import and version check | `fastapi 0.162.0, uvicorn 0.54.0, numpy 2.5.2, pandas 3.0.5, scipy 1.18.0, sklearn 1.9.0, pydantic 2.14.1` (Python 3.12). |

### Not run
- **pytest:** not installed anywhere, and the repo has no pytest tests. The script's only function is `run_tests()`, which pytest would not collect (name does not start with `test_`) [A].
- The frontend dev server (`npm run dev`) and a browser check of the UI. No browser automation was used [U].
- No frontend tests exist.

### Test script facts [V by reading]
- Needs a running server at the hardcoded `http://127.0.0.1:8000/api`. Run `cd backend; python test_backend_api.py` with the server already up.
- **Modifies persistent data on every run:** it trains and persists a model, generates files and DB rows, and Test 11 uploads a new dataset every time.
- Test 2 picks `datasets[0]`, which is the *newest* dataset, not `ds_student_01`. A new upload shifts which dataset later tests use [A].
- Hardcoded final banner "ALL 11 … PASSING" prints regardless of anything beyond the asserts.

### Dependency status
- `requirements.txt` is missing `requests` (used by the test script) and `pytest`. It lists `scikit-learn`, which is unused. No upper bounds or pins.
- `frontend/package.json` has no `test` or `lint` script, no router, no Chart.js, and no HTTP library.
- Packages needed to run the backend from `requirements.txt` were **not installed** before this audit. In this session they were installed with `pip install -r requirements.txt requests` for Python 3.12 (see Section 11 for the side effects).
- The docs say Python 3.15 and Node 20.18; this machine has Python 3.12 / 3.11 and Node 24.21.

---

## 9. Documentation discrepancies

| Document | Claim | Reality |
|---|---|---|
| BACKEND_DEVELOPMENT §status | "All 11 Endpoints Passing", with a pasted log showing `ds_student_01`, 1 dataset, and absolute success | The log was produced on another machine. Reproduced here: fails at Test 4. The log also shows "Found 1 dataset", but the DB now has 2. |
| BACKEND_DEVELOPMENT §2 | 12 rows in the endpoint table, labelled "11" | 12 rows including health; the router has 11, plus `/api/health` and `/`. |
| BACKEND_DEVELOPMENT §2 | `GET /datasets/{id}` takes limit/offset | Not implemented. |
| SYSTEM_DESIGN §3 | 13 endpoints with paths such as `/datasets/{id}/preprocess`, `/models/{id}/status`, `/models/{id}/stop`, `/generate/{id}/preview`, `POST /evaluate`, `/history`, `/export/{id}/csv`, `/export/{id}/report`; numeric IDs | Different paths and string IDs (`ds_…`, `vae_…`, `gen_…`). No stop, report, or preview routes. |
| SYSTEM_DESIGN, PRD | PyTorch / TensorFlow VAE; SQLAlchemy | NumPy VAE; raw `sqlite3`. |
| SYSTEM_DESIGN §3 | Training is polled while running (`status: training`) | Training is synchronous; the response is the final result. |
| PRD / SDD | Upload limit 25 MB; Retail Sales sample dataset | Not enforced / not present. |
| FRONTEND_DEVELOPMENT | "Developed, Tested & Operational"; routes `/dashboard`, `/dataset-studio`, `/model-lab`, …; "Live Interactive Training Engine"; "Instant Record Synthesis Engine" | No routing. Training and synthesis are simulated in the browser (Section 7). |
| FRONTEND_DEVELOPMENT §Module 2 | "Sortable" preview | No sort logic found in `DatasetStudio.tsx` (grep for `sort` returned nothing) [V]; UI behavior itself not exercised [U]. |
| FRONTEND_DEVELOPMENT §4 | "✓ 2383 modules … 604.25 kB" build | Current source builds 2,384 modules / 604.68 kB, and `tsc` now fails. |
| FRONTEND_DEVELOPMENT §Module 5 | Correlation heatmap; Recharts distribution charts from real data | A correlation-delta section exists (`QualityComparison.tsx:150-157`), but the Frobenius norm text (`0.041`) is hardcoded, and the distribution chart uses hardcoded bins [V]. |
| DEVELOPMENT_SETUP | Python 3.15, pandas 3.0.6, Node 20.18, `c:\Users\divya\...` paths; `.env` drives `UPLOAD_DIR`, `CORS_ORIGINS`, `PORT` | Different machine. `.env` is not read by the backend. |
| DEVELOPMENT_SETUP §2 | `backend/app/__init__.py`, `api/`, `core/` | No `__init__.py` files exist. |
| Task brief | Chart.js, pytest, TypeScript `DataGenerator.tsx` + `QualityComparison.tsx` + Navbar + Sidebar | Recharts, no pytest tests. Those components exist; the brief's description of a "basic routing structure" does not match (no router). |

---

## 10. Bugs, incomplete features, and integration risks

### P0 — prevents startup or threatens existing data
| ID | Finding | Location |
|---|---|---|
| P0-1 | The DB stores absolute paths from another machine; every file operation on committed records fails. | `database.py` (`file_path`, `weights_path`, `output_file_path`); rows in `backend/dataforge.db` |
| P0-2 | Two data roots: API writes/reads root `data/`, while seeding and committed files use `backend/data/`. | `endpoints.py:32-35` vs `main.py:14` |
| P0-3 | `node_modules/` (about 9,995 files), `dist/`, `.env` files, `dataforge.db`, and `*.npz` are all committed, with no `.gitignore`. Running the app dirties tracked binaries (7 `.pyc` files were modified by this audit's runs). | repo root |
| P0-4 | The tests mutate persistent data (new model, generations, upload on every run). | `test_backend_api.py` |
| P0-5 | The default port 8000 can collide (it was occupied here by another app). | `main.py:66`, `api.ts`, `vite.config.ts` |

### P1 — prevents the core dataset → generation workflow
| ID | Finding | Location |
|---|---|---|
| P1-1 | The frontend never uses the backend for training, generation, quality, or export (all simulated). | `ModelLab.tsx`, `pages/DataGenerator.tsx`, `QualityComparison.tsx`, `ExportHistory.tsx` |
| P1-2 | The upload response is mishandled (`.id` on the wrapper), which fails `tsc` and so `npm run build`. | `DatasetStudio.tsx:50` |
| P1-3 | The train request shape is wrong; hyperparameters are silently ignored. | `api.ts:startTraining` vs `TrainModelRequest` |
| P1-4 | The `/generate` lookup of the preprocessor file is path-dependent (see P0-2). | `endpoints.py:275` |
| P1-5 | Training is synchronous, with no progress, no stop, and no failure state. A long request will time out in the browser. | `train_model` |
| P1-6 | The model produces poor, near-mean output (fidelity about 71%, KS about 0.39, KL about 0.0004). | `vae.py`, `preprocessor.py` |
| P1-7 | `hidden_layers` of length ≠ 2 → HTTP 500 via the API (the current UI only offers two-layer options, so this affects direct API use and any future custom-layer UI). | `vae.py:__init__`, `TrainModelRequest` |

### P2 — missing pages, integrations, and important functionality
| ID | Finding |
|---|---|
| P2-1 | No frontend loading of datasets, models, or experiments from the API; state is mock-seeded and lost on refresh. |
| P2-2 | No mapping layer between backend (snake_case) and frontend (camelCase) shapes. |
| P2-3 | The frontend's CSV export uses 50 preview rows instead of `GET /api/export/{id}`. The JSON audit report is generated client-side, with a hardcoded conclusion. |
| P2-4 | `QualityComparison` uses hardcoded bins. The backend returns no histogram or correlation matrix data, so a new endpoint or payload is needed. |
| P2-5 | The preprocess column config (types, imputation) is never persisted per dataset, so training ignores user edits. |
| P2-6 | `components/DataGenerator.tsx` is an orphan duplicate of `pages/DataGenerator.tsx`. |
| P2-7 | CORS wildcard with credentials. `.env` is unused. No upload size limit, no dataset or model delete. |
| P2-8 | Categorical evaluation is not meaningful. Integer columns come back as floats. `drop` imputation is not implemented. |

### P3 — documentation, usability, packaging
| ID | Finding |
|---|---|
| P3-1 | All five docs need correcting (Section 9). |
| P3-2 | Add `requests`/`pytest` to a dev requirements file, pin versions, and drop unused `scikit-learn` (or use it). |
| P3-3 | Convert `test_backend_api.py` to real pytest tests with a temp DB and data dir. |
| P3-4 | Replace the hardcoded `127.0.0.1:8000` with an env-based base URL. |
| P3-5 | Code-split the 600 kB bundle. Add lint and test scripts. Add a README with verified startup steps. |
| P3-6 | `.pyc` files tracked for two Python versions. Pydantic `.dict()` deprecation. |

---

## 11. Verified startup commands

**Backend** (works from `backend/`; imports resolve as `app.*`)
```powershell
cd C:\Users\aravi\dataforge\backend
py -3.12 -m pip install -r requirements.txt          # needed once; packages were missing before this audit
py -3.12 -m uvicorn app.main:app --host 127.0.0.1 --port 8000
# Verified here on --port 8001 (port 8000 was in use by another app). Swagger: /docs
```
- Verified [V]: the server starts, `init_db()` runs, and `/api/health` returns 200. Launching from another directory is **not** verified [U] (the `app.*` imports need `backend/` as the working directory or on `PYTHONPATH`).
- With the committed DB, expect the path failures in Section 6 until P0-1/P0-2 are resolved.

**Frontend**
```powershell
cd C:\Users\aravi\dataforge\frontend
npm run dev                                           # Vite on 5173; /api proxied to 127.0.0.1:8000  [U: not started in this audit]
npx vite build --outDir <somewhere-outside-repo>      # [V] builds
npm run build                                         # [V] FAILS today: tsc error TS2339 in DatasetStudio.tsx:50
```
`api.ts` bypasses the Vite proxy and calls `http://127.0.0.1:8000` directly, so the backend port must be 8000 for the UI to reach it.

**Test script** (needs a running server on 8000; writes data — see Section 8)
```powershell
cd C:\Users\aravi\dataforge\backend
py -3.12 test_backend_api.py
```

**Side effects of this audit on the machine:** (a) Python packages from `requirements.txt` plus `requests` were installed into the user-level Python 3.12 environment; (b) 7 tracked `.pyc` files under `backend/` were rewritten when the server and scripts ran (restore with `git checkout -- "*.pyc"` if wanted); (c) temporary files and servers were limited to the session scratchpad, and both servers started by this audit were stopped. `dataforge.db`, `data/`, `.env`, and `frontend/dist/` were not modified [V: `git status` shows only `.pyc` changes plus the pre-existing untracked `backend/.venv/`].

---

## 12. Prioritized implementation roadmap (next six steps)

| Step | Goal | Includes | Exit check |
|---|---|---|---|
| **2. Stabilize storage and repo** (P0) | One data root, portable paths, safe git | Single `DATA_DIR` setting (e.g. `backend/data`, overridable by env). Store paths **relative** and resolve at read time. A one-time, **backed-up** migration of the existing DB rows. Add `.gitignore` and untrack `node_modules`, `dist`, `.pyc` (with the user's approval). Honor `.env`. | `GET /datasets/ds_student_01` has a preview; preprocess, train, generate, and export succeed on the committed records. |
| **3. Backend hardening and tests** (P0/P1) | Reliable API and real tests | pytest suite with a temp DB and data dir (replace the script, keep it as a smoke test). Validate `hidden_layers` (≥ 1 layer, generalize the VAE or constrain to 2). Enable `PRAGMA foreign_keys`, catch read errors in `preprocess`/`train`, clean up failed models, check `model.dataset_id == dataset_id`. Add `requests`/`pytest` dev deps. | `pytest` green without a live server; the data folder is untouched. |
| **4. Frontend ↔ backend integration layer** (P1) | A real data flow, minimal UI change | Fix the `api.ts` base URL (env-driven) and the response unwrapping (resolves the `tsc` error). Add API mapping functions (snake ↔ camel). Load datasets, models, and experiments on start. Correct the `startTraining` payload. Add loading, error, and empty states. | `npm run build` passes; upload shows a backend dataset id; refresh keeps data. |
| **5. Model Lab on the real backend** (P1/P2) | Real training | Wire `ModelLab` to `/models/train`. Run training in a background task with a real status poll (`/models/status`) and a stop route. Show real loss curves. Diagnose the posterior collapse (β warm-up/annealing, per-feature loss, or a longer schedule) **without changing the architecture class unless approved**. | Loss curve from the API; fidelity above a defined target on `student_performance.csv`. |
| **6. Generator, Quality, and Export/History** (P2) | Replace the remaining mocks | Wire `DataGenerator` to `/generate`; `QualityComparison` to `/evaluate` (add histogram and correlation-matrix data to the payload); `ExportHistory` to `/experiments` and `/export/{id}`; server-side audit report. Remove the orphan component. | A full run in the UI uses only backend data. |
| **7. Docs, packaging, and polish** (P3) | Make it reproducible | Correct all five docs to match reality, pin dependencies, add a README with verified startup, code-split the bundle, add lint and frontend tests, tighten CORS, add upload size limits. | A new machine can follow the README to a working app. |

---

*End of audit. Stopping here, as requested; Step 2 has not been started.*
