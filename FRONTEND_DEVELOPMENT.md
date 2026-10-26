# DataForge: Deep Learning-Based Synthetic Data Generation
## Phase 4: Frontend Development & Module Integration Document
**Course / Subject:** AI23531 – Deep Learning  
**Project Title:** DataForge – Deep Learning-Based Synthetic Data Generation Platform  
**Document Version:** 1.0.0  
**Status:** All six pages use the real backend; see section 7 for the end-to-end workflow  

---

## 1. Overview & Architecture

The frontend for **DataForge** is a fully functional Single Page Application built with **React 18**, **TypeScript**, **Tailwind CSS**, and **Recharts**. It provides an interactive interface for the entire lifecycle of tabular synthetic data generation powered by Variational Autoencoders (VAEs).

The application is accessible locally at:
🌐 **http://127.0.0.1:5173/**

---

## 2. Completed Modules & Implementation Details

### Module 1: Dashboard (`#/dashboard`)
Counts of datasets, models and saved experiments, the total number of generated rows, and the average fidelity and validity scores of the saved runs, all computed from backend data. With nothing stored it shows "no data" / "No experiments yet" instead of placeholder figures. The recent-runs table shows the latest eight experiments. The "Deep Learning Formulation" card is static explanatory text.

---

### Module 2: Dataset Studio (`#/dataset-studio`) — connected to the backend
Uploads a CSV to the backend, lists stored datasets, shows the server's schema (types, missing values, statistics) and a 50-row preview, lets you adjust which columns are used and how missing values are handled, and runs the backend's preprocessing dry run. See section 7.3.

---

### Module 3: Model Lab (`#/model-lab`) — connected to the backend
Trains a real model through `POST /api/models/train`. Full description in section 5 below; it no longer simulates training.

---

### Module 4: Data Generator (`#/generator`) — connected to the backend
Generates synthetic rows with a trained model through `POST /api/generate` and shows the real result, preview rows and CSV download. See section 7.4.

---

### Module 5: Quality Comparison (`#/quality`) — connected to the backend
Shows the evaluation the backend saved for the selected experiment. See section 7.5.

---

### Module 6: Export & History (`#/history`) — connected to the backend
Lists the experiments stored by the backend and downloads their generated CSV files. Full description in section 6 below; it no longer builds CSV or reports in the browser.

---

## 3. Reusable UI Components & State Management

* `src/components/Sidebar.tsx`: persistent navigation with the active page highlighted and the backend connection indicator.
* `src/components/Navbar.tsx`: header showing the active dataset and model.
* `src/components/EvaluationPanel.tsx`: the saved-evaluation display shared by Export History and Quality Comparison.
* `src/services/api.ts`: the only place that talks HTTP (typed functions, safe error parsing, timeouts, CSV download).
* `src/services/mappers.ts`, `format.ts`, `download.ts`, `trainingConfig.ts`: model mapping, display helpers ("not available" for missing values), browser file save, training-form validation.
* `src/hooks/useCsvDownload.ts`: one download per experiment at a time, with success/error notices.
* `src/routing.ts`: hash routes and the query parameters that carry identifiers between pages.
* `src/types/api.ts`: TypeScript types mirroring the backend schemas.
* `src/App.tsx`: loads datasets, models and experiments from the backend, owns the current selections and the routing.

There is no mock data in the application code. Test-only fixtures and a stateful stand-in backend live in `src/test/` and are never imported by the app.

---

## 4. Verification & Production Build
See section 7.8 for the commands and the results actually obtained.

---

## 5. ModelLab — real training (Step 3)

### 5.1 Running the frontend and backend
```powershell
# terminal 1 - backend (from backend/)
py -3.12 -m uvicorn app.main:app --host 127.0.0.1 --port 8000
# terminal 2 - frontend (from frontend/)
npm install        # first time only
npm run dev        # http://localhost:5173
```
| Variable (`frontend/.env`) | Purpose | Default |
|---|---|---|
| `VITE_API_BASE_URL` | Backend base URL, used by every API call (`src/services/api.ts`) | `http://127.0.0.1:8000` |
| `VITE_TRAIN_TIMEOUT_MS` | How long the UI waits for the synchronous training request | `600000` (10 min) |

These are public build-time values; never put secrets in them. The backend only accepts browser requests from the Vite dev/preview origins on localhost (see `DATAFORGE_CORS_ORIGINS` in BACKEND_DEVELOPMENT.md section 3), so no proxy is needed; if Vite starts on another port, add that origin.

### 5.2 Navigation
The app has no router library; pages are tabs held in `App.tsx`. The active tab is mirrored in the URL hash so links work directly: `#/dashboard`, `#/dataset-studio`, **`#/model-lab`**, `#/generator`, `#/quality`, `#/history`. The sidebar entry "3. Model Lab" opens ModelLab.

### 5.3 Using ModelLab
1. **Dataset.** The list comes from `GET /api/datasets` (nothing is hardcoded). It shows file name, ID, rows, columns and how many columns are usable (identifier columns are excluded by the backend). "Check preprocessing (dry run)" calls `POST /api/preprocess` and shows the resulting feature count, unused columns and warnings. A separate preprocessing step is **not** required: training prepares the data itself. Datasets with fewer than 10 rows, or with no usable columns, cannot be trained on.
2. **Configuration.** Only parameters the backend accepts are offered, with the backend's limits enforced before the request is sent:

   | Field | Request field | Limits | Default |
   |---|---|---|---|
   | Epochs | `epochs` | 1–500, whole number | 50 |
   | Learning rate | `learning_rate` | > 0 and ≤ 1 | 0.001 |
   | Batch size | `batch_size` | 8–4096 | 32 |
   | Latent dimension | `latent_dim` | 2–64 | 8 |
   | Hidden layer 1 / 2 | `hidden_layers` | exactly two sizes, 2–1024 each | 128 / 64 |
   | KL weight (β) | `beta_weight` | 0–100 | 1 |
   | Random seed (optional) | `random_seed` | 0–4294967295 | none (random) |
   | Model name (optional) | `model_name` | ≤ 100 characters | generated |

   The model is a fixed variational autoencoder with two hidden layers and LeakyReLU; the number of layers and the activation cannot be changed, and the UI says so. The default of 50 epochs is the backend's; datasets of a few hundred rows did better with 200 or more in backend tests (see BACKEND_DEVELOPMENT.md §4).
3. **Start Training** sends the request and disables the form until the server answers. A second submit while one is running is ignored.
4. **Status.** One of: *ready*, *training* (waiting for the server), *completed*, *failed*. See 5.4.
5. **Result** (only values the backend returned): model ID, name, dataset ID, model type, status, epochs run, server start/completion timestamps (UTC) and the duration derived from them, the configuration used, the final training losses and a per-epoch loss chart. Missing values are shown as "unavailable". Training losses are not a quality measure; quality is evaluated after generation. A training run is identified by its **model ID**; experiment IDs (`gen_…`) are created when data is generated.
6. **Models on the backend** lists `GET /api/models` with real statuses (including failed runs) and a Refresh button. Completed models can be selected; "Use this model in Data Generator" selects it and navigates there. Model artifacts live only on the server.

### 5.4 Training status and the synchronous-training limitation
The backend trains **inside** the `POST /api/models/train` request and answers once, when training has finished or failed. Therefore:
* there is no progress percentage or epoch counter, and none is shown; the UI shows an elapsed-seconds counter and says it is waiting;
* the model status endpoint cannot report live progress for this request, so ModelLab does not poll;
* if the request fails or times out (network error, browser/proxy limit, 10-minute UI timeout), the **outcome is unknown**: the server may still be training or may have finished. The UI says this and points to "Refresh models", which shows the real status of every stored model (`training`, `completed` or `failed`). Cancelling the browser request does not stop the server;
* very long trainings should be avoided until the backend runs training in the background (not part of this step).

### 5.5 Tests and checks (`frontend/`)
```powershell
npm run typecheck   # tsc --noEmit
npm test            # vitest run (jsdom, mocked fetch; no server needed)
npm run build       # tsc && vite build  (overwrites the tracked dist/ folder)
```
Tests mock the network only inside the test files (`src/test/backend.ts`); the application never uses mock responses for ModelLab. `src/test/live.contract.test.ts` calls a real backend and is skipped unless `VITE_LIVE_API=1` (it trains a small model, so point it at a scratch backend; the command is in the file header). There is no ESLint configuration, so there is no lint script.

### 5.6 Mock data
None of the six pages uses mock data any more; see section 7.9 for what remains static.

---

## 6. Export History — experiments and CSV downloads (Step 4)

### 6.1 Opening the page
Sidebar entry "6. Export & History", or directly `#/history`. From the page you can jump to Model Lab, the Data Generator (with the experiment's model selected, when that model still exists) and Quality Comparison (with the experiment selected).

### 6.2 Where the history comes from
A row is written to the backend's `experiments` table only when a generation (`POST /api/generate`) fully succeeded: the CSV, the experiment row and its evaluation are stored together. The page lists `GET /api/experiments` (newest first). Consequences:
* **Failed or interrupted generations never appear**; there are no "failed" or "in progress" entries.
* `status` is `completed` while the generated CSV exists on the server and `output_missing` if the file was removed afterwards (the row and its stored evaluation remain). `output_missing` rows cannot be downloaded.
* Dataset or model names show "record removed" when those records no longer exist; experiments created by older versions have no stored model type (shown as "not available").
* The page shows only what the API returns. Nothing is generated or filled in by the browser.

### 6.3 List, search, sort
Columns: experiment ID, dataset (name and ID), model (name and ID), records, fidelity, validity, status, created time (UTC), actions. Controls (client-side, over the loaded list): search by experiment, dataset or model ID/name or file name; status filter (only statuses present are offered); sort by newest, oldest, highest fidelity or most records. There is no pagination (the backend returns the full list).

### 6.4 Details and saved evaluation
"View" shows the experiment (IDs, status, model type, records, temperature, seed, creation time, output-file availability), the training configuration of its model (`GET /api/models/status/{model_id}`) and the evaluation saved at generation time (`GET /api/evaluate/{experiment_id}`):
* fidelity score and validity (percent of valid cells), average KS statistic, correlation differences (mean │ΔR│ and Frobenius / n);
* fidelity components with their weights; per-column KS statistic, p-value and Wasserstein distance for numeric columns; total variation distance and invalid categories for categorical columns; backend warnings; and a bar chart of the per-column distances.
Definitions are in BACKEND_DEVELOPMENT.md section 6. The frontend never recomputes a metric. A value the backend did not store shows as "not available" (never 0), and such columns are left out of the chart (the page says how many). A genuine 0 shows as `0.0000`. If an experiment has no stored evaluation the panel says so. The fidelity score is a heuristic summary, **not** a guarantee of data quality or privacy; the backend's limitation text is shown with every evaluation.

### 6.5 Downloading a CSV
"Download CSV" (row or details panel) calls `GET /api/export/{experiment_id}`, which streams that experiment's own generated file (`text/csv`). The browser saves the bytes unchanged as `dataforge_experiment_<experiment_id>.csv`. Notes:
* the button is disabled for `output_missing` experiments and while that experiment's download is running, so repeated clicks send one request;
* errors (404 file/experiment missing, network failure, a non-CSV answer) are shown in a banner; a 404 also refreshes the list;
* the original uploaded dataset is never substituted, and no CSV content is stored in the browser;
* file paths on the server are never sent to or shown by the frontend.

### 6.6 Limitations
* Experiments exist only for data generated through the backend API: the Data Generator page, other API clients and the backend smoke script all create them.
* No pagination or server-side filtering; no delete or rename.
* The JSON "audit report" of the old mock page was removed (it was built from mock data). A server-side report is not implemented.
* The stored evaluation reflects the data at generation time; it is not recomputed.

### 6.7 Tests
`src/pages/ExportHistory.test.tsx` (list, empty/loading/error states, statuses, filters, details, null metrics, downloads, duplicate clicks, links), `src/App.test.tsx` (navigation) and `src/services/api.test.ts` (download client) run with `npm test`. `src/test/live.contract.test.ts` exercises history, evaluation and download against a real backend when `VITE_LIVE_API=1` (scratch backend only).

---

## 7. End-to-end workflow and integration (Step 5)

### 7.1 Starting everything
```powershell
# terminal 1 - backend (run from the backend folder; the app imports as `app.*`)
cd backend
py -3.12 -m pip install -r requirements.txt
py -3.12 -m uvicorn app.main:app --host 127.0.0.1 --port 8000

# terminal 2 - frontend
cd frontend
npm install        # first time only
npm run dev        # http://localhost:5173
```
Environment variables (all optional):

| Where | Variable | Purpose | Default |
|---|---|---|---|
| `frontend/.env` | `VITE_API_BASE_URL` | Backend base URL used by every request | `http://127.0.0.1:8000` |
| `frontend/.env` | `VITE_TRAIN_TIMEOUT_MS` | UI timeout for the (synchronous) training request | `600000` |
| `frontend/.env` | `VITE_GENERATE_TIMEOUT_MS` | UI timeout for a generation request | `300000` |
| backend environment | `DATAFORGE_DATA_DIR`, `DATAFORGE_DB_PATH` | Where uploads, models, generated CSVs and the SQLite file live | `backend/data`, `backend/dataforge.db` |
| backend environment | `DATAFORGE_MAX_UPLOAD_MB`, `DATAFORGE_MAX_GENERATE_ROWS` | Upload and generation limits | 25, 50000 |

`VITE_*` values are compiled into the public page; never put secrets there. The backend accepts cross-origin requests only from the origins listed in `DATAFORGE_CORS_ORIGINS` (default: the Vite dev server on port 5173 and preview server on 4173, on `localhost` and `127.0.0.1`), so the page can call it directly; if Vite is started on a different port (for example because 5173 is busy) the browser will block the calls until that origin is added. If another program already uses port 8000 (this happened on the development machine), start the backend on a different port and set `VITE_API_BASE_URL` to match.

### 7.2 The workflow and its pages
`Dataset Studio` (upload, review, preprocessing check) → `Model Lab` (train) → `Data Generator` (generate) → `Quality Comparison` (inspect the saved evaluation) → `Export History` (all experiments, CSV download). Every step reads and writes the backend; the browser stores nothing (the column edits made in Dataset Studio are kept in memory until the page is reloaded).

Routes are hash based: `#/dashboard`, `#/dataset-studio`, `#/model-lab`, `#/generator`, `#/quality`, `#/history`. Identifiers travel in the URL so links, reloads and the browser's back button keep the context:

| URL | Meaning |
|---|---|
| `#/dataset-studio?dataset=<ds_id>` and `#/model-lab?dataset=<ds_id>` | the dataset being worked on |
| `#/generator?model=<vae_id>` | the model used for generation |
| `#/quality?experiment=<gen_id>` | the experiment whose evaluation is shown |

Any other address shows a "Page not found" page with a link to the dashboard. The sidebar highlights the current page and carries the current identifier along.

### 7.3 Dataset upload and preprocessing
* **Upload** sends the file to `POST /api/datasets/upload`. The server rejects non-CSV files, empty files, files without data rows, unparsable files (400) and files over `DATAFORGE_MAX_UPLOAD_MB` (413); the page shows the server's message and no dataset is created. On success the server returns the dataset ID, which is used everywhere afterwards, and the dataset list is reloaded.
* **Review**: the column table shows what the server inferred (type, missing values, identifier-like columns excluded by default), the Preview tab shows the first 50 stored rows and the Statistics tab shows min/max/mean/std or categories. Missing numbers are shown as "not available".
* **Configuration**: you can exclude a column, treat it as numeric or categorical, and choose how missing values are handled (median, mean, most frequent, remove rows). The edited configuration is sent to training as `columns`; without edits the server uses the dataset's stored schema.
* **Preprocessing check** (`POST /api/preprocess`, a dry run) shows the number of model input features, rows used, columns not used with the reason, and warnings (for example values treated as missing or constant columns). Errors such as a numeric column that contains mostly text are shown as returned. Training does its own preprocessing with the same rules; the check changes nothing on the server except an informational file.
* Datasets with fewer than 10 rows, or with no usable columns, cannot be trained on; the pages say so and disable the action.

### 7.4 Training and generation
* **Training** (Model Lab): synchronous, no live progress; see section 5.4 for the timeout behaviour. The model ID and dataset ID shown come from the server response.
* **Generation** (`POST /api/generate`): choose a completed model (failed or unfinished models are not offered), the number of records (10 to 50,000; the server may be configured lower and then explains the limit), a temperature (0.1 to 2.5) and an optional seed (0 to 4294967295; empty uses the server default 42). The page sends the model ID together with the ID of the dataset it was trained on. The result shows the experiment ID, the file name, the fidelity and validity scores returned by the server and the first 50 generated rows. Generation is saved on the server together with its evaluation; nothing is kept if it fails. A generation request that is lost (network error, timeout) has an unknown outcome; the page points to Export History to check.
* The Export History list is reloaded after every successful generation.

### 7.5 Quality Comparison and evaluation limits
The page loads `GET /api/evaluate/{experiment_id}` for exactly the selected experiment and refuses a response that names a different experiment. It shows the score cards, per-column KS statistics, categorical distances and per-category proportions (real vs synthetic), real vs synthetic means and standard deviations, the correlation differences, backend warnings and the limitations text. Values the backend did not store are shown as "not available" and are never drawn as zeros; an experiment without a stored evaluation says so. Limits: the metrics are per-column and pairwise statistics; the correlation matrices themselves are not stored (only summary differences); the fidelity score is a heuristic, not a measure of privacy or of usefulness for a task. Definitions: BACKEND_DEVELOPMENT.md section 6.

### 7.6 CSV export
Export History, Data Generator and Quality Comparison all download through the same function (`GET /api/export/{experiment_id}`) and save the bytes unchanged as `dataforge_experiment_<experiment_id>.csv`. Line endings are whatever the server wrote (CRLF on Windows). The file is never regenerated or replaced by the uploaded dataset: if it has been removed from the server, the experiment is listed as "Output file missing", its download is disabled and a failed request shows the server's message. The backend checks that the experiment ID belongs to a stored experiment and serves only that experiment's file.

### 7.7 Persistence
Datasets, models, experiments, evaluations and the generated CSVs are stored by the backend (SQLite plus files under `DATAFORGE_DATA_DIR`) and survive restarts; this is verified by an automated test that restarts a real server process. Reloading the page restores everything from the backend, except unsaved column edits in Dataset Studio.

### 7.8 Test commands and actual results
Run on 2026-10-24 on Windows with Node 24 and Python 3.12:

| Command (folder) | Result |
|---|---|
| `npm run typecheck` (`frontend`) | passes |
| `npm test` (`frontend`) | 200 passed, 4 skipped (the live tests, see below) |
| `tsc` then `vite build` (`frontend`, output redirected to a temporary folder) | passes; 2390 modules transformed |
| `py -3.12 -m pytest` (`backend`) | 96 passed (includes 2 end-to-end tests against a real server process, with a restart) |
| `VITE_LIVE_API=1 VITE_API_BASE_URL=http://127.0.0.1:8003 npx vitest run src/test/live.contract.test.ts` (`frontend`, against a scratch backend started with `DATAFORGE_DATA_DIR`/`DATAFORGE_DB_PATH` pointing at a copy) | 4 passed |
| `npm run lint` | not available: the project has no ESLint configuration |

`npm run build` (`tsc && vite build`) overwrites the tracked `dist/` folder; the check above wrote to a temporary folder instead. The UI tests use a stateful in-memory stand-in for the backend (`src/test/fakeServer.ts`) and cover the complete workflow, error paths and identifier consistency; the live test and the backend end-to-end tests cover the real HTTP contracts. No browser-driven test was run.

### 7.9 Known limitations
* Training is synchronous (no progress, see 5.4); very long runs may exceed the UI timeout while the server keeps training.
* The Dashboard's explanatory "Deep Learning Formulation" card is static text.
* Column edits are held in memory only; reloading the page returns to the server-detected configuration.
* No pagination or server-side filtering of experiments, no delete or rename of datasets, models or experiments.
* The backend remains single-process with no authentication; do not expose it to untrusted networks.
* The model quality caveats in BACKEND_DEVELOPMENT.md section 10 apply: results depend strongly on training length and data, and the evaluation says nothing about privacy.
