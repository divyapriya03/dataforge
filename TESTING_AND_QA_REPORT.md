# DataForge — Testing and QA Report (Step 6)

Date of the runs: 2026-10-24. Status words used throughout: **Passed** (executed, all assertions held), **Failed**, **Blocked** (could not be executed, reason given), **Not run**.

## 1. Executive summary

* The full workflow (upload → validation → preprocessing → training → persistence → generation → evaluation → CSV export → history) was executed end to end **through the real React UI against a real FastAPI process**, across a genuine backend restart, with every number the UI displayed cross-checked against the server's own answers. **Passed.**
* One caveat applies to that statement: the UI ran in **jsdom, not in a browser**. No browser was driven (no browser tool was available in this environment), so layout, styling, real focus handling and real file-save dialogs are **Not run**.
* A bug-hunting pass with hostile and unusual inputs found **6 defects** in application code (none lost data; the most serious were a server error on long or Windows-invalid file names, and an over-permissive CORS policy). All were fixed with regression tests. Each regression test was confirmed to fail without its fix.
* Final results: backend **142 passed**; frontend **205 passed, 6 skipped** (the 6 are the opt-in live tests, which were run separately and passed); type-check **passed**; production build **passed**; real-backend contract and UI tests **passed**. There is **no lint** configuration, so lint is **Not run**.
* Existing user data (`backend/dataforge.db`, `backend/data/`) was not modified by this step; its rows are identical to the originally committed ones.
* Readiness: suitable for a final demo and delivery of the documented scope, with the limitations in section 11. The largest are synchronous training time (grows with rows × epochs) and the absence of a browser-level test.

## 2. Environment and startup configuration

| Item | Value |
|---|---|
| OS | Windows 11 |
| Python | 3.12 (`py -3.12`); backend packages per `backend/requirements.txt` (+ `requirements-dev.txt`: pytest, httpx, requests) |
| Node / npm | 24.21 / 11.19 |
| Backend start | `cd backend` then `py -3.12 -m uvicorn app.main:app --host 127.0.0.1 --port <port>` |
| Frontend start | `cd frontend` then `npm run dev` (Vite served and compiled all pages without errors, checked in Step 5) |
| Isolation | Every test used temporary folders through `DATAFORGE_DATA_DIR` and `DATAFORGE_DB_PATH`. Scratch backends ran on port 8003 because another program ("CreditWise AI") occupies port 8000 on this machine. |
| Git state before editing | 51 modified/new files from earlier steps, nothing committed since the initial upload; all preserved. |

## 3. Tests executed and actual results

| # | Command (folder) | Result | Status |
|---|---|---|---|
| 1 | `py -3.12 -m pytest` (`backend`) | 142 passed, 1 warning (a third-party deprecation notice from Starlette/httpx) | Passed |
| 2 | `npm run typecheck` (`frontend`) | no errors | Passed |
| 3 | `npm test` (`frontend`) | 205 passed, 6 skipped (live tests, see 4–5) | Passed |
| 4 | `VITE_LIVE_API=1 VITE_API_BASE_URL=http://127.0.0.1:8003 npx vitest run src/test/live.contract.test.ts` | 4 passed against a real backend | Passed |
| 5 | `VITE_LIVE_UI=1 VITE_LIVE_PHASE=1 …` then backend restart then `… VITE_LIVE_PHASE=2 VITE_LIVE_EXP_ID=… VITE_LIVE_CSV_HASH=… VITE_LIVE_FIDELITY=…` (`frontend`, file `src/test/live.ui.test.tsx`) | phase 1: 1 passed; phase 2: 1 passed | Passed |
| 6 | `npx tsc` then `npx vite build --outDir <temporary folder>` | 2391 modules transformed, built (the tracked `dist/` was not overwritten) | Passed |
| 7 | Lint | the project has no ESLint configuration or `lint` script | Not run |
| 8 | Browser-driven test (real browser) | no browser tool available | Blocked |
| 9 | Original smoke script `backend/test_backend_api.py` against a scratch server | passed in Step 2 on a data copy; not re-run in this step (its role is covered by the E2E tests below) | Not run (this step) |

## 4. Backend API test results

Covered by `backend/tests/test_pipeline_api.py`, `test_e2e_server.py`, `test_qa_hardening.py`, `test_qa_recovery.py`: all 12 routes (health, upload, list, detail, preprocess, train, model status, model list, generate, evaluate, export, experiments).

* Status codes verified: 200 success; 400 bad input (empty/invalid CSV, wrong extension, dataset mismatch); 404 unknown dataset/model/experiment/file; 409 model not ready or artifacts missing/corrupt/incompatible; 413 oversized upload; 422 invalid parameters; generic 500 for unexpected errors, with no paths in any response.
* Request validation verified for every training and generation parameter (including the boundaries), response schemas, consistent `ds_`/`vae_`/`gen_` identifiers across upload → train → generate → evaluate → export → history, and that rejected requests leave no records or files.
* A real-server test (`test_e2e_server.py`) exercises the same over HTTP.
* Result: **Passed** (after fixing the defects in section 10).

## 5. ML pipeline test results

(`test_preprocessor.py`, `test_vae.py`, `test_evaluator.py`, `test_pipeline_api.py`, plus the exploratory probes.)

* **Validation and preprocessing — Passed:** valid CSV; empty, header-only, malformed, binary, non-UTF-8 and wrong-delimiter files rejected with clear messages; missing values (median/mean/mode/remove rows); numeric and categorical columns; constant columns; invalid numeric values and infinities treated as missing with a warning; integer and decimal preservation; identifier-column detection (including that `paid`/`midterm` are not treated as identifiers and continuous float columns never are); unseen categories; more than 50 categories rejected; preprocessing metadata saved and reused during generation.
* **Training and persistence — Passed:** valid runs; invalid dataset ids/configurations; divergence, crash midway and save failure all recorded as `failed` with no artifacts and unusable for generation; weights/pipeline reload; incompatible, corrupt and missing artifacts reported as 409; the model is loaded without pickle (an `.npz` containing a pickled object is refused); a run is only `completed` after its artifacts were saved *and re-loaded*; status survives a restart.
* **Generation — Passed:** requested row counts, column names and order, integer vs decimal types, valid categories, seed reproducibility (also across a restart and across two separate fresh servers: the same CSV hash `7ac01774` was produced), temperature effect, row-count limits (configurable), no copying of original rows, nothing stored when generation fails.
* **Evaluation — Passed:** KS, categorical distance, correlation differences, validity, composite fidelity, constant columns, missing values, small samples, non-computable metrics returned as `null` (never invented), missing evaluation rows recomputed from the real files or reported as 404.
* **Honest note on model quality:** results depend on training length and data. Training with a learning rate of 1.0 completes with finite loss but produces degenerate output (fidelity 29.8% in the probe); the evaluation shows this. It is allowed by the validation, as documented in BACKEND_DEVELOPMENT.md.

## 6. Database and persistence test results

**Passed.** Verified: first-run initialisation; idempotent additive migration of an old-schema database without losing rows; stable paths independent of the working directory (relative paths stored, legacy absolute paths from another machine still resolve); experiment and evaluation written in one transaction (rolled back together on failure); an unreachable or corrupt database returns a generic 500 while `/health` keeps working and data returns when the database comes back; failures leave no partial files (`.tmp`, CSV or weights); IDs can no longer overwrite an existing record or file; persistence of datasets, models, experiments, evaluations and CSV bytes across real process restarts (E2E, twice). SQLite foreign keys are not enforced (documented).

## 7. Frontend build, lint, type-check and test results

Type-check **Passed**; build **Passed**; lint **Not run** (no configuration). 205 tests passed across: API client (errors, timeouts, base URL configuration, CSV download), training/generation form validation, Dataset Studio, Model Lab, Data Generator, Quality Comparison, Export History, routing and navigation (every page, direct links, browser hash changes, identifier carry-over, unknown routes), and a complete UI workflow with consistent identifiers. Test output contains only the known Recharts "chart has zero width" notice from jsdom (35 occurrences); there are no React `act`, key or error warnings. Mock responses exist only inside `src/test/`.

## 8. End-to-end workflow results

Executed with a dedicated sample CSV (60 rows; `gender`, `grade`, `hours`, `score`) on empty temporary storage, backend on port 8003, UI rendered by the real application code (jsdom) and talking over HTTP to the real server (`src/test/live.ui.test.tsx`).

| # | Step | Result |
|---|---|---|
| 1 | Start the backend | Passed |
| 2 | Health endpoint responds; sidebar shows "Connected" | Passed |
| 3 | Start the frontend | Passed as a dev-server check in Step 5 (every page module compiled, HTTP 200). The E2E test renders the same code in jsdom instead of a browser |
| 4–5 | Upload CSV; dataset appears with the server-assigned id, 60 rows × 4 columns | Passed |
| 6 | Preprocessing check: 7 model input features, ready for training | Passed |
| 7 | Train (60 epochs, seed 5) | Passed |
| 8 | Training status `completed`; params and dataset id match the server | Passed (artifact files checked by the backend E2E test) |
| 9 | Generate 40 rows | Passed |
| 10 | Row count 40 and columns `gender,grade,hours,score`; categories valid, numerics numeric | Passed |
| 11–12 | Evaluation: the fidelity score and average KS shown equal the API's stored values | Passed |
| 13–14 | CSV download: bytes identical to the server's, correct filename, 41 lines | Passed |
| 15–16 | Export History: the same experiment, model, dataset, record count and fidelity; its download equals the first | Passed |
| 17 | Restart the backend (a new process id) | Passed |
| 18 | After restart: experiment still listed as Completed with the same fidelity, details/evaluation load, CSV hash identical | Passed |

Not covered by this test: real-browser rendering, visual layout and the browser's own download handling (**Not run**). Not every stage is "skipped": none was skipped, but one environment substitution is declared above (jsdom for a browser).

## 9. Negative and recovery test results

| Situation | How tested | Result |
|---|---|---|
| Backend unavailable | UI tests per page (list, upload, training, generation, export) and the app-level outage test | Passed: clear error, retry where relevant, no false success |
| Wrong API URL | `api.test.ts` (configured URL used everywhere; error names the address) | Passed |
| Request timeout | client timeout tests (training, download) and a UI test | Passed: outcome reported as unknown with guidance |
| Dataset ID does not exist | backend + UI | Passed (404, message shown) |
| Training fails (divergence, crash midway, save failure) | backend | Passed: `failed` record, no artifacts, generic message, unusable for generation |
| Model artifact missing / corrupt / incompatible | backend + E2E | Passed (409, no paths) |
| Generation fails midway (evaluation error, database error, disk full while writing) | backend | Passed: no experiment, no evaluation, no file |
| Evaluation cannot compute a metric | backend + UI | Passed: `null`, shown as "not available", never charted as zero |
| Generated CSV missing | backend + UI | Passed: 404, experiment listed as "Output file missing", download disabled, no silent regeneration |
| Database cannot be accessed (directory or garbage file) | backend | Passed: generic 500; process stays up; data intact afterwards |
| Experiment history empty | backend + UI | Passed |
| Invalid form values | UI unit and component tests | Passed (inline errors, no request sent) |
| Duplicate submission | UI tests for upload, training, generation, downloads | Passed, including **after leaving and re-entering the page** (defect B5) |
| Failed requests do not change stored data | `test_failed_operations_do_not_change_stored_data` | Passed (rows and files unchanged apart from the intended `failed` model row) |

## 10. Bugs found and fixes applied

| ID | Pri | Failure and reproduction | Root cause | Fix | Regression test |
|---|---|---|---|---|---|
| B1 | P2 | Upload of a 300-character file name returned **HTTP 500**; names with characters Windows forbids (`< > | ? *`) would fail the same way, and generated file names were built from the dataset name too | the original name was used as the on-disk name | on-disk and derived names reduced to `[A-Za-z0-9._-]`, max 80 chars; display name keeps the original (control characters removed, length bounded, extension kept) | `test_qa_hardening.py` (name matrix, generation with awkward names, non-ASCII export header). 17 tests fail without the fixes. |
| B2 | P2 (security) | Any web origin got `Access-Control-Allow-Origin: <that origin>` with `allow-credentials: true` | CORS configured as `*` plus credentials | only configured origins (default: Vite dev/preview on localhost), no credentials, only GET/POST/OPTIONS and `Content-Type`; configurable via `DATAFORGE_CORS_ORIGINS` | `test_only_configured_origins_are_allowed`, `test_cors_origins_can_be_configured` |
| B3 | P3 | A semicolon/tab/pipe-delimited file was accepted as one column named `a;b;c` | pandas default delimiter, no check | such a single-column file is rejected with a hint | `test_files_with_another_delimiter_are_rejected_with_a_hint` |
| B4 | P2 | Leave Model Lab (or the generator) while a request is running and come back: the form is enabled again and a second training/generation could be started | the "in progress" guard lived in component state | shared in-flight registry (`services/inflight.ts`); the returning page shows that a request is still running and refreshes the model list when it ends | `ModelLab.test.tsx` and `DataGenerator.test.tsx` (reproduced failing first) |
| B5 | P3 | IDs were random 8-hex values written with `INSERT OR REPLACE` and open mode `wb`: a collision would have overwritten an existing record or file | no uniqueness check | IDs are re-drawn if a record or file exists; files created exclusively; a failed exclusive create no longer deletes the existing file | `test_a_leftover_file_is_never_overwritten…`, `test_new_id_skips…`, `test_a_colliding_model_id…` |
| B6 | P3 | `np.load` of a model file kept the file handle open until garbage collection (on Windows an open file cannot be replaced) | no context manager | `with np.load(...)` | covered by the existing artifact tests; no separate deterministic reproduction was possible |
| B7 | test | Two live contract tests assumed the server already contained a dataset | test assumption | they upload their own dataset when none exists | live contract test now passes on an empty server |

Two mistakes of my own were caught by the new tests while fixing: the display-name truncation initially removed the `.csv` extension of very long names, and an earlier draft of the upload clean-up could have deleted a pre-existing file. Both are fixed and covered.

Earlier steps' fixes (for example negative seeds returning 500, history ordering within the same second, the upload timestamp) are in the Step 2–5 summaries and remain covered by the suite.

## 11. Known limitations and skipped tests

* **Synchronous training.** Cost grows with rows × epochs: measured 13.1 s for 10 epochs on 20,000 rows × 9 columns (about 1.3 s per epoch), so 500 epochs on that data would take roughly 11 minutes, longer than the 10-minute UI timeout. The server keeps training after the browser gives up; the UI then points to "Refresh models". Fixing this needs background jobs (out of scope).
* **Memory and speed otherwise fine on the samples:** upload 0.14 s; generating the maximum 50,000 rows took 2.9 s with about 218 MB peak Python memory; export of 50,000 rows 0.04 s; 8 parallel train+generate requests all succeeded with unique IDs. Larger inputs (up to the 25 MB upload limit) were not stress-tested.
* **Duplicate GETs:** landing directly on `#/history` loads the experiment list twice, and Model Lab re-requests the dataset list the app already holds. Both are cheap and harmless; React StrictMode also doubles effects in development. Left as is.
* **Heuristics:** a non-float column with all-unique values (20+ rows) is treated as an identifier and excluded by default (visible and editable in Dataset Studio); duplicate CSV header names are renamed by pandas (`a`, `a.1`) without a warning; CSV files must be comma-separated and UTF-8.
* **Unstable settings are allowed:** e.g. learning rate 1.0 (see section 5).
* **Static health text:** `/api/health` always reports the "VAE Neural Engine (Active)" string; it does not probe a model.
* **CORS/ports:** if Vite starts on a port other than 5173/4173/3000 (for example because 5173 is busy) the browser blocks API calls until that origin is added to `DATAFORGE_CORS_ORIGINS`.
* **Not part of the application:** no authentication, no delete/rename of datasets, models or experiments, no pagination, SQLite foreign keys not enforced.
* **Security review scope:** this was a basic, targeted review (upload names and size, path handling of identifiers, path/secret exposure in responses, pickle-free model loading, CORS, overwrite protection, generation limits). It is **not** a complete security audit.
* **Skipped/not run:** lint (no configuration); real-browser tests (blocked); the three live test files are skipped by default and only run with the `VITE_LIVE_*` variables.
* **Tooling noise:** the Recharts "zero width" notices in jsdom; two stalled helper processes (`python3 -` waiting on stdin, started by an unavailable command in this environment) may still be running on the development machine and are harmless.

## 12. Remaining mock-data dependencies

None in application code (searched for mock, fake, dummy, placeholder, sample data, `Math.random`, hardcoded scores in both frontend and backend). Test fixtures and the stateful stand-in backend exist only under `frontend/src/test/` and are never imported by the app. Intentionally static: the Dashboard's explanatory "Deep Learning Formulation" card and the health message above.

## 13. Recommended next steps before final delivery

1. Run the application once by hand in a real browser (`npm run dev` + backend) and click through the workflow; this is the main gap left.
2. Decide whether to commit the work; everything since the initial upload is uncommitted, and `frontend/node_modules`, `dist/`, the database, model weights and `.env` files are tracked. A `.gitignore` and untracking them would remove most of the git noise (needs your approval, as it changes history).
3. If long training runs matter for the demo, keep datasets small (a few thousand rows) or epochs moderate, or schedule background-job support as separate work.
4. Optionally add ESLint if a lint gate is required.
5. Final documentation and presentation work can then proceed from this report and the updated `BACKEND_DEVELOPMENT.md`, `FRONTEND_DEVELOPMENT.md` and `DEVELOPMENT_SETUP.md`.
