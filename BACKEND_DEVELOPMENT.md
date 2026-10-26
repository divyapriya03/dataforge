# DataForge — Backend Development Guide

**Scope:** FastAPI backend, the ML pipeline, storage, and tests.
**Status:** The pipeline *dataset → validation → preprocessing → training → persistence → generation → evaluation → export* runs end to end and is covered by automated tests (see §9). Quality claims in this document are limited to what was measured and are stated with their conditions.

---

## 1. What the ML pipeline actually is

| Item | Fact |
|---|---|
| Model | A **Variational Autoencoder written in plain NumPy** (`backend/app/models/vae.py`). No PyTorch/TensorFlow is used anywhere; earlier documents that said otherwise were wrong. |
| Architecture | Fully connected: input → hidden1 → hidden2 → (μ, log σ²) → z → hidden2 → hidden1 → sigmoid output. **Exactly two hidden layers** (e.g. `[128, 64]`); other depths are rejected with HTTP 422. LeakyReLU (α = 0.2) activations. |
| Optimisation | Hand-derived backpropagation (checked against finite differences in `tests/test_vae.py`), Adam, gradient clipping at ±5. |
| Data representation | Numeric columns min-max scaled to [0, 1]; categorical columns one-hot encoded. |
| Generation | Draw `z ~ N(0, τ²I)` (τ = `temperature`), decode, map back to the original columns (§5). |
| Is it "deep learning"? | It is a small neural generative model trained by gradient descent. It is a shallow network (two hidden layers) trained on small tabular data; no claim is made beyond that. |

### 1.1 Loss

```
loss = recon_weight · recon + β · KL
recon = Σ_features (x − x̂)²           (averaged over the batch)
KL    = −½ Σ_latent (1 + log σ² − μ² − σ²)   (averaged over the batch)
recon_weight = 20                       (= 1 / 2σ² of a fixed-variance Gaussian decoder, σ² = 0.025)
```

`recon_loss` and `kl_loss` in API responses are the two raw terms; `final_loss` (total) includes the weight.

**Why `recon_weight = 20`:** with the original per-feature *mean* squared error and β = 1, the KL term dominated and the posterior collapsed — the decoder ignored z and produced roughly column means (mean KS ≈ 0.4 on the sample data, KL ≈ 0.0004). Summing the reconstruction error and weighting it by 20 removed the collapse in our experiments (KL ≈ 4–7, mean KS 0.07–0.09 after 200 epochs on the sample dataset). 20 was chosen empirically on the sample student dataset (single runs: weight 1 still collapsed, 10 gave mean KS ≈ 0.13, 20 and 50 both gave ≈ 0.07). It is a constant in `vae.py` (`DEFAULT_RECON_WEIGHT`), not a tuned guarantee for other datasets.

> Models trained before this change used the old loss; their stored weights still load and generate, but they are lower quality (fidelity ≈ 75 % in the committed examples).

---

## 2. Preprocessing workflow (`app/services/preprocessor.py`)

1. **Upload validation** (`inspect_csv`): the file must be a readable CSV with a header and ≥ 1 data row. Errors are returned as HTTP 400 with a clear message (empty file, header only, unparsable CSV). Uploads are limited to `DATAFORGE_MAX_UPLOAD_MB` (default 25 MB → HTTP 413); only `.csv` (case-insensitive) is accepted; directory parts of the file name are stripped, and the name used on disk is reduced to letters, digits, `.`, `_` and `-` (at most 80 characters) so any upload name is safe on every operating system. Files with another delimiter (for example `;` or tab) are rejected with a hint, because only comma-separated files are supported.
2. **Schema inference.** Each column gets `inferredType` / `selectedType` (`numeric`, `categorical`, `drop`; `id` as an inferred type), null counts and an imputation strategy.
   * Bool and non-numeric columns → categorical; numeric → numeric.
   * *Identifier* columns are excluded by default: the name contains an `id`/`uuid`/`guid` token (`student_id`, `userId`) **or**, for datasets with ≥ 20 rows, every value is unique in a non-float column. Continuous float columns are never treated as identifiers. Entirely empty columns are excluded. These are defaults; a client may override them through the column configuration.
3. **Fitting** (`fit`, run at training time and by `/api/preprocess`):
   * Missing values: numeric `median` (default) / `mean` / `mode`; categorical `mode`; strategy `drop` removes those *rows* before fitting (reported as a warning).
   * Numeric columns are coerced with `pd.to_numeric`; non-numeric and ±inf entries are treated as missing and counted in `warnings`. If more than half of a column's values are not numbers, or none are valid, an error is returned instead.
   * Constant columns are kept and reproduced as constants (warning). Single-category columns likewise.
   * More than **50** distinct categories in one categorical column → error (it would create an unmanageable one-hot block); exclude it or treat it as an identifier.
   * Stored per column: min/max, fill value, integer-vs-float and the number of decimals seen (≤ 6), category list.
   * **Nothing is dropped silently:** excluded columns are listed in `dropped_columns` with a reason, and every other adjustment is in `warnings`. Both are returned by `/api/preprocess` and saved with the pipeline.
4. **Persistence.** The fitted pipeline is saved as JSON next to the model (`models/<model_id>_preprocessor.json`). Generation **loads** it; it never re-fits.
5. **`transform`** applies a fitted pipeline to new data: missing → fitted fill value; out-of-range numerics clipped to [0, 1]; categories unseen during fitting become an all-zero one-hot block (counted in `unseen_category_count`).

**Unsupported / limitations:** datetime columns are treated as text (categorical, so high-cardinality ones are rejected); free-text columns are not supported; very wide categorical data is limited by the 50-category rule; duplicate header names are renamed by pandas (`a`, `a.1`) and not detected; numeric-looking category codes (1, 2, 3) that are marked categorical come back as strings.

---

## 3. Storage conventions (`app/core/config.py`)

All locations are resolved from the code's own location, never from the working directory, and can be overridden with environment variables.

| Setting | Default | Env var |
|---|---|---|
| Data root | `backend/data/` | `DATAFORGE_DATA_DIR` |
| SQLite database | `backend/dataforge.db` | `DATAFORGE_DB_PATH` |
| Upload limit | 25 MB | `DATAFORGE_MAX_UPLOAD_MB` |
| Max rows per generation | 50 000 | `DATAFORGE_MAX_GENERATE_ROWS` |
| Allowed browser origins (CORS) | Vite dev/preview servers on localhost (ports 5173, 4173, 3000); credentials are not allowed | `DATAFORGE_CORS_ORIGINS` (comma-separated, `*` for everything) |

```
data/uploads/    <dataset_id>_<original name>.csv
data/models/     <model_id>_weights.npz            VAE parameters (NumPy .npz)
                 <model_id>_preprocessor.json      the pipeline fitted for THIS model
                 preprocess_<dataset_id>.json      last /api/preprocess dry run (informational)
data/synthetic/  synthetic_<dataset>_<experiment_id>.csv
```

* **Paths in SQLite are stored relative to the data root** (`models/vae_ab12cd34_weights.npz`) so the project can be moved. Rows written earlier with absolute paths (including paths from another machine) still work: if the stored absolute path does not exist, the file is looked up by name in the matching folder. No existing row was rewritten.
* Binaries are never stored in SQLite; it holds metadata and paths only.
* **Schema migration:** `init_db()` adds new columns with `ALTER TABLE … ADD COLUMN` (idempotent, existing rows untouched): `models.model_type, error_message, started_at, completed_at, preprocessor_path` and `evaluations.details_json`. Verified: all pre-existing rows are identical after migration.
* Models created before this change have no `preprocessor_path`; generation falls back to the old per-dataset file `models/preprocessor_<dataset_id>.json`.
* Files are written atomically (temp file + rename) and a failed operation removes what it created.
* SQLite foreign keys are **not** enforced (no `PRAGMA foreign_keys`); relations are checked in code.

---

## 4. Training (`POST /api/models/train`)

Training runs **synchronously**; the response is the final state. Order of operations:

1. Validate the request (Pydantic) → **422** for e.g. wrong `hidden_layers`, `epochs` outside 1–500, `latent_dim` outside 2–64, `batch_size` outside 8–4096, invalid name. Nothing is written.
2. Look up the dataset (**404**), read it (**404** if the file is missing, **400** if unreadable), fit the preprocessing (**400** with the reason) and require ≥ 10 usable rows (**400**). Nothing is written yet.
3. Insert the model row with status `training`, fit the VAE, check the loss is finite every epoch (otherwise fail with **422** "Training diverged…").
4. Save weights and pipeline, **reload both** to prove they are usable, then mark the model `completed` with its loss history and `completed_at`.
5. Any failure marks the model row `failed` with an `error_message` (no internals), removes partial files and returns an error. A `failed` model cannot be used for generation (**409**).

Request fields beyond the original contract (all optional): `random_seed` (reproducible training), `columns` (per-column preprocessing configuration; defaults to the dataset's stored schema). `activation` from earlier UI drafts is **not** supported.

Because training is synchronous, `/api/models/status/{id}` reflects a *stored* final or failed state; there is no live progress yet. `model_type` is `numpy-vae`.

**Training length matters.** On the 250-row sample data, mean KS fell from ≈ 0.15 (30 epochs) to ≈ 0.07 (400 epochs); on the 300-row test fixture from ≈ 0.25 to ≈ 0.07–0.09 (single runs, one seed each, so treat as indicative). The API default is still 50 epochs / lr 0.001, which gives fidelity ≈ 75 % on the sample; in one comparison `epochs=200, learning_rate=0.003` scored ≈ 87 %, and `learning_rate=0.01` was worse (≈ 72 %). Use ≥ 200 epochs for datasets of a few hundred rows.

---

## 5. Generation (`POST /api/generate`)

* Inputs: `model_id`, optional `dataset_id` (defaults to the model's; if given it must match, otherwise **400**), `num_records` (≥ 10, ≤ `DATAFORGE_MAX_GENERATE_ROWS`, else **422**), `temperature` (0.1–2.5), `random_seed`.
* Model must be `completed` (**409** otherwise). Missing, corrupt or incompatible weights/pipeline → **409** with the reason.
* `z ~ N(0, τ²I)` is decoded by the saved VAE. Output rows are **sampled from the model, not copied** from the original data.
* Inverse transform: numeric values are clipped to the training range and rounded to an integer (integer columns) or to the training decimals; **categorical values are sampled** in proportion to the decoder's per-category outputs, so category frequencies stay realistic.
* Output keeps the original column order and names; identifier/dropped columns are not generated.
* The same model, `random_seed` and `temperature` give identical rows, also across server restarts (tested with a separate process).
* The CSV, the `experiments` row and the `evaluations` row are created together; if anything fails, no row and no file remain.
* Response: `experiment_id` (`gen_…`), `dataset_id`, `model_id`, `output_filename`, `fidelity_score`, `domain_validity_percent`, 50 `preview_records`.

---

## 6. Evaluation (`app/services/evaluator.py`, `GET /api/evaluate/{experiment_id}`)

Everything is computed from the real and synthetic DataFrames at generation time and stored; nothing is a constant. A metric that cannot be computed is returned as `null` (never an invented value) and explained in `warnings`.

| Metric | Definition | Interpretation |
|---|---|---|
| **KS statistic** (per numeric column) | Two-sample Kolmogorov–Smirnov distance between the empirical CDFs, 0–1 | 0 = same distribution. Status: < 0.15 Passed, < 0.25 Warning, else Failed (these thresholds are a convention of this project). |
| **Wasserstein distance** | Earth-mover distance in the column's units | Scale-dependent; compare only within a column. |
| **Total variation distance, TVD** (per categorical column) | ½ Σ│p_real(c) − p_syn(c)│ over all categories | 0 = identical category proportions. Same status thresholds. Per-category proportions are returned. |
| **Correlation difference** | Pearson matrices of numeric columns that vary in both datasets. `correlation_mean_abs_delta` = mean │R_real − R_syn│ over distinct pairs; `correlation_frobenius_delta` = ‖R_real − R_syn‖_F / n | 0 = same pairwise linear relationships. `null` if fewer than two such columns. *Definition change:* earlier versions divided the Frobenius norm by n², which made differences look about n times smaller; stored values from older experiments keep the old definition. |
| **Validity** (`domain_validity_percent`) | % of generated cells that are present and, for numeric columns, finite and within the real [min, max], or, for categorical columns, an existing category | Schema consistency (exact columns, row count, no missing values) is enforced before evaluation; violations fail generation. |
| **Fidelity** (`overall_fidelity_score`, 0–100) | Weighted mean of the components that could be computed: numeric marginals 40 % (100·(1 − mean KS)), categorical marginals 20 % (100·(1 − mean TVD)), correlations 20 % (100·(1 − min(1, 5·mean │ΔR│))), validity 20 %; weights renormalised over available components. Components and weights are returned. | A **heuristic composite, not a universal quality score.** |

**Limitations — please read:**
* These are marginal (per-column) and pairwise statistics. They do not measure higher-order dependencies or usefulness for a downstream task (e.g. model accuracy trained on synthetic data).
* A low KS statistic means similar numeric *distributions*; it says nothing about whether individual records are plausible.
* **No privacy claim:** statistical similarity does not show that records cannot be linked to individuals, and nothing here tests whether the generator memorised training rows. Do not treat synthetic output as anonymised.
* With fewer than 30 rows the statistics are unreliable (a warning is returned).

---

## 7. API reference

Base path `/api`. Errors use `{"detail": "..."}`. Unexpected errors return a generic 500 message; details go to the server log only.

| Method | Route | Notes |
|---|---|---|
| GET | `/health` | `{status, message, version, deep_learning_engine}` (static text; does not probe the model) |
| POST | `/datasets/upload` | multipart `file` (.csv). 400 invalid/empty CSV, 413 too large. `uploadDate` is the stored timestamp, the same value the list and detail endpoints return |
| GET | `/datasets`, `/datasets/{id}` | schema + 15 / 50 preview rows (empty if the file is missing) |
| POST | `/preprocess` | dry run: validates a column configuration; returns layout, `dropped_columns`, `warnings` |
| POST | `/models/train` | §4 |
| GET | `/models/status/{id}`, `/models` | model state; additive fields `model_type, error_message, created_at, completed_at` |
| POST | `/generate` | §5. `random_seed` must be 0 to 4294967295 (otherwise 422) |
| GET | `/evaluate/{experiment_id}` | §6; additive fields `categorical_metrics, fidelity_components, fidelity_weights, correlation_mean_abs_delta, warnings, limitations` |
| GET | `/export/{experiment_id}` | `text/csv` attachment; 404 if the experiment or file is missing |
| GET | `/experiments` | history, newest first (ties within the same second ordered by insertion), with `dataset_name`, `model_name`, `model_type`, `random_seed`, `output_filename`, `output_available` and `status` (`completed`, or `output_missing` if the CSV was removed; failed generations are never stored). Server file paths are not returned. |

### Examples

```http
POST /api/models/train
{"dataset_id": "ds_student_01", "epochs": 100, "learning_rate": 0.003, "latent_dim": 8,
 "hidden_layers": [128, 64], "random_seed": 1}
→ 200 {"model_id": "vae_7e30743d", "dataset_id": "ds_student_01", "name": "VAE-student_performance-Z8",
       "status": "completed", "current_epoch": 100, "total_epochs": 100, "final_loss": 6.6525,
       "recon_loss": ..., "kl_loss": ..., "loss_history": [...], "model_type": "numpy-vae", "error_message": null, ...}
```
(`model_id`, `final_loss` and the figures below come from one real run on a copy of the sample data; they will differ between runs and datasets, and `...` marks values omitted here)

```http
POST /api/generate
{"model_id": "vae_7e30743d", "num_records": 500, "temperature": 1.0, "random_seed": 1}
→ 200 {"experiment_id": "gen_a1f8ade8", "dataset_id": "ds_student_01", "model_id": "vae_7e30743d",
       "num_records": 500, "output_filename": "synthetic_student_performance_gen_a1f8ade8.csv",
       "fidelity_score": 86.6, "domain_validity_percent": 100.0, "preview_records": [{...}, ...]}

GET /api/evaluate/gen_a1f8ade8
→ 200 {"overall_fidelity_score": 86.6, "domain_validity_percent": 100.0, "average_ks_statistic": 0.0792,
       "correlation_frobenius_delta": ..., "correlation_mean_abs_delta": 0.0956,
       "ks_metrics": {"study_hours": {"ks_statistic": ..., "p_value": ..., "wasserstein_distance": ..., "status": "Passed"}, ...},
       "categorical_metrics": {"gender": {"tvd": 0.008, "status": "Passed", "real_proportions": {...}, ...}},
       "fidelity_components": {"numeric": ..., "categorical": ..., "correlation": ..., "validity": 100.0},
       "warnings": [], "limitations": "..."}
```

Error examples: `422` `hidden_layers must list exactly two layer sizes…`; `409` `The model's weights file is missing on the server.`; `400` `Model 'vae_…' was trained on dataset 'ds_…', not 'ds_other'.`

---

## 8. Running the backend

```powershell
cd backend
py -3.12 -m pip install -r requirements.txt
py -3.12 -m uvicorn app.main:app --host 127.0.0.1 --port 8000     # must be started from backend/
# Swagger UI: http://127.0.0.1:8000/docs
```
`scikit-learn` is listed in `requirements.txt` but is not used by the code.

## 9. Tests

```powershell
cd backend
py -3.12 -m pip install -r requirements-dev.txt     # pytest, httpx, requests
py -3.12 -m pytest                                  # no server needed
```
* `tests/` (96 tests, including `test_e2e_server.py`, which starts a real uvicorn process on a temporary data directory, runs the whole workflow over HTTP, restarts the process and checks persistence) use a **temporary data directory and temporary SQLite file per test**, so the real `data/` and `dataforge.db` are never touched. They cover: valid/invalid/empty/oversized uploads; preprocessing with missing values, numeric and categorical columns, constants, invalid values, unseen categories; training success, invalid configuration, divergence and save failures (recorded as `failed`, no misleading success); weight/pipeline persistence, reload, corrupt and incompatible files; generation row counts, schema, types, seeds, temperature; generation from a **separate process** (restart); evaluation computed from real data, stored and re-read; CSV export and history; unknown IDs and missing artifacts; database consistency when generation or the database write fails; legacy rows with foreign absolute paths; schema migration of an old database.
* `test_backend_api.py` is a manual **smoke script** that needs a running server and *writes data* (a model, a generation, an upload). Point it at a scratch server: `DATAFORGE_API_URL=http://127.0.0.1:8001/api python test_backend_api.py`.

## 10. Known limitations / not done

* Training is synchronous (no live progress, no stop endpoint); long trainings block the request.
* Exactly two hidden layers; no other activation functions.
* Quality depends strongly on training length and learning rate and is only demonstrated on the small sample dataset and a synthetic test fixture; correlations between columns are the weakest part (mean │ΔR│ ≈ 0.1–0.3 depending on settings).
* No privacy evaluation (see §6). No authentication. CORS is limited to the origins in `DATAFORGE_CORS_ORIGINS` (default: the Vite dev and preview servers on `localhost`/`127.0.0.1`, ports 5173, 4173 and 3000); if the frontend runs on another port or host, add its origin to that variable.
* `/api/health` text is static. Foreign keys are not enforced. `.env` is not read by the backend (settings come from the environment variables in §3).
* Frontend pages still use mock data; wiring them to these endpoints is a later step.
