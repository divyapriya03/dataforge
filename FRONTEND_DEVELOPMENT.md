# DataForge: Deep Learning-Based Synthetic Data Generation
## Phase 4: Frontend Development & Module Integration Document
**Course / Subject:** AI23531 – Deep Learning  
**Project Title:** DataForge – Deep Learning-Based Synthetic Data Generation Platform  
**Document Version:** 1.0.0  
**Status:** Developed, Tested & Operational  

---

## 1. Overview & Architecture

The frontend for **DataForge** is a fully functional Single Page Application built with **React 18**, **TypeScript**, **Tailwind CSS**, and **Recharts**. It provides an interactive interface for the entire lifecycle of tabular synthetic data generation powered by Variational Autoencoders (VAEs).

The application is accessible locally at:
🌐 **http://127.0.0.1:5173/**

---

## 2. Completed Modules & Implementation Details

### Module 1: Dashboard (`/dashboard`)
* **Key Performance Indicator (KPI) Cards:**
  * Total Datasets uploaded
  * VAE Models trained
  * Synthetic records generated across runs
  * Overall Quality Fidelity Score (e.g., 93.8%) and Domain Validity (99.8%)
* **Active System State Banner:** Displays the current active dataset (`student_performance.csv`) and active model (`VAE-Student-Z8`).
* **Recent Experiment History Table:** Tabular feed of latest runs with timestamps, record counts, and one-click shortcuts to the Quality Lab or Export Vault.
* **Deep Learning Theory & Syllabus Card:** Mathematical breakdown of the Evidence Lower Bound (ELBO):
  $$\mathcal{L} = \text{MSE}(x, \hat{x}) + \beta \cdot \mathcal{D}_{\text{KL}}(q(z|x) \,||\, p(z))$$
  Explicitly mapped to AI23531 syllabus units (Unit I: MLP; Unit II: Adam/Epochs; Unit V: Generative VAE).

---

### Module 2: Dataset Studio (`/dataset-studio`)
* **Drag-and-Drop CSV Uploader:** Client-side CSV file parser capable of ingesting arbitrary tabular `.csv` files.
* **Feature Schema Profiling & Configuration Grid:**
  * Auto-detects Numerical vs. Categorical vs. Identifier columns.
  * Checkboxes to include or exclude specific features from model training.
  * Dropdown selector for feature role overrides (Numerical Min-Max vs. Categorical One-Hot).
  * Missing value count, percentage, and imputation strategy selector (Median, Mean, Mode, Drop).
* **Data Preview Table:** Sortable and scrollable view of the first 200 records.
* **Summary Statistics:** Descriptive statistics table displaying Min, Max, Mean, Std Dev, and scaled target ranges $[0.0, 1.0]$.

---

### Module 3: Model Lab (`/model-lab`)
* **Hyperparameter Configuration Form:**
  * Latent Bottleneck Dimension ($z \in [2, 32]$, default $8$).
  * Hidden Layer Architecture ($[128, 64]$, $[256, 128]$, or $[64, 32]$).
  * Learning Rate ($0.0001$ to $0.01$).
  * Batch Size ($16, 32, 64, 128$).
  * Epochs slider ($5$ to $200$).
  * $\beta$-VAE Regularization weight ($\beta \in [0.1, 5.0]$).
  * Activation functions (`LeakyReLU`, `ReLU`, `ELU`).
* **Live Interactive Training Engine:**
  * Animated progress bar, epoch counter, and percentage complete.
  * Real-time **Recharts LineChart** rendering 3 concurrent loss curves:
    1. **Total ELBO Loss**
    2. **Reconstruction Loss (MSE)**
    3. **KL Divergence ($\mathcal{D}_{\text{KL}}$)**
  * Start and interrupt controls; automatically registers trained models upon completion.

---

### Module 4: Data Generator (`/generator`)
* **Generation Controls:**
  * Model selector targeting trained VAE checkpoints.
  * Number of synthetic records slider/input ($100$ to $20,000$).
  * Latent sampling temperature ($\tau \in [0.2, 1.8]$) to modulate fidelity versus diversity.
  * Random seed input for deterministic reproducibility.
* **Instant Record Synthesis Engine:**
  * Samples Gaussian vectors $z \sim \mathcal{N}(0, \tau^2 I)$ and transforms them through the VAE Decoder.
  * Inverse Min-Max denormalization and domain constraint clipping.
* **Synthetic Output Preview Table:**
  * Paginated table showing synthesized records with domain check validation badges.

---

### Module 5: Quality Comparison & Diagnostics (`/quality`)
* **Fidelity Metric Cards:** Composite Data Fidelity Score, Domain Boundary Validity, and Average Kolmogorov-Smirnov (KS) statistic.
* **Feature Density Distribution Chart:** Interactive **Recharts BarChart** comparing real vs. synthetic probability densities across histogram bins for any selected feature.
* **Correlation Matrix Delta Heatmap:** Full matrix computing absolute difference $|\mathbf{R}_{\text{real}} - \mathbf{R}_{\text{synth}}|$, reporting average pairwise error and Frobenius norm distance ($0.041$).
* **Summary Statistics Side-by-Side Table:** Compares real vs. synthetic Mean, Std Dev, Mean Delta, two-sample KS statistic, and $p$-values.

---

### Module 6: Export & History (`/history`)
* **One-Click CSV Export:** Generates clean, downloadable `.csv` files matching original column names and formats directly in the user's browser.
* **JSON Audit Compliance Report:** Generates structured quality compliance reports documenting hyperparameters, loss history, and statistical hypothesis test results.
* **Historical Run Vault:** Searchable, filterable audit log of all generation sessions.

---

## 3. Reusable UI Components & State Management

* [src/components/Sidebar.tsx](file:///c:/Users/divya/OneDrive/Desktop/dl%20project/frontend/src/components/Sidebar.tsx): Global persistent navigation sidebar with status indicator and route switcher.
* [src/components/Navbar.tsx](file:///c:/Users/divya/OneDrive/Desktop/dl%20project/frontend/src/components/Navbar.tsx): Header displaying active dataset, active model, and backend ping trigger.
* [src/services/mockData.ts](file:///c:/Users/divya/OneDrive/Desktop/dl%20project/frontend/src/services/mockData.ts): Pre-loaded realistic sample datasets (`student_performance.csv`), historical training loss logs, and statistical benchmarks.
* [src/types/index.ts](file:///c:/Users/divya/OneDrive/Desktop/dl%20project/frontend/src/types/index.ts): Strict TypeScript type definitions for datasets, columns, hyperparameters, experiments, and metrics.
* [src/App.tsx](file:///c:/Users/divya/OneDrive/Desktop/dl%20project/frontend/src/App.tsx): Root container coordinating bidirectional state updates across all 6 modules.

---

## 4. Verification & Production Build
```
✓ 2383 modules transformed.
dist/index.html                   0.55 kB
dist/assets/index-0vk8kbee.css   23.25 kB
dist/assets/index-DbHHlm9x.js   604.25 kB
✓ built in 16.48s
HTTP 200 OK verified on http://127.0.0.1:5173/
```
