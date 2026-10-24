# DataForge: Deep Learning-Based Synthetic Data Generation
## Phase 1: Project Requirements & Planning Document
**Course / Subject:** AI23531 – Deep Learning  
**Project Title:** DataForge – Deep Learning-Based Synthetic Data Generation Platform  
**Document Version:** 1.0.0  
**Status:** Approved for Implementation  

---

## 1. Executive Summary & Problem Statement

### 1.1 Problem Statement
Modern machine learning, data science, software testing, and analytics rely heavily on access to rich, domain-specific tabular datasets. However, real-world data collection and sharing face acute constraints:
1. **Privacy & Regulatory Compliance:** Regulations (e.g., GDPR, HIPAA, DPDP Act) prohibit sharing sensitive records containing Personally Identifiable Information (PII) or financial/academic data.
2. **Data Scarcity & Class Imbalance:** Certain rare scenarios (e.g., specific student failure categories, edge cases in retail transactions) have insufficient samples for robust testing and training.
3. **Flaws of Traditional Mock Data Generators:** Naive mock data tools (random sampling, rule-based generators) create independent feature values that completely disregard complex multi-variable correlations, non-linear dependencies, and latent distributions present in real data.

### 1.2 Project Objectives
DataForge resolves these challenges by creating an end-to-end web platform powered by a deep generative model—specifically a **Variational Autoencoder (VAE)**. The system enables users to:
- Ingest arbitrary tabular CSV datasets (such as student academic performance or retail sales metrics).
- Automatically inspect, profile, and preprocess features (handling missing values, min-max scaling, and one-hot categorical encoding).
- Train a configurable deep Variational Autoencoder on numerical and categorical features.
- Sample from the learned continuous latent space to synthesize high-fidelity artificial records.
- Statistically evaluate and benchmark the synthetic data against the ground truth (distribution similarity, correlation matrix fidelity, domain validity).
- Export verified synthetic datasets and comprehensive quality reports.

---

## 2. Target Users & Use Cases

### 2.1 Target Users
| User Persona | Role & Needs | Value Delivered by DataForge |
| :--- | :--- | :--- |
| **Data Scientists / ML Practitioners** | Need augmented training data to address class imbalance without overfitting or leaking raw records. | Produces statistically representative synthetic records matching original feature covariance and distribution. |
| **QA & Software Engineers** | Require realistic edge-case testing data at scale without access to production databases. | Generates thousands of schema-compliant CSV records on-demand without manual mock scripts. |
| **Academic Researchers & Students** | Study deep generative modeling and empirical VAE latent space dynamics without complex hardware setups. | Offers visual interactive loss tracking, latent dimension controls, and statistical diagnostics. |
| **Data Privacy Officers / Compliance Leads** | Need sanitized proxies for downstream analytics without disclosing real identity records. | Generates artificial proxy records with similarity metrics and boundary validation. |

### 2.2 Core Use Cases
- **Use Case 1: Tabular Data Augmentation:** Expand a small dataset (e.g., 500 rows) into 5,000 synthetic rows that respect true covariance structures for ML training.
- **Use Case 2: Privacy-Preserving Dataset Sharing:** Replace confidential customer or student records with artificial counterparts having matching marginal and joint distributions.
- **Use Case 3: System Stress & Load Testing:** Generate arbitrary volume CSV files matching realistic production schema for load and benchmark testing.
- **Use Case 4: Educational VAE Diagnostics:** Inspect how bottleneck latent dimensionality, KL divergence weight ($\beta$), and training epochs impact generative quality.

---

## 3. Dataset Format & Preprocessing Specifications

### 3.1 Input Dataset Specifications
- **File Format:** Comma-Separated Values (`.csv`), UTF-8 encoded.
- **File Size Constraint:** Up to 25 MB (recommended row count: 100 to 50,000 rows for real-time web training).
- **Supported Column Types:**
  - **Numerical (Continuous / Discrete):** Integers, Floats (e.g., age, test scores, purchase amount).
  - **Categorical (Low-to-Medium Cardinality):** Strings, Booleans, Enums (e.g., gender, grade tier, department, payment method).
- **Benchmark Sample Datasets:**
  - *Primary:* Student Performance Dataset (`MathScore`, `ReadingScore`, `WritingScore`, `StudyHours`, `AttendanceRate`, `GradeLevel`).
  - *Secondary:* Retail Sales / Customer Transactions (`Age`, `AnnualIncome`, `SpendingScore`, `PurchaseFrequency`, `MembershipTier`).

### 3.2 Preprocessing Pipeline
```
Raw CSV Upload
      │
      ▼
1. Type Inference & Schema Detection
   - Identify continuous numerical vs. categorical columns
   - Flag unique identifiers (e.g., ID, UUID) for automatic exclusion
      │
      ▼
2. Missing Value Imputation
   - Numerical: Median or Mean imputation
   - Categorical: Mode imputation or "Unknown" category
      │
      ▼
3. Feature Transformation
   - Numerical: Min-Max Normalization to [0, 1] (or Standardization)
   - Categorical: One-Hot Encoding (with recorded inverse mapping dictionaries)
      │
      ▼
4. Model Tensor Assembly
   - Concatenated normalized vector input dimension: D_in
```

### 3.3 Post-Processing & Inverse Transformation (Generation Phase)
- Synthetic continuous outputs generated in $[0, 1]$ are denormalized using saved min/max scalers:
  $$x_{\text{orig}} = \hat{x} \times (x_{\max} - x_{\min}) + x_{\min}$$
- Synthetic one-hot logits are decoded using `argmax` and mapped back to categorical string labels.
- Domain boundaries (e.g., age $> 0$, percentages between $0$ and $100$) are verified and clipped if necessary.

---

## 4. Finalized Six Website Modules

```
┌────────────────────────────────────────────────────────────────────────┐
│                        DATAFORGE PLATFORM ARCHITECTURE                 │
└────────────────────────────────────────────────────────────────────────┘
  [1. Dashboard]          ── Overview stats, active model status, recent runs
  [2. Dataset Studio]     ── Upload CSV, schema profiling, missing values & cleanup
  [3. Model Lab]          ── VAE hyperparameter tuning, live loss curves (ELBO, Recon, KL)
  [4. Data Generator]     ── Latent sampling, record count selection, instant preview
  [5. Quality Comparison] ── Histogram comparisons, correlation diff, KS-test, fidelity score
  [6. Export & History]   ── CSV download, audit metadata, run logs & experiment history
```

### Module 1: Dashboard
- **Key Metrics Overview:** Total datasets uploaded, active models trained, cumulative synthetic rows generated, average similarity fidelity score.
- **Active Model Status Badge:** Displays current model architecture, latent space dimensions, and convergence status.
- **Recent Experiments Feed:** Table of latest generation runs with timestamps, source datasets, record counts, and quality badges.
- **Quick Action Bar:** Direct shortcuts to Upload CSV, Launch Training, or Generate Data.

### Module 2: Dataset Studio
- **Drag-and-Drop CSV Uploader:** Client-side validation for file size, MIME type, and structure.
- **Interactive Data Grid:** Paginated, sortable view of the first 100 rows.
- **Summary Statistics Card:** Column-wise statistics (Mean, Std Dev, Min, 25%, 50%, 75%, Max, Missing Count).
- **Feature Selection & Configuration:**
  - Checkboxes to include/exclude features.
  - Dropdown to override detected feature types (Numerical vs. Categorical).
  - Missing value strategy picker (Drop, Mean, Median, Mode).

### Module 3: Model Lab
- **Hyperparameter Controls:**
  - Latent Bottleneck Dimensions ($z \in \{2, 4, 8, 16, 32\}$).
  - Hidden Layer Architecture (e.g., Encoder: $[128, 64] \to z \to$ Decoder: $[64, 128]$).
  - Activation Functions: ReLU, LeakyReLU, ELU.
  - Training Settings: Learning Rate ($10^{-4}$ to $10^{-2}$), Batch Size (16, 32, 64, 128), Epochs (10 to 200).
  - Loss Weighting: $\beta$ parameter for $\beta$-VAE regularization balancing reconstruction fidelity and latent smoothness.
- **Real-Time Training Monitor:**
  - Animated progress bar and epoch counter.
  - Dual line charts (Recharts) plotting **Total Loss**, **Reconstruction Loss (MSE/BCE)**, and **KL Divergence** over epochs.
  - Stop/Pause training controls.

### Module 4: Data Generator
- **Generation Parameters:**
  - Number of synthetic records slider / number input (1 to 20,000).
  - Latent sampling temperature ($\tau \in [0.1, 2.0]$) to adjust diversity vs. fidelity.
  - Random seed selection for deterministic reproducibility.
- **Instant Generation Engine:** Latent vector sampling $z \sim \mathcal{N}(0, I)$ fed into the trained Decoder network.
- **Live Output Table:** Paginated preview of freshly synthesized records with highlight toggles for out-of-distribution values.

### Module 5: Quality Comparison & Diagnostics
- **Statistical Parity Matrix:** Side-by-side comparison table of Original vs. Synthetic descriptive statistics (Mean, Median, Standard Deviation, Skewness).
- **Distribution Comparison Histograms:** Overlaid histogram charts comparing real vs. synthetic feature probability densities.
- **Correlation Difference Heatmap:** Pearson correlation matrices for original vs. synthetic features, alongside a correlation delta matrix ($|\mathbf{R}_{\text{orig}} - \mathbf{R}_{\text{synth}}|$).
- **Domain Integrity Checks:** Verification of data integrity (null counts, data types, boundary/domain constraint compliance).
- **Overall Data Fidelity Score (0 - 100%):** Weighted composite metric combining statistical distance and correlation preservation.

### Module 6: Export & Experiment History
- **Data Export Options:** Download full synthetic dataset as clean `.csv` with customizable naming.
- **Quality Audit Report:** Download summary report (JSON / PDF) documenting hyperparameters, loss metrics, and fidelity scores.
- **Experiment History Vault:** SQLite-backed audit log of all historical training and generation sessions, with options to reload previous model configurations and review previous runs.

---

## 5. Deep Learning Architecture Specification

### 5.1 Variational Autoencoder (VAE) Formulation
A standard Autoencoder suffers from discontinuous latent spaces where unmapped regions produce nonsensical synthetic outputs. DataForge utilizes a **Variational Autoencoder (VAE)** which structures the latent space as a continuous Gaussian distribution:

```
Input X (D_in)
      │
  [Dense 128 + BatchNorm + LeakyReLU]
      │
  [Dense 64 + LeakyReLU]
      ├──► Dense μ (Latent Mean)
      └──► Dense log(σ²) (Latent Log-Variance)
                │
                ▼
      Reparameterization Trick: z = μ + σ ⊙ ε,   where ε ~ N(0, I)
                │
                ▼
  [Dense 64 + LeakyReLU]
      │
  [Dense 128 + BatchNorm + LeakyReLU]
      │
  [Dense D_in + Sigmoid]  ──► Reconstructed X_hat
```

### 5.2 Objective Function (ELBO Loss)
The VAE optimizes the Evidence Lower Bound (ELBO):
$$\mathcal{L}_{\text{total}}(\theta, \phi; x) = \mathcal{L}_{\text{recon}}(x, \hat{x}) + \beta \cdot \mathcal{D}_{\text{KL}}(q_\phi(z|x) \,||\, p(z))$$

1. **Reconstruction Loss ($\mathcal{L}_{\text{recon}}$):**
   - For normalized continuous features: Mean Squared Error (MSE) or Binary Cross-Entropy (BCE).
   $$\text{MSE}(x, \hat{x}) = \frac{1}{N} \sum_{i=1}^N (x_i - \hat{x}_i)^2$$
2. **Kullback-Leibler (KL) Divergence ($\mathcal{D}_{\text{KL}}$):**
   - Regularizes the latent space distribution $q_\phi(z|x) = \mathcal{N}(\mu, \sigma^2)$ against standard normal prior $p(z) = \mathcal{N}(0, I)$:
   $$\mathcal{D}_{\text{KL}} = -\frac{1}{2} \sum_{j=1}^d \left( 1 + \log(\sigma_j^2) - \mu_j^2 - \sigma_j^2 \right)$$

---

## 6. Evaluation Criteria & Project Limitations

### 6.1 Evaluation Criteria
| Metric / Criteria | Method & Formula | Success Threshold |
| :--- | :--- | :--- |
| **Statistical Distribution Similarity** | Kolmogorov-Smirnov (KS) test statistic & Wasserstein distance between real and synthetic continuous columns | Average KS-statistic $< 0.15$; Wasserstein distance minimized |
| **Correlation Preservation** | Frobenius norm difference between real and synthetic Pearson correlation matrices: $\|\mathbf{R}_{\text{real}} - \mathbf{R}_{\text{syn}}\|_F$ | Average pairwise correlation delta $< 0.12$ |
| **Convergence & Latent Regularity** | Training and validation loss stabilization; healthy KL divergence balance (no KL collapse where $\text{KL} \approx 0$) | Steady monotonic convergence across 50 epochs |
| **Domain Constraint Adherence** | Percentage of generated rows satisfying original feature bounds (e.g., $x_{\min} \le x_{\text{gen}} \le x_{\max}$ and valid categorical categories) | $\ge 99\%$ adherence |
| **Downstream Machine Learning Utility (TSTR)** | Train on Synthetic, Test on Real: Train a baseline regressor/classifier on synthetic data and evaluate $R^2$/Accuracy on real test set | Synthetic model performance within $85\text{--}95\%$ of model trained directly on real data |

### 6.2 Project Limitations
1. **Multimodal Complex Interactions:** Basic MLP-based VAEs may smooth out highly multimodal or clustered distributions compared to high-capacity diffusion models or GANs.
2. **Categorical Feature Imbalance:** High-cardinality categorical variables (e.g., postal codes, unique IDs) expand the one-hot dimension significantly, leading to sparse latent learning.
3. **Small Dataset Overfitting:** If the uploaded CSV has fewer than 100 samples, the VAE may memorize training records rather than generalizing the underlying probability density.
4. **Privacy vs. Fidelity Tradeoff:** Standard VAEs do not enforce formal differential privacy ($\epsilon, \delta$-DP). A synthetic record may occasionally resemble an original training record unless Differential Privacy SGD (DP-SGD) or explicit distance-to-closest-record filtering is applied.
5. **No Time-Series / Sequential Modeling in Baseline:** Assumes independent and identically distributed (i.i.d.) tabular records; temporal sequences require RNN/LSTM VAE extensions.

---

## 7. Technology Stack & Architectural Decisions

| Layer | Chosen Technology | Rationale |
| :--- | :--- | :--- |
| **Frontend Framework** | React 18 + TypeScript + Vite | Type safety, rapid reactivity, component modularity. |
| **UI & Styling** | Tailwind CSS + Lucide Icons | Clean, modern dashboard layout; responsive data tables. |
| **Data Visualizations** | Recharts | Smooth SVG rendering for loss curves, distribution histograms, and heatmaps. |
| **Backend API** | FastAPI (Python 3.10+) | High performance asynchronous endpoints, automatic Swagger UI documentation, native NumPy/Pandas integration. |
| **Deep Learning Framework** | PyTorch / TensorFlow (Keras) | GPU/CPU portability, custom VAE layer implementation, clear loss gradient monitoring. |
| **Data Processing** | Pandas, NumPy, Scikit-learn | Fast tabular manipulation, scaling, encoding, and statistical metrics (KS-test, correlation). |
| **Persistence / Storage** | SQLite + SQLAlchemy | Lightweight embedded metadata storage for runs, experiments, and model histories without external DB overhead. |

---

## 8. Academic Syllabus Alignment (AI23531 – Deep Learning)

| Syllabus Unit | Course Concept | Direct Implementation in DataForge |
| :--- | :--- | :--- |
| **Unit I: Neural Networks** | Multilayer Perceptrons, Activations (ReLU, Sigmoid), Backpropagation, Gradient Descent | Encoder and Decoder dense architectures with customized activation and normalization layers. |
| **Unit II: Training Techniques** | Optimization algorithms (Adam, RMSProp), Epochs, Batch sizes, Regularization, Validation split | Live hyperparameter tuning in Model Lab; learning rate scheduling; batch training over epochs. |
| **Unit III: CNN** | Convolutional operations | Not required for 1D tabular arrays; acknowledged as theoretical contrast to dense generative nets. |
| **Unit IV: RNN / LSTM** | Sequence modeling | Optional future extension for sequential time-series tabular data. |
| **Unit V: Generative Deep Learning** | Autoencoders, Variational Autoencoders, Latent Space Sampling, ELBO, Generative Modeling | **Core engine of DataForge:** Latent probabilistic encoder, reparameterization trick, sampling $z \sim \mathcal{N}(0, I)$, decoder generation. |

---

## 9. Implementation Roadmap & Milestones

- [x] **Phase 1: Requirement Analysis & Planning (Current)**
  - Problem statement, objectives, target users, and use cases defined.
  - CSV dataset format, preprocessing rules, and 6 core modules finalized.
  - Evaluation criteria, mathematical formulation, and limitations documented.
- [ ] **Phase 2: Data Preprocessing & Pipeline Architecture**
  - Schema detector, missing value handling, one-hot encoder, and min-max scaler modules.
- [ ] **Phase 3: Deep Generative Model Implementation (VAE)**
  - PyTorch/Keras VAE architecture, ELBO loss function, reparameterization trick, and training harness.
- [ ] **Phase 4: Backend API Development (FastAPI)**
  - Endpoints for upload, profiling, training, synthetic generation, evaluation metrics, and CSV export.
- [ ] **Phase 5: Frontend Dashboard & Interactive Modules (React + TypeScript)**
  - Complete 6-module UI with live Recharts loss curves, data tables, and quality comparison charts.
- [ ] **Phase 6: Verification, Benchmarking & Final Documentation**
  - End-to-end testing with sample datasets (Student Performance & Retail Sales), fidelity validation, and final report.
