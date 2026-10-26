"""Statistical comparison of a real dataset and a synthetic dataset.

Every number is computed from the two DataFrames passed in; nothing is defaulted to a
"nice looking" constant. When a metric cannot be computed (for example correlations with fewer
than two usable numeric columns) it is reported as ``None`` and listed in ``warnings``.

Metric definitions (also in BACKEND_DEVELOPMENT.md)
---------------------------------------------------
* KS statistic (numeric column): two-sample Kolmogorov-Smirnov distance between the empirical
  CDFs, in [0, 1]; 0 = identical distributions. Status: < 0.15 Passed, < 0.25 Warning, else Failed.
* Wasserstein distance (numeric column): earth mover's distance in the column's own units.
* Total variation distance, TVD (categorical column): 0.5 * sum |p_real(c) - p_syn(c)| over all
  categories, in [0, 1]; 0 = identical category proportions. Same status thresholds as KS.
* Correlation difference: Pearson correlation matrices of the numeric columns that vary in both
  datasets; ``correlation_mean_abs_delta`` is the mean |R_real - R_syn| over distinct pairs and
  ``correlation_frobenius_delta`` is ||R_real - R_syn||_F / n (n = number of columns, i.e. the
  root-mean-square entry of the difference matrix).
* Validity (percent of cells): a cell is valid when it is present, a finite number inside the
  real data's [min, max] (numeric) or an existing category of the real data (categorical). Schema
  consistency (the exact expected columns) is checked by the generation endpoint before evaluation.
* Fidelity (0-100): a documented heuristic, NOT a universal quality measure. It is the weighted
  mean of the components that could be computed:
      numeric marginals    40 %  = 100 * (1 - mean KS)
      categorical marginals 20 %  = 100 * (1 - mean TVD)
      correlations         20 %  = 100 * (1 - min(1, 5 * mean |R_real - R_syn|))
      validity             20 %  = validity percent
  with weights renormalised over the available components.

Limitations: these are marginal and pairwise statistics. They do not measure higher-order
dependencies, do not capture downstream-task utility, and say nothing about privacy: low
distances do not show that records cannot be linked back to individuals or that the generator
has not memorised training rows.
"""
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

KS_PASS, KS_WARN = 0.15, 0.25
SMALL_SAMPLE = 30
COMPONENT_WEIGHTS = {"numeric": 0.4, "categorical": 0.2, "correlation": 0.2, "validity": 0.2}
LIMITATIONS = (
    "Fidelity is a heuristic composite of marginal (per-column) and pairwise (correlation) statistics. "
    "It does not measure higher-order structure or downstream usefulness, and it provides no privacy guarantee."
)


def _status(distance: float) -> str:
    return "Passed" if distance < KS_PASS else "Warning" if distance < KS_WARN else "Failed"


def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


class DataQualityEvaluator:
    @staticmethod
    def evaluate(real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> Dict[str, Any]:
        warnings: List[str] = []
        if len(real_df) == 0 or len(synthetic_df) == 0:
            raise ValueError("Both the real and the synthetic dataset must contain at least one row.")
        if min(len(real_df), len(synthetic_df)) < SMALL_SAMPLE:
            warnings.append(f"Small sample (fewer than {SMALL_SAMPLE} rows): statistics are unreliable.")

        # Columns the synthetic data is expected to contain = columns of the real data it was built from.
        expected = [c for c in synthetic_df.columns if c in real_df.columns]
        missing = [c for c in synthetic_df.columns if c not in real_df.columns]
        if not expected:
            raise ValueError("The real and synthetic datasets share no columns.")
        if missing:
            warnings.append(f"Synthetic columns not present in the real data were ignored: {missing}")

        numeric_cols = [c for c in expected if _is_numeric(real_df[c])]
        categorical_cols = [c for c in expected if c not in numeric_cols]

        ks_results: Dict[str, Dict[str, Any]] = {}
        summary_comparisons: List[Dict[str, Any]] = []
        categorical_metrics: Dict[str, Dict[str, Any]] = {}
        valid_cells = 0
        total_cells = 0
        constant_cols: List[str] = []

        # ---- numeric columns
        for col in numeric_cols:
            real = pd.to_numeric(real_df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            syn = pd.to_numeric(synthetic_df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            real_vals, syn_vals = real.dropna().to_numpy(), syn.dropna().to_numpy()

            total_cells += len(synthetic_df)
            if len(real_vals) == 0:
                warnings.append(f"Column '{col}' has no valid real values and was skipped.")
                continue
            r_min, r_max = float(real_vals.min()), float(real_vals.max())
            in_range = syn.notna() & (syn >= r_min - 1e-9) & (syn <= r_max + 1e-9)
            valid_cells += int(in_range.sum())

            if len(syn_vals) == 0:
                warnings.append(f"Column '{col}' has no valid synthetic values.")
                ks_stat, p_val, w_dist = 1.0, 0.0, float("nan")
            elif r_min == r_max:
                constant_cols.append(col)
                ks_stat = 0.0 if np.all(syn_vals == r_min) else 1.0
                p_val, w_dist = (1.0 if ks_stat == 0.0 else 0.0), float(abs(syn_vals.mean() - r_min))
            else:
                ks_stat, p_val = stats.ks_2samp(real_vals, syn_vals)
                w_dist = stats.wasserstein_distance(real_vals, syn_vals)

            ks_stat = float(round(ks_stat, 4))
            status = _status(ks_stat)
            ks_results[col] = {
                "ks_statistic": ks_stat,
                "p_value": float(round(p_val, 4)),
                "wasserstein_distance": None if np.isnan(w_dist) else float(round(w_dist, 4)),
                "status": status,
            }
            summary_comparisons.append({
                "feature": col,
                "realMean": float(round(real_vals.mean(), 2)),
                "syntheticMean": float(round(syn_vals.mean(), 2)) if len(syn_vals) else None,
                "realStd": float(round(real_vals.std(), 2)),
                "syntheticStd": float(round(syn_vals.std(), 2)) if len(syn_vals) else None,
                "ksStatistic": ks_stat,
                "pValue": float(round(p_val, 4)),
                "status": status,
            })

        if constant_cols:
            warnings.append(f"Constant real column(s) compared by exact match instead of a KS test: {constant_cols}")

        # ---- categorical columns
        for col in categorical_cols:
            real = real_df[col].dropna().astype(str)
            syn_all = synthetic_df[col]
            syn = syn_all.dropna().astype(str)
            total_cells += len(synthetic_df)
            if real.empty:
                warnings.append(f"Column '{col}' has no valid real values and was skipped.")
                continue
            real_cats = set(real.unique())
            valid_cells += int(syn.isin(real_cats).sum())

            p_real = real.value_counts(normalize=True)
            p_syn = syn.value_counts(normalize=True) if len(syn) else pd.Series(dtype=float)
            union = sorted(set(p_real.index) | set(p_syn.index))
            tvd = float(round(0.5 * sum(abs(p_real.get(c, 0.0) - p_syn.get(c, 0.0)) for c in union), 4))
            categorical_metrics[col] = {
                "tvd": tvd,
                "status": _status(tvd),
                "invalid_categories": sorted(set(syn.unique()) - real_cats),
                "real_proportions": {c: float(round(p_real.get(c, 0.0), 4)) for c in union},
                "synthetic_proportions": {c: float(round(p_syn.get(c, 0.0), 4)) for c in union},
            }

        # ---- correlations (numeric columns that vary in both datasets)
        corr_cols = [
            c for c in ks_results
            if c not in constant_cols and pd.to_numeric(synthetic_df[c], errors="coerce").nunique() > 1
        ]
        corr_frobenius: Optional[float] = None
        corr_mean_abs: Optional[float] = None
        if len(corr_cols) >= 2:
            real_corr = real_df[corr_cols].apply(pd.to_numeric, errors="coerce").corr().to_numpy()
            syn_corr = synthetic_df[corr_cols].apply(pd.to_numeric, errors="coerce").corr().to_numpy()
            diff = np.abs(np.nan_to_num(real_corr) - np.nan_to_num(syn_corr))
            iu = np.triu_indices(len(corr_cols), k=1)
            corr_mean_abs = float(round(diff[iu].mean(), 4))
            corr_frobenius = float(round(np.linalg.norm(diff) / len(corr_cols), 4))
        else:
            warnings.append("Correlation comparison skipped: fewer than two numeric columns vary in both datasets.")

        # ---- validity
        validity_pct = 100.0 if total_cells == 0 else float(round(100.0 * valid_cells / total_cells, 1))

        # ---- composite fidelity
        components: Dict[str, Optional[float]] = {
            "numeric": 100.0 * (1.0 - float(np.mean([v["ks_statistic"] for v in ks_results.values()]))) if ks_results else None,
            "categorical": 100.0 * (1.0 - float(np.mean([v["tvd"] for v in categorical_metrics.values()]))) if categorical_metrics else None,
            "correlation": 100.0 * (1.0 - min(1.0, 5.0 * corr_mean_abs)) if corr_mean_abs is not None else None,
            "validity": validity_pct if total_cells else None,
        }
        used = {k: v for k, v in components.items() if v is not None}
        if not used:
            raise ValueError("No metric could be computed for these datasets.")
        weight_sum = sum(COMPONENT_WEIGHTS[k] for k in used)
        overall = round(sum(COMPONENT_WEIGHTS[k] * v for k, v in used.items()) / weight_sum, 1)

        avg_ks = float(round(np.mean([v["ks_statistic"] for v in ks_results.values()]), 4)) if ks_results else None
        return {
            "overall_fidelity_score": overall,
            "domain_validity_percent": validity_pct,
            "average_ks_statistic": avg_ks,
            "correlation_frobenius_delta": corr_frobenius,
            "correlation_mean_abs_delta": corr_mean_abs,
            "ks_metrics": ks_results,
            "categorical_metrics": categorical_metrics,
            "summary_comparisons": summary_comparisons,
            "fidelity_components": {k: (None if v is None else float(round(v, 1))) for k, v in components.items()},
            "fidelity_weights": {k: COMPONENT_WEIGHTS[k] / weight_sum for k in used},
            "warnings": warnings,
            "limitations": LIMITATIONS,
        }
