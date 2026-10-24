import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Any, Tuple

class DataQualityEvaluator:
    """
    Evaluates synthetic data fidelity against original real data
    using statistical tests, correlation similarity, and domain constraint checks.
    """
    @staticmethod
    def evaluate(real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> Dict[str, Any]:
        common_cols = [c for c in real_df.columns if c in synthetic_df.columns]
        numeric_cols = [c for c in common_cols if pd.api.types.is_numeric_dtype(real_df[c]) and 'id' not in c.lower()]
        categorical_cols = [c for c in common_cols if c not in numeric_cols and 'id' not in c.lower()]

        ks_results = {}
        summary_comparisons = []
        domain_violations = 0
        total_checks = 0

        # 1. Column-wise Statistical Diagnostics
        for col in numeric_cols:
            real_vals = real_df[col].dropna().to_numpy()
            syn_vals = synthetic_df[col].dropna().to_numpy()

            if len(real_vals) > 0 and len(syn_vals) > 0:
                ks_stat, p_val = stats.ks_2samp(real_vals, syn_vals)
                w_dist = stats.wasserstein_distance(real_vals, syn_vals)

                r_min, r_max = float(real_vals.min()), float(real_vals.max())
                r_mean, r_std = float(round(real_vals.mean(), 2)), float(round(real_vals.std(), 2))
                s_mean, s_std = float(round(syn_vals.mean(), 2)), float(round(syn_vals.std(), 2))

                # Check boundary violations
                out_of_bounds = np.sum((syn_vals < r_min - 1e-4) | (syn_vals > r_max + 1e-4))
                domain_violations += int(out_of_bounds)
                total_checks += len(syn_vals)

                status = "Passed" if ks_stat < 0.15 else "Warning" if ks_stat < 0.25 else "Failed"

                ks_results[col] = {
                    "ks_statistic": float(round(ks_stat, 4)),
                    "p_value": float(round(p_val, 4)),
                    "wasserstein_distance": float(round(w_dist, 4)),
                    "status": status
                }

                summary_comparisons.append({
                    "feature": col,
                    "realMean": r_mean,
                    "syntheticMean": s_mean,
                    "realStd": r_std,
                    "syntheticStd": s_std,
                    "ksStatistic": float(round(ks_stat, 4)),
                    "pValue": float(round(p_val, 4)),
                    "status": status
                })

        # 2. Categorical Category Checks
        for col in categorical_cols:
            real_cats = set(real_df[col].dropna().unique())
            syn_cats = set(synthetic_df[col].dropna().unique())
            invalid_cats = syn_cats - real_cats
            if invalid_cats:
                domain_violations += len(invalid_cats)
            total_checks += len(syn_cats)

        domain_validity_pct = 100.0 if total_checks == 0 else round(max(0.0, 100.0 - (domain_violations / total_checks) * 100), 1)

        # 3. Correlation Matrix Delta
        corr_frobenius_delta = 0.04
        corr_matrix_delta_table = []
        if len(numeric_cols) >= 2:
            real_corr = real_df[numeric_cols].corr().fillna(0).to_numpy()
            syn_corr = synthetic_df[numeric_cols].corr().fillna(0).to_numpy()
            diff_corr = np.abs(real_corr - syn_corr)
            corr_frobenius_delta = float(round(np.linalg.norm(diff_corr) / (len(numeric_cols) ** 2), 4))

        # 4. Overall Composite Fidelity Score (0 to 100%)
        # Weighted combination of average KS-distance complement, correlation parity, and domain validity
        avg_ks = np.mean([v["ks_statistic"] for v in ks_results.values()]) if ks_results else 0.05
        ks_fidelity = max(0.0, 1.0 - avg_ks) * 100.0
        corr_fidelity = max(0.0, 1.0 - corr_frobenius_delta * 5) * 100.0

        overall_fidelity = round(0.40 * ks_fidelity + 0.40 * corr_fidelity + 0.20 * domain_validity_pct, 1)

        return {
            "overall_fidelity_score": overall_fidelity,
            "domain_validity_percent": domain_validity_pct,
            "average_ks_statistic": float(round(avg_ks, 4)),
            "correlation_frobenius_delta": corr_frobenius_delta,
            "ks_metrics": ks_results,
            "summary_comparisons": summary_comparisons
        }
