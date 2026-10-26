"""CSV validation, schema inference and the fitted preprocessing pipeline.

Workflow
--------
1. ``inspect_csv``   - validate a CSV and infer a per-column schema (type, nulls, imputation).
2. ``fit``           - learn scalers / category lists / fill values from a DataFrame and a
                       column configuration. ``fit_transform`` = ``fit`` + ``transform``.
3. ``save``/``load`` - persist the fitted state as JSON next to the model so the exact same
                       transformation is reused later (generation never re-fits).
4. ``inverse_transform`` - map model output in [0, 1] back to readable columns.

Behaviour that changes the data is recorded in ``warnings`` / ``dropped_columns`` (and saved
with the pipeline) instead of happening silently.
"""
import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.core.config import MAX_CATEGORIES

PIPELINE_VERSION = 2
VALID_TYPES = ("numeric", "categorical", "drop")
VALID_IMPUTATION = ("median", "mean", "mode", "drop")
MAX_DECIMALS = 6


class PreprocessingError(ValueError):
    """The dataset or column configuration cannot be used; the message is safe to show to users."""


def _is_id_name(name: str) -> bool:
    tokens = [t for t in re.split(r"[^a-z0-9]+", name.lower()) if t]
    if any(t in ("id", "uuid", "guid") for t in tokens):
        return True
    return bool(re.search(r"[a-z0-9](Id|ID)$", name))  # camelCase, e.g. userId


def _finite_or_none(value: Any, digits: int = 2) -> Optional[float]:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return round(value, digits) if np.isfinite(value) else None


class DataPreprocessor:
    def __init__(self):
        self.numeric_scalers: Dict[str, Dict[str, float]] = {}  # col -> {min, max}
        self.categorical_encoders: Dict[str, List[str]] = {}    # col -> sorted categories
        self.column_order: List[str] = []
        self.processed_feature_names: List[str] = []
        self.feature_slices: Dict[str, Any] = {}                # col -> (start, end, type)
        self.fill_values: Dict[str, Any] = {}                   # col -> value used for missing data
        self.numeric_dtypes: Dict[str, str] = {}                # col -> "int" | "float"
        self.numeric_decimals: Dict[str, int] = {}              # col -> decimals kept for floats
        self.dropped_columns: List[Dict[str, str]] = []         # [{name, reason}]
        self.warnings: List[str] = []
        self.n_rows_fit: int = 0
        self.fitted_at: Optional[str] = None
        self.unseen_category_count: int = 0                     # set by the last transform() call

    # ------------------------------------------------------------------ inspection
    @staticmethod
    def read_csv(file_path: str) -> pd.DataFrame:
        """Read a CSV and raise ``PreprocessingError`` with a clear message when it is unusable."""
        try:
            df = pd.read_csv(file_path)
        except pd.errors.EmptyDataError:
            raise PreprocessingError("The file is empty: no header or data found.")
        except (pd.errors.ParserError, UnicodeDecodeError) as e:
            raise PreprocessingError(f"The file is not a valid CSV: {e}")
        if df.shape[1] == 0:
            raise PreprocessingError("The CSV has no columns.")
        if len(df) == 0:
            raise PreprocessingError("The CSV has a header but no data rows.")
        if df.shape[1] == 1 and any(ch in str(df.columns[0]) for ch in (";", "	", "|")):
            raise PreprocessingError(
                "The file has a single column whose header contains ';', a tab or '|'. Only comma-separated CSV files "
                "are supported; export the file again with commas as the delimiter."
            )
        return df

    @staticmethod
    def inspect_csv(file_path: str) -> Tuple[int, int, List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Validate the CSV and return (row count, column count, schema metadata, preview rows).

        Type inference: bool columns and non-numeric columns are categorical; numeric columns are
        numeric. A column is proposed as an identifier (excluded by default) when its name contains
        an ``id`` token (``student_id``, ``userId``) or, with at least 20 rows, when every value is
        unique and the column is not a float. Entirely empty columns are excluded. All of these are
        only defaults: the caller can override them through the column configuration.
        """
        df = DataPreprocessor.read_csv(file_path)
        row_count = len(df)
        col_count = len(df.columns)

        schema = []
        for col in df.columns:
            col = str(col) if not isinstance(col, str) else col
            series = df[col]
            null_count = int(series.isnull().sum())
            null_pct = round((null_count / max(row_count, 1)) * 100, 1)
            all_null = null_count == row_count
            is_numeric_dtype = pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)
            is_float = pd.api.types.is_float_dtype(series)
            unique_id = (
                row_count >= 20 and not is_float and series.nunique(dropna=True) == row_count
            )
            is_id = _is_id_name(col) or unique_id
            exclude = is_id or all_null
            is_numeric = is_numeric_dtype and not is_id

            meta: Dict[str, Any] = {
                "name": col,
                "inferredType": "id" if is_id else "numeric" if is_numeric else "categorical",
                "selectedType": "drop" if exclude else "numeric" if is_numeric else "categorical",
                "nullCount": null_count,
                "nullPercentage": null_pct,
                "imputationStrategy": "drop" if exclude else "median" if is_numeric else "mode",
                "included": not exclude,
            }

            if is_numeric:
                clean = series.replace([np.inf, -np.inf], np.nan).dropna()
                if len(clean) > 0:
                    meta["min"] = _finite_or_none(clean.min())
                    meta["max"] = _finite_or_none(clean.max())
                    meta["mean"] = _finite_or_none(clean.mean())
                    meta["std"] = _finite_or_none(clean.std())
            else:
                meta["categories"] = [str(c) for c in series.dropna().unique().tolist()[:20]]

            schema.append(meta)

        preview = df.head(50).fillna("").to_dict(orient="records")
        return row_count, col_count, schema, preview

    # ------------------------------------------------------------------ fitting
    @staticmethod
    def _validate_configs(df: pd.DataFrame, column_configs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        config_map: Dict[str, Dict[str, Any]] = {}
        unknown = []
        for cfg in column_configs:
            name = cfg.get("name")
            if name not in df.columns:
                unknown.append(str(name))
                continue
            sel = cfg.get("selectedType", cfg.get("inferredType", "numeric"))
            if sel not in VALID_TYPES:
                raise PreprocessingError(
                    f"Column '{name}': selectedType must be one of {list(VALID_TYPES)}, got '{sel}'."
                )
            imp = cfg.get("imputationStrategy", "median")
            if imp not in VALID_IMPUTATION:
                raise PreprocessingError(
                    f"Column '{name}': imputationStrategy must be one of {list(VALID_IMPUTATION)}, got '{imp}'."
                )
            config_map[name] = cfg
        if unknown:
            raise PreprocessingError(f"Columns not found in the dataset: {unknown}")
        return config_map

    @staticmethod
    def _to_numeric(series: pd.Series) -> Tuple[pd.Series, int]:
        """Coerce to float; return (series with NaN for missing/invalid/inf, count of invalid entries)."""
        numeric = pd.to_numeric(series, errors="coerce").astype("float64")
        numeric = numeric.replace([np.inf, -np.inf], np.nan)
        invalid = int((numeric.isna() & series.notna()).sum())
        return numeric, invalid

    def fit(self, df: pd.DataFrame, column_configs: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Learn the pipeline from ``df``. Returns the DataFrame actually fitted on (rows removed by a
        ``drop`` imputation strategy are excluded). Everything non-trivial is reported in ``warnings``.
        """
        self.numeric_scalers, self.categorical_encoders = {}, {}
        self.column_order, self.processed_feature_names, self.feature_slices = [], [], {}
        self.fill_values, self.numeric_dtypes, self.numeric_decimals = {}, {}, {}
        self.dropped_columns, self.warnings = [], []

        config_map = self._validate_configs(df, column_configs)

        active: List[str] = []
        for col in df.columns:
            cfg = config_map.get(col)
            if cfg is None:
                self.dropped_columns.append({"name": col, "reason": "no column configuration was supplied"})
            elif not cfg.get("included", True) or cfg.get("selectedType") == "drop":
                self.dropped_columns.append({"name": col, "reason": "excluded by configuration"})
            else:
                active.append(col)
        if not active:
            raise PreprocessingError("No usable columns selected: every column is excluded or dropped.")

        # Rows with a missing value in a column whose strategy is 'drop' are removed before fitting.
        keep = pd.Series(True, index=df.index)
        coerced: Dict[str, pd.Series] = {}
        for col in active:
            cfg = config_map[col]
            if cfg.get("selectedType", cfg.get("inferredType")) == "numeric":
                coerced[col], invalid = self._to_numeric(df[col])
                present = df[col].notna().sum()
                if present == 0 or coerced[col].notna().sum() == 0:
                    raise PreprocessingError(f"Column '{col}' has no valid numeric values; exclude it or mark it categorical.")
                if invalid > 0.5 * present:
                    raise PreprocessingError(
                        f"Column '{col}' is marked numeric but {invalid} of {present} values are not numbers; "
                        "mark it categorical or exclude it."
                    )
                if invalid:
                    self.warnings.append(f"Column '{col}': {invalid} non-numeric/infinite value(s) treated as missing.")
                missing = coerced[col].isna()
            else:
                missing = df[col].isna()
            if cfg.get("imputationStrategy") == "drop" and missing.any():
                keep &= ~missing
        n_dropped_rows = int((~keep).sum())
        if n_dropped_rows:
            self.warnings.append(f"{n_dropped_rows} row(s) with missing values removed (imputationStrategy='drop').")
        if keep.sum() == 0:
            raise PreprocessingError("No rows remain after removing rows with missing values.")
        fit_df = df.loc[keep]
        self.n_rows_fit = int(len(fit_df))

        current = 0
        for col in active:
            cfg = config_map[col]
            sel = cfg.get("selectedType", cfg.get("inferredType"))
            strat = cfg.get("imputationStrategy", "median")
            self.column_order.append(col)

            if sel == "numeric":
                values = coerced[col].loc[keep]
                valid = values.dropna()
                if strat == "mean":
                    fill = float(valid.mean())
                elif strat == "mode":
                    fill = float(valid.mode().iloc[0])
                else:  # median, or drop (no missing values remain)
                    fill = float(valid.median())
                filled = values.fillna(fill)
                col_min, col_max = float(filled.min()), float(filled.max())
                if col_min == col_max:
                    self.warnings.append(f"Column '{col}' is constant ({col_min}); it will be reproduced as a constant.")
                is_int = bool(np.all(np.isclose(valid, np.rint(valid))))
                decimals = 0
                if not is_int:
                    decimals = MAX_DECIMALS
                    for d in range(1, MAX_DECIMALS + 1):
                        if np.allclose(valid, np.round(valid, d), atol=1e-9, rtol=0):
                            decimals = d
                            break
                self.numeric_scalers[col] = {"min": col_min, "max": col_max}
                self.fill_values[col] = fill
                self.numeric_dtypes[col] = "int" if is_int else "float"
                self.numeric_decimals[col] = decimals
                self.feature_slices[col] = (current, current + 1, "numeric")
                self.processed_feature_names.append(col)
                current += 1

            else:  # categorical
                raw = fit_df[col]
                present = raw.dropna().astype(str)
                if present.empty:
                    raise PreprocessingError(f"Column '{col}' has no values; exclude it.")
                fill = present.mode().iloc[0]
                categories = sorted(present.unique().tolist())
                if len(categories) > MAX_CATEGORIES:
                    raise PreprocessingError(
                        f"Column '{col}' has {len(categories)} distinct values (limit {MAX_CATEGORIES} for categorical "
                        "columns); exclude it or mark it as an identifier."
                    )
                if len(categories) == 1:
                    self.warnings.append(f"Column '{col}' has a single category ('{categories[0]}').")
                self.categorical_encoders[col] = categories
                self.fill_values[col] = fill
                self.feature_slices[col] = (current, current + len(categories), "categorical")
                self.processed_feature_names.extend(f"{col}_{c}" for c in categories)
                current += len(categories)

        self.fitted_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return fit_df

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Apply the already-fitted pipeline to ``df`` (never re-fits). Missing values use the fitted fill
        value, numeric values outside the fitted range are clipped to [0, 1] and categories not seen
        during fitting become an all-zero one-hot block (count in ``unseen_category_count``).
        """
        if not self.column_order:
            raise PreprocessingError("The preprocessing pipeline has not been fitted.")
        missing_cols = [c for c in self.column_order if c not in df.columns]
        if missing_cols:
            raise PreprocessingError(f"Columns required by the pipeline are missing: {missing_cols}")

        parts = []
        self.unseen_category_count = 0
        for col in self.column_order:
            _, _, ftype = self.feature_slices[col]
            if ftype == "numeric":
                values, _ = self._to_numeric(df[col])
                values = values.fillna(self.fill_values[col])
                lo, hi = self.numeric_scalers[col]["min"], self.numeric_scalers[col]["max"]
                denom = hi - lo if hi > lo else 1.0
                scaled = np.clip((values - lo) / denom, 0.0, 1.0)
                parts.append(scaled.to_numpy(dtype=np.float32).reshape(-1, 1))
            else:
                categories = self.categorical_encoders[col]
                index = {c: i for i, c in enumerate(categories)}
                values = df[col].astype(object).where(df[col].notna(), self.fill_values[col]).astype(str)
                one_hot = np.zeros((len(values), len(categories)), dtype=np.float32)
                for row_i, val in enumerate(values):
                    idx = index.get(val)
                    if idx is None:
                        self.unseen_category_count += 1
                    else:
                        one_hot[row_i, idx] = 1.0
                parts.append(one_hot)
        return np.hstack(parts)

    def fit_transform(self, df: pd.DataFrame, column_configs: List[Dict[str, Any]]) -> np.ndarray:
        """Fit on ``df`` and return the normalised float32 matrix (rows x processed features)."""
        fit_df = self.fit(df, column_configs)
        return self.transform(fit_df)

    # ------------------------------------------------------------------ inverse
    def inverse_transform(self, normalized_data: np.ndarray, rng: Optional[np.random.Generator] = None) -> pd.DataFrame:
        """
        Convert model output in [0, 1] back to the original columns.

        Numeric: de-normalise, clip to the fitted range, then round to an integer (integer columns)
        or to the number of decimals seen in the training data. Categorical: when ``rng`` is given a
        category is *sampled* in proportion to the decoder's per-category outputs (this keeps category
        frequencies realistic); without ``rng`` the highest-scoring category is chosen.
        """
        expected = len(self.processed_feature_names)
        if normalized_data.ndim != 2 or normalized_data.shape[1] != expected:
            raise PreprocessingError(
                f"Model output has {normalized_data.shape[-1] if normalized_data.ndim == 2 else '?'} features, "
                f"pipeline expects {expected}."
            )
        n = normalized_data.shape[0]
        output: Dict[str, Any] = {}
        for col in self.column_order:
            start, end, ftype = self.feature_slices[col]
            block = normalized_data[:, start:end]
            if ftype == "numeric":
                lo, hi = self.numeric_scalers[col]["min"], self.numeric_scalers[col]["max"]
                values = np.clip(block.reshape(-1).astype(np.float64) * (hi - lo) + lo, lo, hi)
                if self.numeric_dtypes.get(col) == "int":
                    output[col] = np.rint(values).astype(np.int64)
                else:
                    output[col] = np.round(values, self.numeric_decimals.get(col, 1))
            else:
                categories = np.asarray(self.categorical_encoders[col], dtype=object)
                if rng is None:
                    idx = np.argmax(block, axis=1)
                else:
                    probs = np.clip(block.astype(np.float64), 0.0, None)
                    totals = probs.sum(axis=1, keepdims=True)
                    probs = np.where(totals > 0, probs / np.where(totals > 0, totals, 1.0), 1.0 / probs.shape[1])
                    cum = np.cumsum(probs, axis=1)
                    idx = np.minimum((rng.random((n, 1)) > cum).sum(axis=1), len(categories) - 1)
                output[col] = categories[idx].tolist()
        return pd.DataFrame(output, columns=self.column_order)

    # ------------------------------------------------------------------ persistence
    def to_state(self) -> Dict[str, Any]:
        return {
            "version": PIPELINE_VERSION,
            "fitted_at": self.fitted_at,
            "n_rows_fit": self.n_rows_fit,
            "numeric_scalers": self.numeric_scalers,
            "categorical_encoders": self.categorical_encoders,
            "column_order": self.column_order,
            "processed_feature_names": self.processed_feature_names,
            "feature_slices": {k: list(v) for k, v in self.feature_slices.items()},
            "fill_values": self.fill_values,
            "numeric_dtypes": self.numeric_dtypes,
            "numeric_decimals": self.numeric_decimals,
            "dropped_columns": self.dropped_columns,
            "warnings": self.warnings,
        }

    def save(self, file_path: str):
        """Write the fitted pipeline as JSON (atomically)."""
        tmp = file_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.to_state(), f, indent=2)
        os.replace(tmp, file_path)

    def load(self, file_path: str):
        """Load a pipeline written by ``save`` (including version-1 files without dtype metadata)."""
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                state = json.load(f)
            except json.JSONDecodeError as e:
                raise PreprocessingError(f"Preprocessing file is corrupt: {e}")
        try:
            self.numeric_scalers = state["numeric_scalers"]
            self.categorical_encoders = state["categorical_encoders"]
            self.column_order = state["column_order"]
            self.processed_feature_names = state["processed_feature_names"]
            self.feature_slices = {k: tuple(v) for k, v in state["feature_slices"].items()}
        except (KeyError, TypeError, AttributeError) as e:
            raise PreprocessingError(f"Preprocessing file is incompatible (missing {e}).")
        # v1 files: numeric columns were rounded to one decimal and had no stored fill values
        self.numeric_dtypes = state.get("numeric_dtypes") or {c: "float" for c in self.numeric_scalers}
        self.numeric_decimals = state.get("numeric_decimals") or {c: 1 for c in self.numeric_scalers}
        self.fill_values = state.get("fill_values") or {}
        self.dropped_columns = state.get("dropped_columns", [])
        self.warnings = state.get("warnings", [])
        self.n_rows_fit = state.get("n_rows_fit", 0)
        self.fitted_at = state.get("fitted_at")
