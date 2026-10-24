import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

class DataPreprocessor:
    def __init__(self):
        self.numeric_scalers: Dict[str, Dict[str, float]] = {} # col -> {min, max}
        self.categorical_encoders: Dict[str, List[str]] = {}   # col -> list of categories
        self.column_order: List[str] = []
        self.processed_feature_names: List[str] = []
        self.feature_slices: Dict[str, Any] = {} # col -> (start_idx, end_idx, type)

    @staticmethod
    def inspect_csv(file_path: str) -> Tuple[int, int, List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Reads CSV and returns row count, col count, schema metadata, and preview rows.
        """
        df = pd.read_csv(file_path)
        row_count = len(df)
        col_count = len(df.columns)
        
        schema = []
        for col in df.columns:
            null_count = int(df[col].isnull().sum())
            null_pct = round((null_count / max(row_count, 1)) * 100, 1)
            is_id = 'id' in col.lower() or df[col].nunique() == row_count
            
            is_numeric = pd.api.types.is_numeric_dtype(df[col]) and not is_id
            
            meta: Dict[str, Any] = {
                "name": col,
                "inferredType": "id" if is_id else "numeric" if is_numeric else "categorical",
                "selectedType": "drop" if is_id else "numeric" if is_numeric else "categorical",
                "nullCount": null_count,
                "nullPercentage": null_pct,
                "imputationStrategy": "drop" if is_id else "median" if is_numeric else "mode",
                "included": not is_id
            }
            
            if is_numeric:
                clean_series = df[col].dropna()
                if len(clean_series) > 0:
                    meta["min"] = float(round(clean_series.min(), 2))
                    meta["max"] = float(round(clean_series.max(), 2))
                    meta["mean"] = float(round(clean_series.mean(), 2))
                    meta["std"] = float(round(clean_series.std(), 2))
            else:
                cats = df[col].dropna().unique().tolist()
                meta["categories"] = [str(c) for c in cats[:20]]
                
            schema.append(meta)
            
        preview = df.head(50).fillna("").to_dict(orient="records")
        return row_count, col_count, schema, preview

    def fit_transform(self, df: pd.DataFrame, column_configs: List[Dict[str, Any]]) -> np.ndarray:
        """
        Fits scalers and encoders on selected columns and converts to normalized float32 array.
        """
        self.numeric_scalers = {}
        self.categorical_encoders = {}
        self.column_order = []
        self.processed_feature_names = []
        self.feature_slices = {}

        processed_parts = []
        current_idx = 0

        # Filter included columns
        config_map = {c['name']: c for c in column_configs if c.get('included', True)}

        for col, cfg in config_map.items():
            if col not in df.columns or cfg.get('selectedType') == 'drop':
                continue

            self.column_order.append(col)
            series = df[col].copy()
            sel_type = cfg.get('selectedType', cfg.get('inferredType', 'numeric'))
            impute_strat = cfg.get('imputationStrategy', 'median')

            if sel_type == 'numeric':
                # Impute missing
                if impute_strat == 'mean':
                    fill_val = series.mean()
                else: # median or default
                    fill_val = series.median()
                series = series.fillna(fill_val)

                # Min-Max Scaling to [0.0, 1.0]
                col_min = float(series.min())
                col_max = float(series.max())
                denom = col_max - col_min if col_max > col_min else 1.0
                normalized = ((series - col_min) / denom).to_numpy(dtype=np.float32).reshape(-1, 1)

                self.numeric_scalers[col] = {"min": col_min, "max": col_max}
                self.feature_slices[col] = (current_idx, current_idx + 1, "numeric")
                self.processed_feature_names.append(col)
                current_idx += 1
                processed_parts.append(normalized)

            elif sel_type == 'categorical':
                # Impute mode
                mode_val = series.mode()[0] if not series.mode().empty else "Unknown"
                series = series.fillna(mode_val).astype(str)

                categories = sorted(series.unique().tolist())
                self.categorical_encoders[col] = categories

                # One-hot encoding
                cat_to_idx = {cat: i for i, cat in enumerate(categories)}
                num_cats = len(categories)
                one_hot = np.zeros((len(series), num_cats), dtype=np.float32)

                for row_i, val in enumerate(series):
                    idx = cat_to_idx.get(val, 0)
                    one_hot[row_i, idx] = 1.0

                self.feature_slices[col] = (current_idx, current_idx + num_cats, "categorical")
                for cat in categories:
                    self.processed_feature_names.append(f"{col}_{cat}")
                current_idx += num_cats
                processed_parts.append(one_hot)

        if not processed_parts:
            raise ValueError("No valid features selected for preprocessing.")

        return np.hstack(processed_parts)

    def inverse_transform(self, normalized_data: np.ndarray) -> pd.DataFrame:
        """
        Converts generated [0, 1] array back to original feature distributions and categorical names.
        """
        output_dict: Dict[str, Any] = {}
        n_samples = normalized_data.shape[0]

        for col, (start, end, ftype) in self.feature_slices.items():
            col_slice = normalized_data[:, start:end]

            if ftype == "numeric":
                scaler = self.numeric_scalers[col]
                c_min = scaler["min"]
                c_max = scaler["max"]
                # Denormalize: val * (max - min) + min
                denorm = col_slice.flatten() * (c_max - c_min) + c_min
                # Domain clipping
                clipped = np.clip(denorm, c_min, c_max)
                output_dict[col] = np.round(clipped, 1)

            elif ftype == "categorical":
                categories = self.categorical_encoders[col]
                # Argmax over one-hot logits/probabilities
                best_indices = np.argmax(col_slice, axis=1)
                decoded_categories = [categories[idx] if idx < len(categories) else categories[0] for idx in best_indices]
                output_dict[col] = decoded_categories

        return pd.DataFrame(output_dict)

    def save(self, file_path: str):
        state = {
            "numeric_scalers": self.numeric_scalers,
            "categorical_encoders": self.categorical_encoders,
            "column_order": self.column_order,
            "processed_feature_names": self.processed_feature_names,
            "feature_slices": {k: list(v) for k, v in self.feature_slices.items()}
        }
        with open(file_path, "w") as f:
            json.dump(state, f, indent=2)

    def load(self, file_path: str):
        with open(file_path, "r") as f:
            state = json.load(f)
        self.numeric_scalers = state["numeric_scalers"]
        self.categorical_encoders = state["categorical_encoders"]
        self.column_order = state["column_order"]
        self.processed_feature_names = state["processed_feature_names"]
        self.feature_slices = {k: tuple(v) for k, v in state["feature_slices"].items()}
