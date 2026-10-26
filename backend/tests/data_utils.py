"""Small deterministic datasets for the tests."""
import io

import numpy as np
import pandas as pd


def make_df(n: int = 300, seed: int = 0, missing: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    hours = rng.normal(15, 5, n).clip(1, 30).round(1)
    df = pd.DataFrame({
        "student_id": [f"S{i:04d}" for i in range(n)],
        "gender": rng.choice(["Female", "Male"], n),
        "grade": rng.choice(["Freshman", "Junior", "Senior", "Sophomore"], n),
        "age": rng.integers(18, 30, n),
        "hours": hours,
        "score": (45 + 2.0 * hours + rng.normal(0, 4, n)).clip(0, 100).round(1),
    })
    if missing:
        df.loc[df.index[:6], "hours"] = np.nan
        df.loc[df.index[10:16], "grade"] = None
    return df


def csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()
