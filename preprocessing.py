"""
preprocessing.py
----------------
Handles all data preprocessing tasks:
  - Loading data from uploaded CSV or file path
  - Validating required columns
  - Handling missing values (mean / median / mode)
  - Label-encoding categorical columns
  - Standard-scaling numerical features
  - Preparing feature matrix and target vector for modelling

Pandas inplace operations are intentionally avoided to prevent
FutureWarning deprecation notices.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ── Column definitions ────────────────────────────────────────────────────────
CATEGORICAL_COLS = ["Gender", "Degree", "Branch"]
NUMERICAL_COLS = [
    "Age", "CGPA", "Internships", "Projects",
    "Coding_Skills", "Communication", "Aptitude_Test",
    "Soft_Skills", "Certifications", "Backlogs",
]
TARGET_COL = "Placement_Status"
ALL_FEATURE_COLS = NUMERICAL_COLS + CATEGORICAL_COLS

REQUIRED_FEATURE_COLS = set(ALL_FEATURE_COLS)
REQUIRED_TRAINING_COLS = REQUIRED_FEATURE_COLS | {TARGET_COL}


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_data(source) -> pd.DataFrame:
    """
    Load a CSV into a DataFrame.
    `source` can be a Streamlit UploadedFile or a file-system path string.
    """
    df = pd.read_csv(source)
    return df


def generate_sample_data(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic student placement dataset with realistic distributions.
    Uses a weighted score to derive Placement_Status so feature-target
    correlations are meaningful for ML models.
    """
    rng = np.random.default_rng(seed)

    degrees = ["B.Tech", "BCA", "MCA", "MBA", "B.Sc"]
    branches = ["CS", "IT", "ECE", "Mechanical", "Civil"]

    df = pd.DataFrame({
        "Age":           rng.integers(19, 27, n),
        "Gender":        rng.choice(["Male", "Female"], n),
        "Degree":        rng.choice(degrees, n),
        "Branch":        rng.choice(branches, n),
        "CGPA":          np.round(rng.uniform(5.0, 10.0, n), 2),
        "Internships":   rng.integers(0, 4, n),
        "Projects":      rng.integers(0, 6, n),
        "Coding_Skills": rng.integers(1, 11, n),
        "Communication": rng.integers(1, 11, n),
        "Aptitude_Test": rng.integers(30, 101, n),
        "Soft_Skills":   rng.integers(1, 11, n),
        "Certifications":rng.integers(0, 6, n),
        "Backlogs":      rng.integers(0, 4, n),
    })

    # Weighted placement score — mirrors realistic industry expectations
    score = (
        df["CGPA"]           * 0.30
        + df["Coding_Skills"]  * 0.20
        + df["Communication"]  * 0.15
        + df["Aptitude_Test"]  / 100 * 10 * 0.15
        + df["Internships"]    * 0.50
        + df["Projects"]       * 0.30
        - df["Backlogs"]       * 0.80
        + df["Certifications"] * 0.20
    )
    df["Placement_Status"] = np.where(
        score >= score.quantile(0.45), "Placed", "Not Placed"
    )
    return df


# ── Validation ────────────────────────────────────────────────────────────────

def validate_columns(df: pd.DataFrame, require_target: bool = True) -> list[str]:
    """
    Return a list of missing column names.
    If require_target is True, TARGET_COL is also checked.
    """
    needed = REQUIRED_TRAINING_COLS if require_target else REQUIRED_FEATURE_COLS
    return [c for c in needed if c not in df.columns]


# ── Missing-value handling ────────────────────────────────────────────────────

def get_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame summarising columns that have missing values."""
    missing = df.isnull().sum()
    pct = (missing / len(df) * 100).round(2)
    summary = pd.DataFrame({"Missing Count": missing, "Missing %": pct})
    return summary[summary["Missing Count"] > 0].copy()


def handle_missing_values(df: pd.DataFrame, strategy: str = "median") -> pd.DataFrame:
    """
    Impute missing values without using deprecated inplace operations.

    Parameters
    ----------
    df       : Input DataFrame
    strategy : "median" (default) or "mean" for numerical columns;
               categorical columns always use mode imputation.

    Returns
    -------
    A new DataFrame with no missing values in feature / target columns.
    """
    df = df.copy()

    for col in NUMERICAL_COLS:
        if col in df.columns and df[col].isnull().any():
            fill_val = df[col].mean() if strategy == "mean" else df[col].median()
            df[col] = df[col].fillna(fill_val)

    for col in CATEGORICAL_COLS:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode().iloc[0])

    if TARGET_COL in df.columns:
        df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    return df


# ── Encoding & scaling ────────────────────────────────────────────────────────

def encode_and_scale(
    df: pd.DataFrame,
    fit: bool = True,
    encoders: dict | None = None,
    scaler: StandardScaler | None = None,
):
    """
    Encode categorical columns and scale numerical columns.

    Parameters
    ----------
    df       : DataFrame containing feature columns
    fit      : If True, fit new encoders/scaler on df.
               If False, transform using the provided encoders/scaler.
    encoders : {col_name: LabelEncoder} — required when fit=False
    scaler   : Fitted StandardScaler — required when fit=False

    Returns
    -------
    X        : float64 feature array, shape (n_samples, n_features)
    encoders : Updated encoder dict
    scaler   : (Fitted) StandardScaler
    """
    df = df.copy()
    encoders = encoders or {}

    # ── Encode categoricals ───────────────────────────────────────────────────
    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            continue
        df[col] = df[col].astype(str).str.strip()

        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
        else:
            le = encoders[col]
            known = set(le.classes_)
            # Map unseen labels to the most frequent known class (index 0)
            df[col] = df[col].apply(lambda x: x if x in known else le.classes_[0])
            df[col] = le.transform(df[col])

    # ── Build feature array ───────────────────────────────────────────────────
    feature_cols = [c for c in ALL_FEATURE_COLS if c in df.columns]
    X = df[feature_cols].values.astype(np.float64)

    num_indices = [feature_cols.index(c) for c in NUMERICAL_COLS if c in feature_cols]

    if fit:
        scaler = StandardScaler()
        X[:, num_indices] = scaler.fit_transform(X[:, num_indices])
    else:
        X[:, num_indices] = scaler.transform(X[:, num_indices])

    return X, encoders, scaler


def prepare_features_target(df: pd.DataFrame):
    """
    Full preprocessing pipeline: impute → encode → scale → split X / y.

    Returns
    -------
    X             : float64 feature array
    y             : int label array  (0 = Not Placed, 1 = Placed)
    encoders      : Fitted LabelEncoder dict
    scaler        : Fitted StandardScaler
    feature_names : Ordered list of feature column names
    target_le     : Fitted LabelEncoder for the target column
    """
    df = handle_missing_values(df)

    target_le = LabelEncoder()
    y = target_le.fit_transform(df[TARGET_COL].astype(str))

    X, encoders, scaler = encode_and_scale(df, fit=True)
    feature_names = [c for c in ALL_FEATURE_COLS if c in df.columns]

    return X, y, encoders, scaler, feature_names, target_le
