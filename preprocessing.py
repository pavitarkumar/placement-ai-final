"""
preprocessing.py
"""

import pandas as pd
import numpy as np

from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler,
)

# ─────────────────────────────────────────────────────
# Columns
# ─────────────────────────────────────────────────────

CATEGORICAL_COLS = [
    "Gender",
    "Degree",
    "Branch",
]

NUMERICAL_COLS = [
    "Age",
    "CGPA",
    "Internships",
    "Projects",
    "Coding_Skills",
    "Communication",
    "Aptitude_Test",
    "Soft_Skills",
    "Certifications",
    "Backlogs",
]

TARGET_COL = "Placement_Status"

ALL_FEATURE_COLS = (
    NUMERICAL_COLS + CATEGORICAL_COLS
)

# ─────────────────────────────────────────────────────
# Load CSV
# ─────────────────────────────────────────────────────

def load_data(source):

    return pd.read_csv(source)


# ─────────────────────────────────────────────────────
# Missing Values Summary
# ─────────────────────────────────────────────────────

def get_missing_summary(df):

    missing_count = df.isnull().sum()

    missing_percent = (
        missing_count / len(df) * 100
    ).round(2)

    summary = pd.DataFrame({

        "Missing Count": missing_count,

        "Missing %": missing_percent,
    })

    return summary[
        summary["Missing Count"] > 0
    ]


# ─────────────────────────────────────────────────────
# Validate Columns
# ─────────────────────────────────────────────────────

def validate_columns(
    df,
    require_target=True,
):

    required = ALL_FEATURE_COLS.copy()

    if require_target:
        required.append(TARGET_COL)

    missing = [

        col for col in required

        if col not in df.columns
    ]

    return missing


# ─────────────────────────────────────────────────────
# Handle Missing Values
# ─────────────────────────────────────────────────────

def handle_missing_values(
    df,
    strategy="median",
):

    df = df.copy()

    for col in NUMERICAL_COLS:

        if (
            col in df.columns
            and df[col].isnull().any()
        ):

            if strategy == "mean":

                fill_value = df[col].mean()

            else:

                fill_value = df[col].median()

            df[col] = df[col].fillna(
                fill_value
            )

    for col in CATEGORICAL_COLS:

        if (
            col in df.columns
            and df[col].isnull().any()
        ):

            df[col] = df[col].fillna(
                df[col].mode()[0]
            )

    return df


# ─────────────────────────────────────────────────────
# Generate Dataset
# ─────────────────────────────────────────────────────

def generate_sample_data(
    n=3000,
    seed=42,
):

    rng = np.random.default_rng(seed)

    degrees = [
        "B.Tech",
        "BCA",
        "MCA",
        "MBA",
        "B.Sc",
    ]

    branches = [
        "CS",
        "IT",
        "ECE",
        "Mechanical",
        "Civil",
    ]

    df = pd.DataFrame({

        "Age": rng.integers(
            19,
            27,
            n
        ),

        "Gender": rng.choice(
            ["Male", "Female"],
            n
        ),

        "Degree": rng.choice(
            degrees,
            n
        ),

        "Branch": rng.choice(
            branches,
            n
        ),

        "CGPA": np.round(
            np.clip(
                rng.normal(
                    7.5,
                    0.6,
                    n
                ),
                5,
                10
            ),
            2
        ),

        "Internships": rng.integers(
            0,
            5,
            n
        ),

        "Projects": rng.integers(
            0,
            6,
            n
        ),

        "Coding_Skills": rng.integers(
            2,
            11,
            n
        ),

        "Communication": rng.integers(
            2,
            11,
            n
        ),

        "Aptitude_Test": rng.integers(
            45,
            100,
            n
        ),

        "Soft_Skills": rng.integers(
            2,
            11,
            n
        ),

        "Certifications": rng.integers(
            0,
            5,
            n
        ),

        "Backlogs": rng.integers(
            0,
            3,
            n
        ),
    })

    # ─────────────────────────────────────────────────
    # Strong Placement Logic
    # ─────────────────────────────────────────────────

    score = (

        df["CGPA"] * 0.55

        + df["Coding_Skills"] * 0.42

        + df["Communication"] * 0.30

        + df["Internships"] * 0.28

        + df["Projects"] * 0.22

        + df["Soft_Skills"] * 0.18

        + df["Certifications"] * 0.14

        + df["Aptitude_Test"] * 0.06

        - df["Backlogs"] * 1.10
    )

    # Small randomness
    score += rng.normal(
        0,
        0.35,
        n
    )

    threshold = np.percentile(
        score,
        50
    )

    df[TARGET_COL] = np.where(
        score >= threshold,
        "Placed",
        "Not Placed"
    )

    # Very small label noise
    flip_count = int(
        n * 0.005
    )

    flip_idx = rng.choice(
        n,
        flip_count,
        replace=False
    )

    df.loc[
        flip_idx,
        TARGET_COL
    ] = np.where(
        df.loc[
            flip_idx,
            TARGET_COL
        ] == "Placed",
        "Not Placed",
        "Placed"
    )

    return df


# ─────────────────────────────────────────────────────
# Encode & Scale
# ─────────────────────────────────────────────────────

def encode_and_scale(
    df,
    fit=True,
    encoders=None,
    scaler=None,
):

    df = df.copy()

    encoders = encoders or {}

    if "Student_ID" in df.columns:

        df = df.drop(
            columns=["Student_ID"]
        )

    # Encode categorical
    for col in CATEGORICAL_COLS:

        if col not in df.columns:
            continue

        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
        )

        if fit:

            le = LabelEncoder()

            df[col] = le.fit_transform(
                df[col]
            )

            encoders[col] = le

        else:

            le = encoders[col]

            known = set(
                le.classes_
            )

            df[col] = df[col].apply(
                lambda x:
                x if x in known
                else le.classes_[0]
            )

            df[col] = le.transform(
                df[col]
            )

    X = df[
        ALL_FEATURE_COLS
    ].values.astype(np.float64)

    num_indices = [

        ALL_FEATURE_COLS.index(c)

        for c in NUMERICAL_COLS
    ]

    if fit:

        scaler = StandardScaler()

        X[:, num_indices] = (
            scaler.fit_transform(
                X[:, num_indices]
            )
        )

    else:

        X[:, num_indices] = (
            scaler.transform(
                X[:, num_indices]
            )
        )

    return X, encoders, scaler


# ─────────────────────────────────────────────────────
# Prepare Features & Target
# ─────────────────────────────────────────────────────

def prepare_features_target(df):

    df = handle_missing_values(df)

    target_le = LabelEncoder()

    y = target_le.fit_transform(
        df[TARGET_COL]
    )

    X, encoders, scaler = (
        encode_and_scale(
            df,
            fit=True
        )
    )

    feature_names = (
        ALL_FEATURE_COLS.copy()
    )

    return (

        X,

        y,

        encoders,

        scaler,

        feature_names,

        target_le,
    )