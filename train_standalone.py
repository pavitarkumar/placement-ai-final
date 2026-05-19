"""
train_standalone.py
"""

import os
import sys
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler,
)

# ── Path Setup ──────────────────────────────────────────────────────────────

_HERE = os.path.dirname(
    os.path.abspath(__file__)
)

if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from preprocessing import (
    generate_sample_data,
    handle_missing_values,
)

from train_model import (
    train_with_gridsearch,
    evaluate_model,
    MODELS_DIR,
)

SAVED_PATH = os.path.join(
    MODELS_DIR,
    "session_artifacts.joblib"
)


def main():

    # ── 1. Generate Synthetic Dataset ─────────────────────────────────────

    print("Generating synthetic dataset...")

    df = generate_sample_data(3000)

    print(f"Shape: {df.shape}")

    print(
        f"\nPlacement distribution:\n"
        f"{df['Placement_Status'].value_counts()}\n"
    )

    # ── 2. Preprocessing ──────────────────────────────────────────────────

    print("Preprocessing...")

    df = handle_missing_values(df)

    df = df.drop_duplicates()

    TARGET_COL = "Placement_Status"

    # Remove unwanted columns
    if "Student_ID" in df.columns:

        df = df.drop(
            columns=["Student_ID"]
        )

    X = df.drop(
        columns=[TARGET_COL]
    )

    y = df[TARGET_COL]

    # Encode target
    target_le = LabelEncoder()

    y = target_le.fit_transform(y)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # Detect columns
    cat_cols = X_train.select_dtypes(
        include=["object"]
    ).columns

    num_cols = X_train.select_dtypes(
        exclude=["object"]
    ).columns

    encoders = {}

    # Encode categorical columns
    for col in cat_cols:

        le = LabelEncoder()

        X_train[col] = le.fit_transform(
            X_train[col].astype(str)
        )

        X_test[col] = X_test[col].astype(str)

        unseen = (
            set(X_test[col])
            - set(le.classes_)
        )

        if unseen:

            le.classes_ = (
                list(le.classes_)
                + list(unseen)
            )

        X_test[col] = le.transform(
            X_test[col]
        )

        encoders[col] = le

    # Scale numerical columns
    scaler = StandardScaler()

    X_train[num_cols] = scaler.fit_transform(
        X_train[num_cols]
    )

    X_test[num_cols] = scaler.transform(
        X_test[num_cols]
    )

    feature_names = (
        X_train.columns.tolist()
    )

    # Convert to numpy arrays
    X_train = X_train.values

    X_test = X_test.values

    print(
        f"\nTrain: {X_train.shape[0]}"
        f" | Test: {X_test.shape[0]}"
    )

    # ── 3. Train Models ───────────────────────────────────────────────────

    print("\nTraining models...\n")

    def progress(name, step, total):

        print(
            f"[{step}/{total}] Training {name}"
        )

    results = train_with_gridsearch(
        X_train,
        y_train,
        cv_folds=2,
        progress_callback=progress,
    )

    # ── 4. Evaluate Models ────────────────────────────────────────────────

    print()

    print(
        f"{'Model':<24}"
        f"{'Accuracy':>10}"
        f"{'Precision':>12}"
        f"{'Recall':>10}"
        f"{'F1':>10}"
    )

    print("-" * 70)

    eval_results = {}

    for name, res in results.items():

        ev = evaluate_model(
            res["model"],
            X_test,
            y_test,
        )

        eval_results[name] = ev

        print(
            f"{name:<24}"
            f"{ev['accuracy']:>10.4f}"
            f"{ev['precision']:>12.4f}"
            f"{ev['recall']:>10.4f}"
            f"{ev['f1']:>10.4f}"
        )

    # Best model
    best = max(
        eval_results,
        key=lambda n:
        eval_results[n]["accuracy"]
    )

    print(
        f"\nBest model: {best}"
        f" ({eval_results[best]['accuracy']*100:.2f}%)"
    )

    # ── 5. Save Everything ────────────────────────────────────────────────

    print("\nSaving models...")

    joblib.dump(
        {
            "df_raw": df,
            "df_clean": df,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "encoders": encoders,
            "scaler": scaler,
            "feature_names": feature_names,
            "target_le": target_le,
            "train_results": results,
            "eval_results": eval_results,
        },
        SAVED_PATH,
    )

    print("\nDone Successfully!")


if __name__ == "__main__":
    main()