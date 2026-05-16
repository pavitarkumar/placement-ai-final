"""
train_standalone.py
-------------------
Pre-trains all four models and saves a complete session bundle to disk so the
Streamlit app loads with results already available on first visit.

Usage
-----
    python train_standalone.py

Output
------
    models/logistic_regression.joblib
    models/knn.joblib
    models/random_forest.joblib
    models/gradient_boosting.joblib
    models/session_artifacts.joblib   ← loaded automatically by app.py
"""

import os
import sys
import joblib
from sklearn.model_selection import train_test_split

# Ensure imports resolve from this file's directory (needed when run as a script)
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from preprocessing import generate_sample_data, handle_missing_values, prepare_features_target
from train_model import train_with_gridsearch, evaluate_model, MODELS_DIR

SAVED_PATH = os.path.join(MODELS_DIR, "session_artifacts.joblib")


def main() -> None:
    # ── 1. Generate dataset ───────────────────────────────────────────────────
    print("Generating 500-row sample dataset…")
    df = generate_sample_data(500)
    print(f"  Shape: {df.shape}")
    print(f"  Placement distribution:\n{df['Placement_Status'].value_counts().to_string()}\n")

    # ── 2. Preprocess ─────────────────────────────────────────────────────────
    print("Preprocessing…")
    df = handle_missing_values(df)
    X, y, encoders, scaler, feature_names, target_le = prepare_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"  Train: {X_train.shape[0]} rows  |  Test: {X_test.shape[0]} rows\n")

    # ── 3. Train with GridSearchCV ────────────────────────────────────────────
    print("Training 4 models with GridSearchCV + StratifiedKFold (this takes ~60 s)…")

    def _progress(name: str, step: int, total: int) -> None:
        print(f"  [{step}/{total}] Training {name}…")

    results = train_with_gridsearch(
        X_train, y_train, cv_folds=5, progress_callback=_progress
    )

    # ── 4. Evaluate ───────────────────────────────────────────────────────────
    print()
    print(f"{'Model':<24} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'ROC-AUC':>9} {'CV Mean':>9}")
    print("-" * 83)

    eval_results: dict = {}
    for name, res in results.items():
        ev = evaluate_model(res["model"], X_test, y_test)
        eval_results[name] = ev
        roc = f"{ev['roc_auc']:.4f}" if ev.get("roc_auc") is not None else "   —  "
        print(
            f"  {name:<22} {ev['accuracy']:>9.4f} {ev['precision']:>10.4f}"
            f" {ev['recall']:>8.4f} {ev['f1']:>8.4f} {roc:>9} {res['cv_mean']:>9.4f}"
        )

    best = max(eval_results, key=lambda n: eval_results[n]["accuracy"])
    print(f"\n  🏆 Best model: {best}  ({eval_results[best]['accuracy']*100:.2f}% accuracy)")

    # ── 5. Save session bundle ────────────────────────────────────────────────
    print(f"\nSaving session bundle to {SAVED_PATH} …")
    joblib.dump(
        {
            "df_raw":        df,
            "df_clean":      df,
            "X_train":       X_train,
            "X_test":        X_test,
            "y_train":       y_train,
            "y_test":        y_test,
            "encoders":      encoders,
            "scaler":        scaler,
            "feature_names": feature_names,
            "target_le":     target_le,
            "train_results": results,
            "eval_results":  eval_results,
        },
        SAVED_PATH,
    )
    print("Done — all models trained and saved successfully!")


if __name__ == "__main__":
    main()
