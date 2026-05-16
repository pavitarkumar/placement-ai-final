import os
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODELS_DIR, exist_ok=True)

PARAM_GRIDS = {

    "Logistic Regression": {
        "C": [0.1, 1, 10],
        "solver": ["liblinear"],
        "max_iter": [1000],
    },

    "KNN": {
        "n_neighbors": [3, 5, 7],
        "weights": ["uniform", "distance"],
    },

    "Random Forest": {
        "n_estimators": [100, 200],
        "max_depth": [None, 10],
        "min_samples_split": [2, 5],
    },

    "Gradient Boosting": {
        "n_estimators": [100],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3],
    },
}


def get_base_models():

    return {

        "Logistic Regression": LogisticRegression(
            random_state=42,
            solver="liblinear",
            max_iter=1000,
        ),

        "KNN": KNeighborsClassifier(),

        "Random Forest": RandomForestClassifier(
            random_state=42,
            n_jobs=-1,
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            random_state=42,
        ),
    }


def train_with_gridsearch(
    X_train,
    y_train,
    cv_folds=5,
    progress_callback=None,
):

    cv = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=42,
    )

    models = get_base_models()

    results = {}

    total_models = len(models)

    for idx, (name, model) in enumerate(models.items(), start=1):

        if progress_callback:
            try:
                progress_callback(name, idx, total_models)
            except Exception:
                pass

        print(f"Training: {name}")

        grid = GridSearchCV(
            estimator=model,
            param_grid=PARAM_GRIDS[name],
            scoring="accuracy",
            cv=cv,
            n_jobs=-1,
            refit=True,
            error_score="raise",
        )

        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_

        cv_scores = cross_val_score(
            best_model,
            X_train,
            y_train,
            cv=cv,
            scoring="accuracy",
        )

        safe_name = name.lower().replace(" ", "_")

        model_path = os.path.join(
            MODELS_DIR,
            f"{safe_name}.joblib"
        )

        joblib.dump(best_model, model_path)

        results[name] = {
            "model": best_model,
            "best_params": grid.best_params_,
            "cv_scores": cv_scores,
            "cv_mean": float(cv_scores.mean()),
            "cv_std": float(cv_scores.std()),
            "model_path": model_path,
        }

        print(f"{name} saved successfully")

    return results


def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    try:
        y_proba = model.predict_proba(X_test)[:, 1]
    except Exception:
        y_proba = None

    try:
        roc_auc = roc_auc_score(y_test, y_proba)
    except Exception:
        roc_auc = None

    results = {

        "accuracy": float(accuracy_score(y_test, y_pred)),

        "precision": float(
            precision_score(
                y_test,
                y_pred,
                zero_division=0,
            )
        ),

        "recall": float(
            recall_score(
                y_test,
                y_pred,
                zero_division=0,
            )
        ),

        "f1": float(
            f1_score(
                y_test,
                y_pred,
                zero_division=0,
            )
        ),

        "roc_auc": roc_auc,

        "confusion_matrix": confusion_matrix(
            y_test,
            y_pred,
        ),

        "report": classification_report(
            y_test,
            y_pred,
            target_names=[
                "Not Placed",
                "Placed",
            ],
            zero_division=0,
        ),

        "y_pred": y_pred,

        "y_proba": y_proba,
    }

    return results


def load_model(model_name):

    safe_name = model_name.lower().replace(" ", "_")

    model_path = os.path.join(
        MODELS_DIR,
        f"{safe_name}.joblib"
    )

    if not os.path.exists(model_path):

        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    model = joblib.load(model_path)

    return model


def get_feature_importances(model, feature_names):

    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame()

    importance_df = pd.DataFrame({

        "Feature": feature_names,

        "Importance": model.feature_importances_,
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False,
    )

    importance_df.reset_index(
        drop=True,
        inplace=True,
    )

    return importance_df