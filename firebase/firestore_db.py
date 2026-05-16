from firebase.firebase_config import db
from datetime import datetime


# =========================================================
# SAVE PREDICTION
# =========================================================

def save_prediction(
    username: str,
    prediction_data: dict,
):

    prediction_data["username"] = username

    prediction_data["timestamp"] = str(
        datetime.now()
    )

    db.collection("predictions").add(
        prediction_data
    )


# =========================================================
# GET USER PREDICTIONS
# =========================================================

def get_user_predictions(username: str):

    docs = (
        db.collection("predictions")
        .where("username", "==", username)
        .stream()
    )

    results = []

    for doc in docs:
        results.append(doc.to_dict())

    return results