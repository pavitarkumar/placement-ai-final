import hashlib
from firebase.firebase_config import auth


# =========================================================
# HASH PASSWORD
# =========================================================

def hash_password(password: str) -> str:

    return hashlib.sha256(
        password.encode("utf-8")
    ).hexdigest()


# =========================================================
# CREATE USER
# =========================================================

def create_user(
    full_name: str,
    email: str,
    username: str,
    password: str,
):

    try:

        user = auth.create_user_with_email_and_password(
            email,
            password
        )

        return True, "✅ Account created successfully!"

    except Exception as e:

        return False, "❌ Email already exists or invalid."


# =========================================================
# LOGIN USER
# =========================================================

def login_user(
    username: str,
    password: str,
):

    try:

        user = auth.sign_in_with_email_and_password(
            username,
            password
        )

        user_data = {
            "name": username,
            "email": username,
        }

        return True, user_data

    except Exception:

        return False, None