from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
import os
import joblib
import pandas as pd
import shap
import sqlite3
import sys

# Make the model folder importable.
MODEL_DIR = os.path.join(
    os.path.dirname(__file__),
    "model"
)

if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

from hierarchical_cascade import (
    HierarchicalThyroidCascade,
    stage1_model,
    stage2_model
)

from ann_feature_conversion import (
    convert_to_ann_features
)


# ============================================================
# Flask application
# ============================================================
load_dotenv()
app = Flask(__name__)

app.secret_key = os.getenv("FLASK_SECRET_KEY")


# ============================================================
# Load final XGBoost model
# ============================================================

model = joblib.load(
    r".\model\final_xgboost_model.pkl"
)

explainer = shap.TreeExplainer(model)
# ============================================================
# New hierarchical cascade
# ============================================================

hierarchical_cascade = HierarchicalThyroidCascade(
    stage1_model,
    stage2_model
)

# ============================================================
# SQLite database
# ============================================================

DATABASE = "thyroid_predictions.db"


# ============================================================
# Save prediction
# ============================================================

def save_prediction(
    user_id,
    predicted_class,
    negative_probability,
    hypothyroid_probability,
    shap_summary
):
    connection = sqlite3.connect(DATABASE)

    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO predictions (
            user_id,
            predicted_class,
            negative_probability,
            hypothyroid_probability,
            shap_summary
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            user_id,
            predicted_class,
            negative_probability,
            hypothyroid_probability,
            shap_summary
        )
    )

    connection.commit()
    connection.close()


# ============================================================
# Home page
# ============================================================

@app.route("/")
def home():

    if "user_id" not in session:
        return render_template("welcome.html")

    return render_template(
        "index.html",
        session_username=session["username"]
    )


# ============================================================
# Welcome page
# ============================================================

@app.route("/welcome")
def welcome():

    return render_template("welcome.html")


# ============================================================
# Registration
# ============================================================

@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        username = request.form["username"].strip()
        email = request.form["email"].strip()
        password = request.form["password"]

        password_hash = generate_password_hash(password)

        connection = sqlite3.connect(DATABASE)
        cursor = connection.cursor()

        try:

            cursor.execute(
                """
                INSERT INTO users (
                    username,
                    email,
                    password_hash
                )
                VALUES (?, ?, ?)
                """,
                (
                    username,
                    email,
                    password_hash
                )
            )

            connection.commit()

        except sqlite3.IntegrityError:

            connection.close()

            return "Username or email already exists."

        connection.close()

        return redirect(url_for("home"))

    return render_template("register.html")


# ============================================================
# Login
# ============================================================

@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        username = request.form["username"].strip()
        password = request.form["password"]

        connection = sqlite3.connect(DATABASE)
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                id,
                username,
                password_hash
            FROM users
            WHERE username = ?
            """,
            (username,)
        )

        user = cursor.fetchone()

        connection.close()

        if user and check_password_hash(
            user[2],
            password
        ):

            session["user_id"] = user[0]
            session["username"] = user[1]

            return redirect(url_for("home"))

        return render_template(
            "login.html",
            error="Invalid username or password."
        )

    return render_template("login.html")


# ============================================================
# Logout
# ============================================================

@app.route("/logout")
def logout():

    session.clear()

    return redirect(url_for("login"))


# ============================================================
# Prediction history
# ============================================================

@app.route("/history")
def history():

    if "user_id" not in session:
        return redirect(url_for("login"))

    connection = sqlite3.connect(DATABASE)
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            id,
            predicted_class,
            negative_probability,
            hypothyroid_probability,
            created_at,
            shap_summary
        FROM predictions
        WHERE user_id = ?
        ORDER BY id DESC
        """,
        (session["user_id"],)
    )

    predictions = cursor.fetchall()

    connection.close()

    return render_template(
        "history.html",
        predictions=predictions,
        session_username=session["username"]
    )


# ============================================================
# Prediction
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    # --------------------------------------------------------
    # Check login
    # --------------------------------------------------------

    if "user_id" not in session:
        return redirect(url_for("login"))

    # --------------------------------------------------------
    # Validate age
    # --------------------------------------------------------

    try:
        age = float(request.form["age"])
    except (ValueError, TypeError):

        return render_template(
            "index.html",
            session_username=session["username"],
            error="Please enter a valid age."
        )

    if age < 1 or age > 120:

        return render_template(
            "index.html",
            session_username=session["username"],
            error="Age must be between 1 and 120."
        )
        # --------------------------------------------------------
    # Validate thyroid measurements
    # --------------------------------------------------------

    try:
        TSH = float(request.form["TSH"])
        T3 = float(request.form["T3"])
        TT4 = float(request.form["TT4"])
        T4U = float(request.form["T4U"])
        FTI = float(request.form["FTI"])

    except (ValueError, TypeError):

        return render_template(
            "index.html",
            session_username=session["username"],
            error="Please enter valid thyroid measurement values."
        )

    if TSH < 0 or T3 < 0 or TT4 < 0 or T4U < 0 or FTI < 0:

        return render_template(
            "index.html",
            session_username=session["username"],
            error="Thyroid measurement values cannot be negative."
        )

    # --------------------------------------------------------
    # Collect form values
    # --------------------------------------------------------

    data = {

        "age": age,

        "sex": int(
            request.form["sex"]
        ),

        "on_thyroxine": int(
            request.form["on_thyroxine"]
        ),

        "query_on_thyroxine": int(
            request.form["query_on_thyroxine"]
        ),

        "on_antithyroid_medication": int(
            request.form["on_antithyroid_medication"]
        ),

        "thyroid_surgery": int(
            request.form["thyroid_surgery"]
        ),
        "I131_treatment": int(
    request.form["I131_treatment"]
),

"hypopituitary": int(
    request.form["hypopituitary"]
),

"psych": int(
    request.form["psych"]
),

        "query_hypothyroid": int(
            request.form["query_hypothyroid"]
        ),

        "query_hyperthyroid": int(
            request.form["query_hyperthyroid"]
        ),

        "pregnant": int(
            request.form["pregnant"]
        ),

        "sick": int(
            request.form["sick"]
        ),

        "tumor": int(
            request.form["tumor"]
        ),

        "lithium": int(
            request.form["lithium"]
        ),

        "goitre": int(
            request.form["goitre"]
        ),

        "TSH_measured": int(
            request.form["TSH_measured"]
        ),

        "TSH": TSH,

        "T3_measured": int(
            request.form["T3_measured"]
        ),

        "T3": T3,

        "TT4_measured": int(
            request.form["TT4_measured"]
        ),

        "TT4": TT4,

        "T4U_measured": int(
            request.form["T4U_measured"]
        ),

        "T4U": T4U,

        "FTI_measured": int(
            request.form["FTI_measured"]
        ),

       "FTI": FTI
    }
        # ============================================================
    # Convert raw form values to the 21 ANN features
    # ============================================================

    ann_sample = convert_to_ann_features(
        data
    )

    # Use the converted 21-feature sample
    sample = ann_sample
    

    # --------------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------------

        # Use the new 21-feature ANN sample
    sample = ann_sample

       # ============================================================
    # Hierarchical prediction
    # ============================================================

    cascade_result = hierarchical_cascade.predict(
        sample
    )

    stage1_class = cascade_result[
        "stage1_class"
    ]

    stage2_class = cascade_result[
        "stage2_class"
    ]

    final_class = cascade_result[
        "final_class"
    ]
    

    stage1_probabilities = (
        cascade_result[
            "stage1_probabilities"
        ]
    )

    stage2_probabilities = (
        cascade_result[
            "stage2_probabilities"
        ]
    )


    # ============================================================
    # Stage 1 probability percentages
    # ============================================================

    stage1_normal_probability = round(
        float(stage1_probabilities[0]) * 100,
        2
    )

    stage1_dysfunction_probability = round(
        float(stage1_probabilities[1]) * 100,
        2
    )


    # ============================================================
    # Stage 2 probability percentages
    # ============================================================

    stage2_normal_probability = None
    stage2_hyperfunction_probability = None
    stage2_subnormal_probability = None

    if stage2_probabilities is not None:

        stage2_normal_probability = round(
            float(stage2_probabilities[0]) * 100,
            2
        )

        stage2_hyperfunction_probability = round(
            float(stage2_probabilities[1]) * 100,
            2
        )

        stage2_subnormal_probability = round(
            float(stage2_probabilities[2]) * 100,
            2
        )
        # ============================================================
    # SHAP explanation using hierarchical Stage 1 XGBoost
    # ============================================================

    stage1_explainer = shap.TreeExplainer(
        stage1_model
    )

    shap_values = stage1_explainer.shap_values(
        sample
    )

    # Handle both NumPy-array and SHAP Explanation outputs
    if hasattr(shap_values, "values"):
        shap_array = shap_values.values
    else:
        shap_array = shap_values

    # Binary XGBoost returns one SHAP value per feature
    shap_array = pd.DataFrame(
        shap_array
    ).iloc[0].to_numpy()

    shap_result = pd.DataFrame(
        {
            "Feature": sample.columns,
            "SHAP_Value": shap_array
        }
    )

    shap_result["Absolute_SHAP"] = (
        shap_result["SHAP_Value"].abs()
    )

    shap_result = shap_result.sort_values(
        by="Absolute_SHAP",
        ascending=False
    )

    top_shap_features = (
        shap_result
        .head(5)
        .to_dict("records")
    )

    # Create a short SHAP summary for SQLite
    shap_summary = "; ".join(
        [
            f"{feature['Feature']}: "
            f"{feature['SHAP_Value']:.4f}"
            for feature in top_shap_features
        ]
    )

        # ============================================================
    # Convert hierarchical prediction to display class
    # ============================================================

    if final_class == "normal":

        predicted_class = "Normal"

    elif final_class == "hyperfunction":

        predicted_class = "Hyperfunction"

    elif final_class == "subnormal":

        predicted_class = "Subnormal"

    else:

        predicted_class = "Unknown"


    # ============================================================
    # Compatibility values for the existing result/database code
    #
    # These will be renamed/restructured in a later step.
    # ============================================================

    negative_probability = (
        stage1_normal_probability
    )

    hypothyroid_probability = (
        stage1_dysfunction_probability
    )

    # --------------------------------------------------------
    # Save prediction to SQLite
    # --------------------------------------------------------

    save_prediction(
        session["user_id"],
        predicted_class,
        negative_probability,
        hypothyroid_probability,
        shap_summary
    )

    # --------------------------------------------------------
    # Render result page
    # --------------------------------------------------------
    return render_template(
        "result.html",

        # Final prediction
        predicted_class=predicted_class,
        final_class=final_class,

        # Stage 1
        stage1_class=stage1_class,
        stage1_normal_probability=stage1_normal_probability,
        stage1_dysfunction_probability=stage1_dysfunction_probability,

        # Stage 2
        stage2_class=stage2_class,
        stage2_normal_probability=stage2_normal_probability,
        stage2_hyperfunction_probability=stage2_hyperfunction_probability,
        stage2_subnormal_probability=stage2_subnormal_probability,

        # Session
        session_username=session["username"],

        # Explainability
        top_shap_features=top_shap_features
    )


# ============================================================
# Run Flask
# ============================================================

if __name__ == "__main__":
    app.run(debug=True)