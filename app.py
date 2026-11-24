from flask import Flask, render_template, request, redirect, url_for, session, send_file
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
import os
import joblib
import pandas as pd
import shap
import sqlite3
import sys
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)

MODEL_DIR = os.path.join(
    os.path.dirname(__file__),
    "model"
)

if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)


# ============================================================
# Enhanced thyroid prediction system
# ============================================================

from enhanced_hierarchical_cascade import (
    EnhancedHierarchicalCascade
)


# ============================================================
# Flask application
# ============================================================

load_dotenv()

app = Flask(__name__)

app.secret_key = os.getenv("FLASK_SECRET_KEY")


# ============================================================
# Load fitted enhanced preprocessing pipeline
# ============================================================

enhanced_preprocessor = joblib.load(
    os.path.join(
        MODEL_DIR,
        "enhanced_preprocessor.pkl"
    )
)


# ============================================================
# Load final enhanced hierarchical cascade
# ============================================================

enhanced_cascade = joblib.load(
    os.path.join(
        MODEL_DIR,
        "final_enhanced_hierarchical_cascade.pkl"
    )
)


# ============================================================
# SQLite database
# ============================================================

DATABASE = os.path.join(
    os.path.dirname(__file__),
    "thyroid_predictions.db"
)

# ============================================================
# Initialize SQLite database
# ============================================================

def init_db():

    connection = sqlite3.connect(DATABASE)
    cursor = connection.cursor()

    # Users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL
        )
    """)

    # Predictions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            predicted_class TEXT,
            negative_probability REAL,
            hypothyroid_probability REAL,
            shap_summary TEXT,
            stage1_class TEXT,
            stage1_confidence REAL,
            routing_decision TEXT,
            stage2_used INTEGER,
            stage2_normal_probability REAL,
            stage2_hyperthyroid_probability REAL,
            stage2_hypothyroid_probability REAL,
            stage2_other_probability REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    connection.commit()
    connection.close()


# Initialize database when application starts
init_db()



# ============================================================
# Save prediction
# ============================================================

def save_prediction(
    user_id,
    predicted_class,
    negative_probability,
    hypothyroid_probability,
    shap_summary,
    stage1_class,
    stage1_confidence,
    routing_decision,
    stage2_used,
    stage2_normal_probability,
    stage2_hyperthyroid_probability,
    stage2_hypothyroid_probability,
    stage2_other_probability
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
            shap_summary,
            stage1_class,
            stage1_confidence,
            routing_decision,
            stage2_used,
            stage2_normal_probability,
            stage2_hyperthyroid_probability,
            stage2_hypothyroid_probability,
            stage2_other_probability
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            predicted_class,
            negative_probability,
            hypothyroid_probability,
            shap_summary,
            stage1_class,
            stage1_confidence,
            routing_decision,
            int(stage2_used),
            stage2_normal_probability,
            stage2_hyperthyroid_probability,
            stage2_hypothyroid_probability,
            stage2_other_probability
        )
    )

    connection.commit()
    connection.close()



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

        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        # ------------------------------
        # Username validation
        # ------------------------------
        username_clean = username.replace("_", "")

        if (
            len(username) < 3
            or len(username) > 30
            or not username_clean.isalnum()
        ):
            return render_template(
                "register.html",
                error=(
                    "Username must be 3â€“30 characters and contain "
                    "only letters, numbers, and underscores."
                )
            )

        # ------------------------------
        # Email validation
        # ------------------------------
        if (
            "@" not in email
            or "." not in email.rsplit("@", 1)[-1]
            or email.startswith("@")
            or email.endswith("@")
        ):
            return render_template(
                "register.html",
                error="Please enter a valid email address."
            )

        # ------------------------------
        # Password validation
        # ------------------------------
        if len(password) < 8:
            return render_template(
                "register.html",
                error="Password must contain at least 8 characters."
            )

        if not any(char.isupper() for char in password):
            return render_template(
                "register.html",
                error="Password must contain at least one uppercase letter."
            )

        if not any(char.islower() for char in password):
            return render_template(
                "register.html",
                error="Password must contain at least one lowercase letter."
            )

        if not any(char.isdigit() for char in password):
            return render_template(
                "register.html",
                error="Password must contain at least one number."
            )

        # ------------------------------
        # Confirm password
        # ------------------------------
        if password != confirm_password:
            return render_template(
                "register.html",
                error="Password and confirm password do not match."
            )

        # ------------------------------
        # Check existing account
        # ------------------------------
        connection = sqlite3.connect(DATABASE)
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT username, email
            FROM users
            WHERE LOWER(username) = LOWER(?)
               OR LOWER(email) = LOWER(?)
            """,
            (
                username,
                email
            )
        )

        existing_user = cursor.fetchone()

        if existing_user:
            connection.close()

            if existing_user[0].lower() == username.lower():
                message = "That username is already registered."
            else:
                message = "That email address is already registered."

            return render_template(
                "register.html",
                error=message
            )

        # ------------------------------
        # Create secure password hash
        # ------------------------------
        password_hash = generate_password_hash(password)

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

            connection.rollback()
            connection.close()

            return render_template(
                "register.html",
                error="Unable to create the account because the username or email already exists."
            )

        connection.close()

        return redirect(url_for("login", registered="1"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        login_identifier = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if not login_identifier or not password:
            return render_template(
                "login.html",
                error="Please enter your username/email and password."
            )

        connection = sqlite3.connect(DATABASE)
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                id,
                username,
                password_hash
            FROM users
            WHERE LOWER(username) = LOWER(?)
               OR LOWER(email) = LOWER(?)
            """,
            (
                login_identifier,
                login_identifier
            )
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
            error="Invalid username/email or password."
        )

    return render_template(
        "login.html",
        registered=request.args.get("registered") == "1"
    )


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
            shap_summary,
            stage1_class,
            stage1_confidence,
            routing_decision,
            stage2_used,
            stage2_normal_probability,
            stage2_hyperthyroid_probability,
            stage2_hypothyroid_probability,
            stage2_other_probability
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

        age = float(
            request.form["age"]
        )

    except (ValueError, TypeError, KeyError):

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

        TSH = float(
            request.form["TSH"]
        )

        T3 = float(
            request.form["T3"]
        )

        TT4 = float(
            request.form["TT4"]
        )

        T4U = float(
            request.form["T4U"]
        )

        FTI = float(
            request.form["FTI"]
        )

    except (ValueError, TypeError, KeyError):

        return render_template(
            "index.html",
            session_username=session["username"],
            error="Please enter valid thyroid measurement values."
        )

    if (
        TSH < 0
        or T3 < 0
        or TT4 < 0
        or T4U < 0
        or FTI < 0
    ):

        return render_template(
            "index.html",
            session_username=session["username"],
            error="Thyroid measurement values cannot be negative."
        )


    # --------------------------------------------------------
    # Validate TBG
    # --------------------------------------------------------

    TBG_text = request.form.get(
        "TBG",
        ""
    ).strip()

    if TBG_text:

        try:
            TBG = float(TBG_text)

        except (ValueError, TypeError):

            return render_template(
                "index.html",
                session_username=session["username"],
                error="Please enter a valid TBG value."
            )

        if TBG < 0:

            return render_template(
                "index.html",
                session_username=session["username"],
                error="TBG cannot be negative."
            )

    else:

        TBG = float("nan")


    # --------------------------------------------------------
    # Validate referral source
    # --------------------------------------------------------

    allowed_referral_sources = {
        "other",
        "STMW",
        "SVHC",
        "SVHD",
        "SVI",
        "WEST"
    }

    referral_source = request.form.get(
        "referral_source",
        ""
    ).strip()

    if referral_source not in allowed_referral_sources:

        return render_template(
            "index.html",
            session_username=session["username"],
            error="Please select a valid referral source."
        )


    # --------------------------------------------------------
    # Collect 29 raw enhanced-model fields
    # --------------------------------------------------------

    binary_fields = [
        "on_thyroxine",
        "query_on_thyroxine",
        "on_antithyroid_medication",
        "sick",
        "pregnant",
        "thyroid_surgery",
        "I131_treatment",
        "query_hypothyroid",
        "query_hyperthyroid",
        "lithium",
        "goitre",
        "tumor",
        "hypopituitary",
        "psych",
        "TSH_measured",
        "T3_measured",
        "TT4_measured",
        "T4U_measured",
        "FTI_measured",
        "TBG_measured"
    ]


    # --------------------------------------------------------
    # Convert HTML categorical values to training representation
    # --------------------------------------------------------

    sex_value = request.form.get(
        "sex",
        ""
    )

    if sex_value == "1":

        sex = "M"

    elif sex_value == "0":

        sex = "F"

    else:

        return render_template(
            "index.html",
            session_username=session["username"],
            error="Please select a valid sex."
        )


    # --------------------------------------------------------
    # Convert HTML binary values
    # --------------------------------------------------------

    binary_values = {}

    for field in binary_fields:

        value = request.form.get(
            field,
            ""
        )

        if value == "1":

            binary_values[field] = "t"

        elif value == "0":

            binary_values[field] = "f"

        else:

            return render_template(
                "index.html",
                session_username=session["username"],
                error=f"Invalid value for {field}."
            )


    # --------------------------------------------------------
    # Build the exact 29-column raw model input
    # --------------------------------------------------------

    data = {

        "age": age,

        "sex": sex,

        **binary_values,

        "TSH": TSH,

        "T3": T3,

        "TT4": TT4,

        "T4U": T4U,

        "FTI": FTI,

        "TBG": TBG,

        "referral_source": referral_source
    }


    # --------------------------------------------------------
    # Create raw enhanced DataFrame
    # --------------------------------------------------------

    raw_sample = pd.DataFrame(
        [data]
    )


    # --------------------------------------------------------
    # Apply the already-fitted preprocessing pipeline
    # --------------------------------------------------------

    try:

        sample_array = enhanced_preprocessor.transform(
            raw_sample
        )

    except Exception:

        return render_template(
            "index.html",
            session_username=session["username"],
            error="Unable to preprocess the supplied patient data."
        )


    sample = pd.DataFrame(
        sample_array,
        columns=enhanced_preprocessor.get_feature_names_out()
    )


    # --------------------------------------------------------
    # Enhanced hierarchical prediction
    # --------------------------------------------------------

    try:

        cascade_result = enhanced_cascade.predict_one(
            sample
        )

    except Exception:

        return render_template(
            "index.html",
            session_username=session["username"],
            error="Unable to generate the thyroid prediction."
        )


    # --------------------------------------------------------
    # Stage 1 result
    # --------------------------------------------------------

    stage1_class = (
        cascade_result.stage1_label
    )

    stage1_probability_normal = (
        float(
            cascade_result.stage1_probability_normal
        )
    )

    stage1_probability_non_normal = (
        float(
            cascade_result.stage1_probability_non_normal
        )
    )

    stage1_confidence = (
        float(
            cascade_result.stage1_confidence
        )
    )

    routing_decision = (
        cascade_result.routing_decision
    )


    # --------------------------------------------------------
    # Stage 2 result
    # --------------------------------------------------------

    stage2_probabilities = (
        cascade_result.stage2_probabilities
    )

    stage2_used = (
        bool(
            cascade_result.stage2_used
        )
    )

    if stage2_probabilities is not None:

        stage2_probabilities = [
            float(value)
            for value in stage2_probabilities
        ]

        stage2_class_id = int(
            max(
                range(
                    len(stage2_probabilities)
                ),
                key=stage2_probabilities.__getitem__
            )
        )

    else:

        stage2_class_id = None


    # --------------------------------------------------------
    # Four-class label mapping
    # --------------------------------------------------------

    class_labels = {
        0: "Normal",
        1: "Hyperthyroid",
        2: "Hypothyroid",
        3: "Other / Related Diagnosis"
    }

    stage2_class_label = (
        class_labels.get(
            stage2_class_id
        )
        if stage2_class_id is not None
        else None
    )


    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    final_class_id = int(
        cascade_result.final_class
    )

    final_class_label = (
        cascade_result.final_label
    )

    predicted_class = (
        final_class_label
    )


    # --------------------------------------------------------
    # Probability percentages
    # --------------------------------------------------------

    stage1_normal_probability = round(
        stage1_probability_normal * 100,
        2
    )

    stage1_dysfunction_probability = round(
        stage1_probability_non_normal * 100,
        2
    )

    stage1_confidence_percentage = round(
        stage1_confidence * 100,
        2
    )


    # --------------------------------------------------------
    # Four-class Stage 2 probabilities
    # --------------------------------------------------------

    stage2_normal_probability = None
    stage2_hyperthyroid_probability = None
    stage2_hypothyroid_probability = None
    stage2_other_probability = None

    if stage2_probabilities is not None:

        stage2_normal_probability = round(
            stage2_probabilities[0] * 100,
            2
        )

        stage2_hyperthyroid_probability = round(
            stage2_probabilities[1] * 100,
            2
        )

        stage2_hypothyroid_probability = round(
            stage2_probabilities[2] * 100,
            2
        )

        stage2_other_probability = round(
            stage2_probabilities[3] * 100,
            2
        )


    # --------------------------------------------------------
    # Patient-level SHAP explanation for enhanced Stage 1
    # --------------------------------------------------------

    top_shap_features = []
    shap_summary = "SHAP explanation unavailable."

    try:

        # Exact verified path to the underlying trained XGBoost:
        #
        # CalibratedClassifierCV
        #     -> calibrated_classifiers_[0]
        #     -> FrozenEstimator
        #     -> XGBClassifier
        #
        stage1_xgb = (
            enhanced_cascade
            .stage1_model
            .calibrated_classifiers_[0]
            .estimator
            .estimator
        )

        stage1_explainer = shap.TreeExplainer(
            stage1_xgb
        )

        shap_values = stage1_explainer.shap_values(
            sample
        )

        # Verified SHAP output for the current binary XGBoost:
        # shape = (1, 56)
        if hasattr(shap_values, "values"):
            shap_array = shap_values.values
        else:
            shap_array = shap_values

        shap_array = (
            pd.DataFrame(shap_array)
            .iloc[0]
            .to_numpy()
        )

        if len(shap_array) != len(sample.columns):
            raise ValueError(
                "SHAP feature count does not match processed feature count."
            )

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

        shap_summary = "; ".join(
            [
                (
                    f"{feature['Feature']}: "
                    f"{feature['SHAP_Value']:.4f}"
                )
                for feature in top_shap_features
            ]
        )

    except Exception as exc:

        top_shap_features = []
        shap_summary = "SHAP explanation unavailable."

        print(
            "SHAP ERROR:",
            type(exc).__name__,
            str(exc)
        )


    # --------------------------------------------------------
    # Compatibility values for the existing SQLite schema
    # --------------------------------------------------------

    negative_probability = (
        stage1_normal_probability
    )

    hypothyroid_probability = (
        stage1_dysfunction_probability
    )


    # --------------------------------------------------------
    # Save prediction
    # --------------------------------------------------------

    save_prediction(
        session["user_id"],
        predicted_class,
        negative_probability,
        hypothyroid_probability,
        shap_summary,
        stage1_class,
        stage1_confidence,
        routing_decision,
        stage2_used,
        stage2_normal_probability,
        stage2_hyperthyroid_probability,
        stage2_hypothyroid_probability,
        stage2_other_probability
    )


    # --------------------------------------------------------
    # Render result page
    # --------------------------------------------------------

    return render_template(
        "result.html",

        # Final prediction
        predicted_class=predicted_class,
        final_class=final_class_label,
        final_class_id=final_class_id,
        final_class_label=final_class_label,

        # Stage 1
        stage1_class=stage1_class,
        stage1_normal_probability=stage1_normal_probability,
        stage1_dysfunction_probability=stage1_dysfunction_probability,
        stage1_confidence=stage1_confidence,
        stage1_confidence_percentage=stage1_confidence_percentage,

        # Routing
        routing_decision=routing_decision,
        stage2_used=stage2_used,

        # Stage 2
        stage2_class=stage2_class_label,
        stage2_class_id=stage2_class_id,
        stage2_class_label=stage2_class_label,

        stage2_normal_probability=stage2_normal_probability,
        stage2_hyperthyroid_probability=stage2_hyperthyroid_probability,
        stage2_hypothyroid_probability=stage2_hypothyroid_probability,
        stage2_other_probability=stage2_other_probability,

        # Session
        session_username=session["username"],

        # Explainability
        top_shap_features=top_shap_features
    )


# ============================================================


# ==========================================================
# Download Prediction Report
# ==========================================================

@app.route("/download-report")
def download_report():

    # ------------------------------------------------------
    # Authentication protection
    # ------------------------------------------------------

    if "user_id" not in session:
        return redirect(url_for("login"))

    user_id = session["user_id"]

    # ------------------------------------------------------
    # Retrieve latest prediction for logged-in user
    # ------------------------------------------------------

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
            shap_summary,
            stage1_class,
            stage1_confidence,
            routing_decision,
            stage2_used,
            stage2_normal_probability,
            stage2_hyperthyroid_probability,
            stage2_hypothyroid_probability,
            stage2_other_probability
        FROM predictions
        WHERE user_id = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (user_id,)
    )

    prediction = cursor.fetchone()

    connection.close()

    # ------------------------------------------------------
    # No prediction available
    # ------------------------------------------------------

    if prediction is None:
        return redirect(url_for("history"))

    # ------------------------------------------------------
    # Extract database values
    # ------------------------------------------------------

    (
        prediction_id,
        predicted_class,
        negative_probability,
        hypothyroid_probability,
        created_at,
        shap_summary,
        stage1_class,
        stage1_confidence,
        routing_decision,
        stage2_used,
        stage2_normal_probability,
        stage2_hyperthyroid_probability,
        stage2_hypothyroid_probability,
        stage2_other_probability
    ) = prediction

    # ------------------------------------------------------
    # Create PDF in memory
    # ------------------------------------------------------

    pdf_buffer = BytesIO()

    document = SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TCMAPY2573ReportTitle",
        parent=styles["Title"],
        fontSize=20,
        leading=24,
        alignment=1,
        spaceAfter=18
    )

    heading_style = ParagraphStyle(
        "TCMAPY2573ReportHeading",
        parent=styles["Heading2"],
        fontSize=13,
        leading=16,
        spaceBefore=12,
        spaceAfter=8
    )

    normal_style = ParagraphStyle(
        "TCMAPY2573ReportNormal",
        parent=styles["BodyText"],
        fontSize=10,
        leading=14
    )

    story = []

    # ------------------------------------------------------
    # Title
    # ------------------------------------------------------

    story.append(
        Paragraph(
            "TCMAPY2573 - Thyroid Prediction Report",
            title_style
        )
    )

    story.append(
        Paragraph(
            f"Prediction ID: {prediction_id}",
            normal_style
        )
    )

    story.append(
        Paragraph(
            f"User: {session.get('username', 'User')}",
            normal_style
        )
    )

    story.append(
        Paragraph(
            f"Prediction Date: {created_at}",
            normal_style
        )
    )

    # ------------------------------------------------------
    # Final Prediction
    # ------------------------------------------------------

    story.append(
        Paragraph(
            "Final Prediction",
            heading_style
        )
    )

    final_data = [
        ["Parameter", "Value"],
        ["Predicted Class", str(predicted_class)],
        [
            "Normal / Negative Probability",
            f"{float(negative_probability):.2f}%"
        ],
        [
            "Dysfunction Probability",
            f"{float(hypothyroid_probability):.2f}%"
        ]
    ]

    story.append(
        Table(
            final_data,
            colWidths=[250, 250],
            style=TableStyle([
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("PADDING", (0, 0), (-1, -1), 6)
            ])
        )
    )

    # ------------------------------------------------------
    # Stage 1
    # ------------------------------------------------------

    story.append(
        Paragraph(
            "Stage 1 - Confidence-Guided Screening",
            heading_style
        )
    )

    stage1_data = [
        ["Parameter", "Value"],
        ["Stage 1 Class", str(stage1_class)],
        [
            "Stage 1 Confidence",
            f"{float(stage1_confidence) * 100:.2f}%"
        ],
        [
            "Normal Probability",
            f"{float(negative_probability):.2f}%"
        ],
        [
            "Non-Normal Probability",
            f"{float(hypothyroid_probability):.2f}%"
        ],
        ["Routing Decision", str(routing_decision)]
    ]

    story.append(
        Table(
            stage1_data,
            colWidths=[250, 250],
            style=TableStyle([
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("PADDING", (0, 0), (-1, -1), 6)
            ])
        )
    )

    # ------------------------------------------------------
    # Stage 2
    # ------------------------------------------------------

    story.append(
        Paragraph(
            "Stage 2 - Multi-Class Diagnosis",
            heading_style
        )
    )

    stage2_data = [
        ["Parameter", "Value"],
        [
            "Stage 2 Used",
            "Yes" if stage2_used else "No"
        ]
    ]

    if stage2_used:

        stage2_data.extend([
            [
                "Normal Probability",
                f"{float(stage2_normal_probability):.2f}%"
            ],
            [
                "Hyperthyroid Probability",
                f"{float(stage2_hyperthyroid_probability):.2f}%"
            ],
            [
                "Hypothyroid Probability",
                f"{float(stage2_hypothyroid_probability):.2f}%"
            ],
            [
                "Other / Related Probability",
                f"{float(stage2_other_probability):.2f}%"
            ]
        ])

    else:

        stage2_data.append(
            ["Stage 2 Probabilities", "Not used"]
        )

    story.append(
        Table(
            stage2_data,
            colWidths=[250, 250],
            style=TableStyle([
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("PADDING", (0, 0), (-1, -1), 6)
            ])
        )
    )

    # ------------------------------------------------------
    # SHAP Explanation
    # ------------------------------------------------------

    story.append(
        Paragraph(
            "Explainable AI - SHAP Summary",
            heading_style
        )
    )

    safe_shap = str(
        shap_summary
        if shap_summary
        else "SHAP explanation unavailable."
    )

    safe_shap = (
        safe_shap
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )

    story.append(
        Paragraph(
            safe_shap,
            normal_style
        )
    )

    # ------------------------------------------------------
    # Medical disclaimer
    # ------------------------------------------------------

    story.append(
        Paragraph(
            "Disclaimer: This report contains a machine-learning "
            "prediction generated by the TCMAPY2573 thyroid "
            "prediction system. It is not a medical diagnosis "
            "and should not replace professional medical advice.",
            normal_style
        )
    )

    # ------------------------------------------------------
    # Generate PDF
    # ------------------------------------------------------

    document.build(story)

    pdf_buffer.seek(0)

    return send_file(
        pdf_buffer,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=(
            f"TCMAPY2573_Prediction_Report_{prediction_id}.pdf"
        )
    )


# Run Flask
# ============================================================


# ============================================================
# Healthcare Finder
# ============================================================

@app.route("/healthcare")
def healthcare():

    if "user_id" not in session:
        return redirect(url_for("login"))

    return render_template(
        "healthcare.html",
        session_username=session["username"]
    )


# Thyroid Information & Education page
@app.route("/thyroid-information")
def thyroid_information():
    if "user_id" not in session:
        return redirect(url_for("login"))

    return render_template(
        "thyroid_information.html",
        session_username=session["username"]
    )

# Prediction Explanation page
@app.route("/prediction-explanation")
def prediction_explanation():
    if "user_id" not in session:
        return redirect(url_for("login"))

    return render_template(
        "prediction_explanation.html",
        session_username=session["username"]
    )

# Consultation Preparation page

@app.route("/thyroid-faq")
def thyroid_faq():
    if "user_id" not in session:
        return redirect(url_for("login"))
    return render_template("thyroid_faq.html")
@app.route("/consultation-preparation")
def consultation_preparation():
    if "user_id" not in session:
        return redirect(url_for("login"))

    return render_template(
        "consultation_preparation.html",
        session_username=session["username"]
    )
if __name__ == "__main__":

    app.run(
        debug=True
    )





