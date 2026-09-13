import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)

from app import app


# ------------------------------------------------------------
# Create Flask test client
# ------------------------------------------------------------

client = app.test_client()


# ------------------------------------------------------------
# Test form data
# ------------------------------------------------------------

data = {
    "age": "50",
    "sex": "0",
    "on_thyroxine": "0",
    "query_on_thyroxine": "0",
    "on_antithyroid_medication": "0",
    "thyroid_surgery": "0",
    "I131_treatment": "0",
    "hypopituitary": "0",
    "psych": "0",
    "query_hypothyroid": "0",
    "query_hyperthyroid": "0",
    "pregnant": "0",
    "sick": "0",
    "tumor": "0",
    "lithium": "0",
    "goitre": "0",
    "TSH_measured": "1",
    "TSH": "2.0",
    "T3_measured": "1",
    "T3": "1.5",
    "TT4_measured": "1",
    "TT4": "100",
    "T4U_measured": "1",
    "T4U": "80",
    "FTI_measured": "1",
    "FTI": "125"
}


# ------------------------------------------------------------
# Create a logged-in test session
# ------------------------------------------------------------

with client.session_transaction() as session:
    session["user_id"] = 1
    session["username"] = "testuser"


# ------------------------------------------------------------
# Submit prediction
# ------------------------------------------------------------

response = client.post(
    "/predict",
    data=data
)


# ------------------------------------------------------------
# Display result
# ------------------------------------------------------------

print("=" * 60)
print("FLASK HIERARCHICAL PREDICTION TEST")
print("=" * 60)

print("\nStatus code:")
print(response.status_code)

print("\nPrediction page returned:")
print(response.status_code == 200)

print("\nFinal prediction text found:")

page = response.get_data(as_text=True)

print(
    "subnormal" in page
    or "hyperfunction" in page
    or "normal" in page
)