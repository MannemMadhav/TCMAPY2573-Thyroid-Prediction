import sys
from pathlib import Path

# Make project root importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)

from app import app


# ============================================================
# Create Flask test client
# ============================================================

client = app.test_client()


# ============================================================
# Create logged-in test session
# ============================================================

with client.session_transaction() as session:
    session["user_id"] = 1
    session["username"] = "testuser"


# ============================================================
# Open prediction history
# ============================================================

response = client.get(
    "/history"
)


# ============================================================
# Check result
# ============================================================

page = response.get_data(
    as_text=True
)

print("=" * 60)
print("PREDICTION HISTORY TEST")
print("=" * 60)

print("\nHistory status:")
print(response.status_code)

print("\nHistory page loaded:")
print(response.status_code == 200)

print("\nUpdated labels present:")
print(
    "Stage 1 Normal Probability" in page
    and
    "Stage 1 Dysfunction Probability" in page
)