
import sqlite3
from pathlib import Path

DB = Path(__file__).resolve().parent / "thyroid_predictions.db"

conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row

cursor = conn.cursor()

# 1. Registered users
print("\n========== ALL REGISTERED USERS ==========")

cursor.execute("""
    SELECT id, username, email, created_at
    FROM users
    ORDER BY id
""")

users = cursor.fetchall()

for user in users:
    print(dict(user))

print("Total users:", len(users))


# 2. All predictions
print("\n========== ALL THYROID PREDICTIONS ==========")

cursor.execute("""
    SELECT *
    FROM predictions
    ORDER BY id DESC
""")

predictions = cursor.fetchall()

for prediction in predictions:
    print(dict(prediction))

print("Total predictions:", len(predictions))


# 3. User-wise prediction details
print("\n========== USER-WISE PREDICTION DETAILS ==========")

cursor.execute("""
    SELECT
        u.id AS user_id,
        u.username,
        u.email,
        p.id AS prediction_id,
        p.predicted_class,
        p.stage1_class,
        p.stage1_confidence,
        p.routing_decision,
        p.stage2_used,
        p.created_at
    FROM users u
    LEFT JOIN predictions p
    ON u.id = p.user_id
    ORDER BY u.id, p.id DESC
""")

for record in cursor.fetchall():
    print(dict(record))

conn.close()

print("\n========== DATABASE CHECK COMPLETED ==========")
