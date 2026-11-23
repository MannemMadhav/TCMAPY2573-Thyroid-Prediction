
import sqlite3
from pathlib import Path

# Always use the database located beside this file
DATABASE = Path(__file__).resolve().parent / "thyroid_predictions.db"


def create_database():

    connection = sqlite3.connect(str(DATABASE))
    cursor = connection.cursor()

    # Users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Predictions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            predicted_class TEXT NOT NULL,
            negative_probability REAL NOT NULL,
            hypothyroid_probability REAL NOT NULL,
            shap_summary TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            stage1_class TEXT,
            stage1_confidence REAL,
            routing_decision TEXT,
            stage2_used INTEGER DEFAULT 0,
            stage2_normal_probability REAL,
            stage2_hyperthyroid_probability REAL,
            stage2_hypothyroid_probability REAL,
            stage2_other_probability REAL
        )
    """)

    # Safely add missing columns to an existing database
    existing_columns = {
        column[1]
        for column in cursor.execute(
            "PRAGMA table_info(predictions)"
        ).fetchall()
    }

    required_columns = {
        "user_id": "INTEGER",
        "shap_summary": "TEXT",
        "stage1_class": "TEXT",
        "stage1_confidence": "REAL",
        "routing_decision": "TEXT",
        "stage2_used": "INTEGER DEFAULT 0",
        "stage2_normal_probability": "REAL",
        "stage2_hyperthyroid_probability": "REAL",
        "stage2_hypothyroid_probability": "REAL",
        "stage2_other_probability": "REAL"
    }

    for column_name, column_type in required_columns.items():

        if column_name not in existing_columns:

            cursor.execute(
                f"ALTER TABLE predictions "
                f"ADD COLUMN {column_name} {column_type}"
            )

    connection.commit()
    connection.close()

    print("Database initialization completed.")
    print("Users table ready.")
    print("Predictions table ready.")
    print("All required prediction columns verified.")


if __name__ == "__main__":
    create_database()
