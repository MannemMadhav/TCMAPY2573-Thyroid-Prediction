import sqlite3

DATABASE = "thyroid_predictions.db"


def create_database():
    connection = sqlite3.connect(DATABASE)
    cursor = connection.cursor()

    # --------------------------------------------------------
    # Predictions table
    # --------------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            predicted_class TEXT NOT NULL,
            negative_probability REAL NOT NULL,
            hypothyroid_probability REAL NOT NULL,
            shap_summary TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # --------------------------------------------------------
    # Check existing prediction columns
    # --------------------------------------------------------

    cursor.execute("PRAGMA table_info(predictions)")
    prediction_columns = [column[1] for column in cursor.fetchall()]

    if "user_id" not in prediction_columns:
        cursor.execute(
            "ALTER TABLE predictions ADD COLUMN user_id INTEGER"
        )

        cursor.execute("""
            UPDATE predictions
            SET user_id = (
                SELECT id
                FROM users
                ORDER BY id
                LIMIT 1
            )
            WHERE user_id IS NULL
        """)

    if "shap_summary" not in prediction_columns:
        cursor.execute(
            "ALTER TABLE predictions ADD COLUMN shap_summary TEXT"
        )

    # --------------------------------------------------------
    # Users table
    # --------------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    connection.commit()
    connection.close()

    print("Database setup completed!")
    print("Predictions table ready!")
    print("Users table ready!")
    print("shap_summary column ready!")


if __name__ == "__main__":
    create_database()