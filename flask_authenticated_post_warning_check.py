from pathlib import Path
import warnings
import pandas as pd
import numpy as np
import app as app_module

print("===== STEP 728 - AUTHENTICATED FLASK POST WARNING-FREE REGRESSION =====")

DATA_FILE = Path("data/enhanced_test_with_targets.csv")

required_features = [
    "age", "sex",
    "on_thyroxine", "query_on_thyroxine",
    "on_antithyroid_medication", "sick", "pregnant",
    "thyroid_surgery", "I131_treatment",
    "query_hypothyroid", "query_hyperthyroid",
    "lithium", "goitre", "tumor", "hypopituitary",
    "psych",
    "TSH_measured", "TSH",
    "T3_measured", "T3",
    "TT4_measured", "TT4",
    "T4U_measured", "T4U",
    "FTI_measured", "FTI",
    "TBG_measured", "TBG",
    "referral_source",
]

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
]

measured_fields = [
    "TSH_measured",
    "T3_measured",
    "TT4_measured",
    "T4U_measured",
    "FTI_measured",
    "TBG_measured",
]

numeric_fields = [
    "age", "TSH", "T3", "TT4", "T4U", "FTI", "TBG"
]

print()
print("--- Loading enhanced test data ---")

if not DATA_FILE.exists():
    raise SystemExit("Enhanced test dataset not found.")

df = pd.read_csv(DATA_FILE)

missing = [c for c in required_features if c not in df.columns]
if missing:
    raise SystemExit(f"Missing required columns: {missing}")

print("Dataset rows:", len(df))
print("Required 29 fields: PASS")

# Load the exact Flask application's fitted preprocessing/cascade objects.
preprocessor = app_module.enhanced_preprocessor
cascade = app_module.enhanced_cascade

# Find two valid real samples: one direct-normal and one Stage-2 routed.
candidates = []

for idx, row in df.iterrows():
    try:
        raw = row[required_features].copy()

        for col in numeric_fields:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")

        if pd.isna(raw["age"]):
            continue

        processed = preprocessor.transform(
            pd.DataFrame([raw.to_dict()], columns=required_features)
        )

        result = cascade.predict_one(processed)

        candidates.append((idx, row, result))

    except Exception:
        continue

direct_case = None
stage2_case = None

for item in candidates:
    result = item[2]

    if direct_case is None and result.routing_decision == "FINAL_NORMAL":
        direct_case = item

    if stage2_case is None and result.stage2_used:
        stage2_case = item

    if direct_case is not None and stage2_case is not None:
        break

if direct_case is None:
    raise SystemExit("Could not find a direct-normal case.")

if stage2_case is None:
    raise SystemExit("Could not find a Stage-2 routed case.")

print("Direct-normal case found: PASS")
print("Stage-2 case found: PASS")


def row_to_form(row):
    payload = {}

    # Age
    payload["age"] = str(int(float(row["age"])))

    # Sex: HTML uses 1 = Male, 0 = Female.
    sex = str(row["sex"]).strip().lower()
    if sex in {"m", "male", "1"}:
        payload["sex"] = "1"
    elif sex in {"f", "female", "0"}:
        payload["sex"] = "0"
    else:
        raise ValueError(f"Unsupported sex value: {row['sex']}")

    # Binary clinical fields.
    for field in binary_fields:
        value = str(row[field]).strip().lower()

        if value in {"t", "true", "1", "yes"}:
            payload[field] = "1"
        elif value in {"f", "false", "0", "no"}:
            payload[field] = "0"
        else:
            raise ValueError(f"Unsupported binary value for {field}: {row[field]}")

    # Measurement flags.
    for field in measured_fields:
        value = str(row[field]).strip().lower()

        if value in {"t", "true", "1", "yes"}:
            payload[field] = "1"
        elif value in {"f", "false", "0", "no"}:
            payload[field] = "0"
        else:
            raise ValueError(
                f"Unsupported measurement value for {field}: {row[field]}"
            )

    # Numeric laboratory fields.
    for field in ["TSH", "T3", "TT4", "T4U", "FTI", "TBG"]:
        value = row[field]

        if pd.isna(value):
            # Use a harmless numeric value for the web form.
            # The measured flag still carries the missing-measurement state.
            payload[field] = "0"
        else:
            payload[field] = str(float(value))

    # Referral source.
    referral = str(row["referral_source"]).strip()
    if referral.lower() in {"nan", "", "none"}:
        referral = "other"

    payload["referral_source"] = referral

    return payload


def execute_case(client, label, item):
    idx, row, expected = item
    payload = row_to_form(row)

    print()
    print(f"--- {label} ---")
    print("Test row index:", idx)
    print("Expected routing:", expected.routing_decision)
    print("Expected final label:", expected.final_label)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")

        response = client.post(
            "/predict",
            data=payload,
            follow_redirects=False,
        )

    print("HTTP status:", response.status_code)
    print("Response bytes:", len(response.data))
    print("Warnings captured:", len(captured))

    text = response.get_data(as_text=True)

    checks = {
        "HTTP 200": response.status_code == 200,
        "Final Prediction rendered": "Final Prediction" in text,
        "Routing Decision rendered": "Routing Decision" in text,
        "Stage 2 Used rendered": "Stage 2 Used" in text,
        "No runtime warnings": len(captured) == 0,
    }

    for name, passed in checks.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")

    return all(checks.values())


print()
print("--- Creating authenticated Flask test client ---")

app = app_module.app
client = app.test_client()

with client.session_transaction() as session:
    session["user_id"] = 1
    session["username"] = "testuser"

print("Authenticated session: PASS")

direct_ok = execute_case(
    client,
    "Direct-normal POST /predict",
    direct_case,
)

stage2_ok = execute_case(
    client,
    "Stage-2 POST /predict",
    stage2_case,
)

print()
print("--- Final Step 728 verification ---")

print("Direct-normal POST: " + ("PASS" if direct_ok else "FAIL"))
print("Stage-2 POST: " + ("PASS" if stage2_ok else "FAIL"))

if direct_ok and stage2_ok:
    print()
    print("STEP 728 AUTHENTICATED FLASK POST WARNING-FREE REGRESSION: PASS")
else:
    print()
    print("STEP 728 AUTHENTICATED FLASK POST WARNING-FREE REGRESSION: FAIL")
    raise SystemExit(1)
