import json
import sqlite3
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / "predictions.db"

app = Flask(__name__)
model = joblib.load(BASE_DIR / "adaboost_model.pkl")

# Fallback used when the saved model does not expose feature names.
DEFAULT_FEATURE_NAMES = [
    "Age",
    "Gender",
    "Travel History",
    "Temp",
    "SPO2",
    "Contact to NCOVID Patient",
    "Respiratory Support",
    "Respiratory rate(breaths per minute)",
    "BMI",
    "O2 supplementation required",
    "bp_systolic",
    "bp_diastolic",
    "heart_rate",
    "Platelets",
    "neutrophils",
    "lymphocytes",
    "monocytes",
    "eosinophils",
    "basophils",
    "crp",
    "troponin",
    "d_dimer",
    "lactate",
    "ldh",
    "cpk",
    "alt",
    "ast",
    "bilirubin",
    "creatinine",
    "urea",
    "sodium",
    "potassium",
    "chloride",
    "bicarbonate",
    "calcium",
    "magnesium",
    "phosphate",
    "glucose",
    "bodyache",
    "breathlessness",
    "cough",
    "fever",
    "headache",
    "sore throat",
    "asymptomatic",
    "cold",
    "malaise",
    "myalgia",
]
FEATURE_NAMES = list(getattr(model, "feature_names_in_", DEFAULT_FEATURE_NAMES))


def get_db_connection():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    with get_db_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_name TEXT NOT NULL,
                features_json TEXT NOT NULL,
                prediction_value INTEGER NOT NULL,
                prediction_label TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.commit()


def parse_feature_payload(source):
    feature_values = {}

    for feature in FEATURE_NAMES:
        raw_value = source.get(feature, 0)
        if isinstance(raw_value, str):
            raw_value = raw_value.strip()

        try:
            feature_values[feature] = float(raw_value or 0)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid value for '{feature}'. Please enter numbers only."
            ) from exc

    return feature_values


def run_prediction(feature_values):
    final_features = pd.DataFrame([feature_values], columns=FEATURE_NAMES)
    prediction_value = int(model.predict(final_features)[0])
    prediction_label = "Severity: High" if prediction_value == 1 else "Severity: Low"
    return prediction_value, prediction_label


def serialize_record(row):
    features = json.loads(row["features_json"])
    return {
        "id": row["id"],
        "patient_name": row["patient_name"],
        "features": features,
        "prediction_value": row["prediction_value"],
        "prediction_label": row["prediction_label"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def fetch_prediction(record_id):
    with get_db_connection() as connection:
        row = connection.execute(
            "SELECT * FROM predictions WHERE id = ?", (record_id,)
        ).fetchone()
    return row


@app.route("/")
def home():
    return render_template("index.html", feature_names=FEATURE_NAMES)


@app.route("/favicon.ico")
def favicon():
    return ("", 204)


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/api/records", methods=["GET"])
def list_records():
    with get_db_connection() as connection:
        rows = connection.execute(
            "SELECT * FROM predictions ORDER BY updated_at DESC, id DESC"
        ).fetchall()

    return jsonify({"records": [serialize_record(row) for row in rows]})


@app.route("/api/records/<int:record_id>", methods=["GET"])
def get_record(record_id):
    row = fetch_prediction(record_id)
    if row is None:
        return jsonify({"error": "Record not found."}), 404

    return jsonify({"record": serialize_record(row)})


@app.route("/predict", methods=["POST"])
def create_prediction():
    payload = request.get_json(silent=True) or request.form.to_dict()
    patient_name = str(payload.get("patient_name", "")).strip()

    if not patient_name:
        return jsonify({"error": "Patient name is required."}), 400

    try:
        feature_values = parse_feature_payload(payload)
        prediction_value, prediction_label = run_prediction(feature_values)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {exc}"}), 500

    timestamp = datetime.utcnow().isoformat(timespec="seconds")
    with get_db_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO predictions (
                patient_name,
                features_json,
                prediction_value,
                prediction_label,
                created_at,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                patient_name,
                json.dumps(feature_values),
                prediction_value,
                prediction_label,
                timestamp,
                timestamp,
            ),
        )
        connection.commit()
        record_id = cursor.lastrowid

    row = fetch_prediction(record_id)
    return (
        jsonify(
            {
                "message": "Prediction saved successfully.",
                "prediction": prediction_label,
                "record": serialize_record(row),
            }
        ),
        201,
    )


@app.route("/api/records/<int:record_id>", methods=["PUT"])
def update_record(record_id):
    existing = fetch_prediction(record_id)
    if existing is None:
        return jsonify({"error": "Record not found."}), 404

    payload = request.get_json(silent=True) or request.form.to_dict()
    patient_name = str(payload.get("patient_name", "")).strip()

    if not patient_name:
        return jsonify({"error": "Patient name is required."}), 400

    try:
        feature_values = parse_feature_payload(payload)
        prediction_value, prediction_label = run_prediction(feature_values)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {exc}"}), 500

    timestamp = datetime.utcnow().isoformat(timespec="seconds")
    with get_db_connection() as connection:
        connection.execute(
            """
            UPDATE predictions
            SET patient_name = ?,
                features_json = ?,
                prediction_value = ?,
                prediction_label = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                patient_name,
                json.dumps(feature_values),
                prediction_value,
                prediction_label,
                timestamp,
                record_id,
            ),
        )
        connection.commit()

    row = fetch_prediction(record_id)
    return jsonify(
        {
            "message": "Record updated successfully.",
            "prediction": prediction_label,
            "record": serialize_record(row),
        }
    )


@app.route("/api/records/<int:record_id>", methods=["DELETE"])
def delete_record(record_id):
    existing = fetch_prediction(record_id)
    if existing is None:
        return jsonify({"error": "Record not found."}), 404

    with get_db_connection() as connection:
        connection.execute("DELETE FROM predictions WHERE id = ?", (record_id,))
        connection.commit()

    return jsonify({"message": "Record deleted successfully."})


init_db()


if __name__ == "__main__":
    # Bind to localhost during development to avoid stray network probes
    # hitting the Flask dev server and showing noisy TLS/HTTP errors.
    app.run(debug=True, host="127.0.0.1", port=5000)
