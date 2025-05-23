from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import pickle
import pandas as pd
from datetime import datetime, timedelta
import os
import time

# Load dataset
dataset_path = 'medication_adherence_data.csv'
df = pd.read_csv(dataset_path)

# Load saved encoders & scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('med_encoder.pkl', 'rb') as f:
    med_encoder = pickle.load(f)

with open('user_encoder.pkl', 'rb') as f:
    user_encoder = pickle.load(f)

# Medicine mapping
medicine_mapping = {
    'Medicine A': 2,
    'Medicine B': 1,
    'Medicine C': 0
}

# Rename columns
column_mapping = {'Patient ID': 'user_id', 'DateTime': 'datetime', 'Medicine Name': 'medicine_name', 'Taken': 'taken', 'Taken_Probability': 'prob'}
df.rename(columns=column_mapping, inplace=True)

# Convert datetime column to pandas datetime format
df['datetime'] = pd.to_datetime(df['datetime'])

# Convert datetime to timestamp
df['timestamp'] = df['datetime'].astype(np.int64) // 10**9

# Extract time-based features
df['day_of_week'] = df['datetime'].dt.dayofweek
df['hour_of_day'] = df['datetime'].dt.hour

# Apply scaler
df['timestamp'] = scaler.transform(df[['timestamp']])

# Apply medicine mapping
df['medicine_name'] = df['medicine_name'].map(medicine_mapping)

# Load the trained model
MODEL_PATH = "model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

app = Flask(__name__)




def get_dynamic_threshold(past_14_days, predictions):
    """
    Adjusts the threshold dynamically based on past 14 days adherence.
    """
    past_true_count = sum(past_14_days)
    adherence_rate = past_true_count / len(past_14_days) if len(past_14_days) > 0 else 0.5
    sorted_probs = sorted(predictions)
    cutoff_index = int(len(sorted_probs) * (1 - adherence_rate))
    threshold = sorted_probs[cutoff_index] if len(sorted_probs) > 0 else 0.5
    return threshold, adherence_rate * 100

# Medicine Mapping (New)
medicine_mapping = {
    'Metformin': 2,
    'Lisinopril': 1,
    'Atorvastatin': 0
}


# Reverse mapping for output (if needed)
reverse_medicine_mapping = {v: k for k, v in medicine_mapping.items()}

# User ID Mapping
user_id_mapping = {
    "44Zy133BvyeG5KgbqgizlPGHruZ2": 1,
    "1rEZSdfXxAdLjJUovHj9B7NpWHo2": 2,
    "3kwM4eekrReeNaaJ77e5MEJ6MNC2": 3,
    "b1OOS3Hb9JczS2PNhaEIcZiwBk73": 4,
    "GU8a2eeI8shEqkw4R1tn3bLKnAt2": 5
}

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    user_string_id = data.get("user_id")  # User ID as a string from requester
    medicines = data.get("medicines")

    # ✅ Accept any medicines, but filter to allowed ones only
    medicines = [med for med in medicines if med in medicine_mapping]

    # Convert string user_id to numeric ID
    user_id = user_id_mapping.get(user_string_id)
    if user_id is None:
        return jsonify({"error": "Invalid user_id."}), 400

    if not isinstance(medicines, list) or len(medicines) == 0:
        return jsonify({"error": "No valid medicines. Accepted: Metformin, Lisinopril, Atorvastatin"}), 400

    encoded_medicines = [medicine_mapping[med] for med in medicines]
    today = datetime.now()
    start_date = today - timedelta(days=14)
    dates = [(today + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, 15)]
    user_df = df[df['user_id'] == user_id]
    if user_df.empty:
        return jsonify({"error": "User not found"}), 404

    predictions_output = []
    sequence_length = 14

    for medicine_encoded in encoded_medicines:
        med_df = user_df[user_df['medicine_name'] == medicine_encoded].sort_values(by='timestamp')
        if med_df.empty:
            continue

        last_14_days = med_df[med_df['datetime'] >= start_date]['taken'].tolist()
        data = med_df[['timestamp', 'day_of_week', 'hour_of_day', 'user_id', 'medicine_name']].values

        # Add today's data to the sequence
        today_timestamp = scaler.transform([[today.timestamp()]])[0][0]
        today_day_of_week = today.weekday()
        today_hour = today.hour
        today_data = np.array([[today_timestamp, today_day_of_week, today_hour, user_id, medicine_encoded]])
        data = np.vstack([data, today_data])

        # Ensure sequence length is 14
        if len(data) > sequence_length:
            data = data[-sequence_length:]
        if len(data) < sequence_length:
            continue

        X_input = np.array([data])
        predictions = model.predict(X_input)[0].flatten().tolist()
        threshold, _ = get_dynamic_threshold(last_14_days, predictions)
        false_predictions = []

        for date, prob in zip(dates, predictions):
            if prob < threshold:
                false_predictions.append({
                    "medicine": reverse_medicine_mapping[medicine_encoded],
                    "date": date,
                    "probability": round(float(prob), 4),
                    "likely_to_take": False
                })
            if len(false_predictions) >= 3:
                break

        predictions_output.extend(false_predictions)

    return jsonify({"predictions": predictions_output})

# Run the API
if __name__ == "__main__":
    app.run(port=5000)
