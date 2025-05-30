# predict.py
from flask import Flask, request, jsonify
from datetime import datetime
import pandas as pd
import pickle
from surprise import Reader, Dataset
from utils import preprocess_data
import os
from flask_cors import cross_origin
import requests
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)

# Load mô hình đã huấn luyện
with open("svd_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/recommend", methods=["POST"])
@cross_origin(origins=[os.environ.get("FRONTEND_URL")], supports_credentials=True)
def recommend():
    patient_id = request.json.get("patient_id")

    BASE_API_URL = os.environ.get("BACKEND_URL")
    # print("BASE_API_URL:", BASE_API_URL)

    url = f"{BASE_API_URL}/user/suggestions?limit=100"
    response = requests.get(url)
    if response:
        print("Data fetched successfully from the external service.")
    else:
        print("Failed to fetch data from the external service.")

    data = response.json()['data']

    # print(data) 

    df = preprocess_data(data)

    all_doctor_ids = df['doctor_id'].unique().tolist()
    # print("All doctor IDs:", all_doctor_ids)
    known_doctor_ids = df[df['patient_id'] == patient_id]['doctor_id'].tolist()
    unseen_doctor_ids = [doc_id for doc_id in all_doctor_ids if doc_id not in known_doctor_ids]

    specialty_df = df[df['patient_id'] == patient_id]
    top_specialty = None
    if not specialty_df.empty:
        top_specialty = specialty_df.groupby('specialty_id')[['visits','click_count']].sum().sum(axis=1).idxmax()
    print("Top specialty ID:", top_specialty)
    top_specialty_doctors = df[df['specialty_id'] == top_specialty]['doctor_id'].unique().tolist()

    predictions = []
    for doc_id in all_doctor_ids:
        pred = model.predict(patient_id, doc_id)
        bonus = 0.0

        if doc_id in top_specialty_doctors:
            bonus += 0.4

        df['last_visit_date'] = pd.to_datetime(df['last_visit_date'], errors='coerce')
        recent_date = df[df['doctor_id'] == doc_id]['last_visit_date'].max()
        if pd.notnull(recent_date):
            recent_date = recent_date.tz_localize(None) 
            days_ago = (datetime.now() - recent_date).days
            recent_bonus = max(0, (30 - days_ago) / 30) * 0.1
            bonus += recent_bonus

        if doc_id not in known_doctor_ids:
            bonus += 0.15

        predictions.append((doc_id, pred.est + bonus))

    predictions.sort(key=lambda x: x[1], reverse=True)
    # plot_predictions(predictions[:8])  # Vẽ top 8 doctor được đề xuất

    results = [{'doctor_id': doc_id, 'predicted_rating': round(score, 2)} for doc_id, score in predictions[:8]]
    return jsonify({
        "status": 200,
        "message": "Success",
        "data": results
    })

# import matplotlib.pyplot as plt

# def plot_predictions(predictions):
#     doctor_ids = [str(doc_id) for doc_id, _ in predictions]
#     scores = [round(score, 2) for _, score in predictions]

#     plt.figure(figsize=(10, 6))
#     bars = plt.bar(doctor_ids, scores, color='lightgreen')
#     plt.title('Top Recommended Doctors')
#     plt.xlabel('Doctor ID')
#     plt.ylabel('Predicted Rating')
#     plt.ylim(0, 6)

#     for bar, score in zip(bars, scores):
#         yval = bar.get_height()
#         plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.1, f'{score}', ha='center', va='bottom')

#     plt.tight_layout()
#     plt.savefig("recommendation_chart.png")
#     plt.close()

from surprise import accuracy

@app.route("/evaluate", methods=["GET"])
def evaluate():
    BASE_API_URL = os.environ.get("BACKEND_URL")
    url = f"{BASE_API_URL}/user/suggestions?limit=100"
    response = requests.get(url)
    data = response.json()['data']

    df = preprocess_data(data)

    reader = Reader(rating_scale=(0, 5))
    df = df.dropna(subset=['rating'])
    dataset = Dataset.load_from_df(df[['patient_id', 'doctor_id', 'rating']], reader)
    trainset = dataset.build_full_trainset()

    predictions = model.test(trainset.build_testset())

    rmse = accuracy.rmse(predictions)
    mae = accuracy.mae(predictions)

    return jsonify({
        "status": 200,
        "message": "Evaluation success",
        "rmse": round(rmse, 4),
        "mae": round(mae, 4)
    })


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
