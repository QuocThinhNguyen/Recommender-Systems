# predict.py
from flask import Flask, request, jsonify
from datetime import datetime
import pandas as pd
import pickle
from surprise import Reader, Dataset
from utils import preprocess_data
from flask_cors import cross_origin
import requests

app = Flask(__name__)

# Load mô hình đã huấn luyện
with open("svd_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/recommend", methods=["POST"])
@cross_origin(origins=["http://localhost:5173"], supports_credentials=True)
def recommend():
    patient_id = request.json.get("patient_id")
    # print("Patient ID:", patient_id)
    # data = request.json.get("data")

    url = "http://localhost:9000/user/suggestions?limit=100"
    response = requests.get(url)
    if response:
        print("Data fetched successfully from the external service.")
    else:
        print("Failed to fetch data from the external service.")


    data = response.json()['data']

    print(data) 

#     data = [
#     {'patient_id': 1, 'doctor_id': 101, 'specialty_id': 1, 'visits': 3, 'rating': 4.5, 'last_visit_date': '2025-04-11', 'click_count': 20},
#     {'patient_id': 1, 'doctor_id': 102, 'specialty_id': 2, 'visits': 1, 'rating': 4.0, 'last_visit_date': '2025-04-11', 'click_count': 40},
#     {'patient_id': 2, 'doctor_id': 105, 'specialty_id': 1, 'visits': 2, 'rating': 5.0, 'last_visit_date': '2025-01-20', 'click_count': 1},
#     {'patient_id': 3, 'doctor_id': 103, 'specialty_id': 2, 'visits': 4, 'rating': 4.8, 'last_visit_date': '2025-03-24', 'click_count': 7},
#     {'patient_id': 3, 'doctor_id': 104, 'specialty_id': 3, 'visits': 1, 'rating': 4.8, 'last_visit_date': '2025-03-25', 'click_count': 10},
#     {'patient_id': 4, 'doctor_id': 106, 'specialty_id': 2, 'visits': 0, 'rating': 5.0, 'last_visit_date': None, 'click_count': 1},
#     {'patient_id': 4, 'doctor_id': 107, 'specialty_id': 3, 'visits': 0, 'rating': 4.9, 'last_visit_date': None, 'click_count': 10},
# ]

    df = preprocess_data(data)
    # df['last_visit_date'] = pd.to_datetime(df['last_visit_date'], errors='coerce')

    print("Data after preprocessing:")
    print(df.head())

    all_doctor_ids = df['doctor_id'].unique().tolist()
    known_doctor_ids = df[df['patient_id'] == patient_id]['doctor_id'].tolist()
    unseen_doctor_ids = [doc_id for doc_id in all_doctor_ids if doc_id not in known_doctor_ids]

    specialty_df = df[df['patient_id'] == patient_id]
    top_specialty = None
    if not specialty_df.empty:
        top_specialty = specialty_df.groupby('specialty_id')[['visits','click_count']].sum().sum(axis=1).idxmax()

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
            # recent_date = recent_date.split('T')[0]
            # recent_date = str(recent_date).split('T')[0]  # Đảm bảo recent_date là chuỗi
            # recent_date = datetime.strptime(recent_date, '%Y-%m-%d')
            # days_ago = (datetime.now() - recent_date).days
            # recent_bonus = max(0, (30 - days_ago) / 30) * 0.2
            # bonus += recent_bonus
            recent_date = recent_date.tz_localize(None) 
            days_ago = (datetime.now() - recent_date).days
            recent_bonus = max(0, (30 - days_ago) / 30) * 0.2
            bonus += recent_bonus

        if doc_id not in known_doctor_ids:
            bonus += 0.1

        predictions.append((doc_id, pred.est + bonus))

    predictions.sort(key=lambda x: x[1], reverse=True)
    # plot_predictions(predictions[:8])  # Vẽ top 8 doctor được đề xuất

    results = [{'doctor_id': doc_id, 'predicted_rating': round(score, 2)} for doc_id, score in predictions[:8]]
    return jsonify({
        "status": 200,
        "message": "Success",
        "data": results
    })

import matplotlib.pyplot as plt

def plot_predictions(predictions):
    doctor_ids = [str(doc_id) for doc_id, _ in predictions]
    scores = [round(score, 2) for _, score in predictions]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(doctor_ids, scores, color='lightgreen')
    plt.title('Top Recommended Doctors')
    plt.xlabel('Doctor ID')
    plt.ylabel('Predicted Rating')
    plt.ylim(0, 6)

    for bar, score in zip(bars, scores):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.1, f'{score}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig("recommendation_chart.png")
    plt.close()


if __name__ == '__main__':
    app.run(port=5000)
