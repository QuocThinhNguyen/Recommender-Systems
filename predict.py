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
    specialist_symptoms = request.json.get("specialist_symptoms") or []

    print("patient_id:", patient_id)
    print("specialist_symptoms:", specialist_symptoms)    

    BASE_API_URL = os.environ.get("BACKEND_URL")
    url = f"{BASE_API_URL}/user/suggestions?limit=500"
    response = requests.get(url)
    if not response:
        return jsonify({"status": 500, "message": "Failed to fetch data"}), 500

    data = response.json()['data']
    # print("Check get data:",data)
    df = preprocess_data(data)
    
    df['last_visit_date'] = pd.to_datetime(df['last_visit_date'], errors='coerce')
    all_doctor_ids = df['doctor_id'].unique().tolist()
    print("Check all_doctor_ids:", all_doctor_ids)
    known_doctor_ids = df[df['patient_id'] == patient_id]['doctor_id'].tolist()
    is_new_user = len(known_doctor_ids) == 0

    results = []

    # Trường hợp 1: Người dùng mới & không chọn chuyên khoa
    if is_new_user and not specialist_symptoms:
        print("=> Case: Cold start hoàn toàn - fallback")
        top_doctors_df = (
            df.groupby('doctor_id')[['visits', 'click_count']]
            .sum()
            .sum(axis=1)
            .sort_values(ascending=False)
            .head(8)
        )
        results = [{'doctor_id': int(doc_id), 'predicted_rating': 4.0} for doc_id in top_doctors_df.index]
        print ("Check kết quả 1:", results)
        return jsonify({
            "status": 200,
            "message": "Gợi ý mặc định cho người dùng mới chưa chọn chuyên khoa",
            "data": results
        })

    # Trường hợp 2: Người dùng mới nhưng có chọn chuyên khoa → lọc bác sĩ theo chuyên khoa
    if is_new_user and specialist_symptoms:
        print("=> Case: Người dùng mới + có chọn chuyên khoa")
        doctor_ids = df[df['specialty_id'].isin(specialist_symptoms)]['doctor_id'].unique().tolist()
        top_doctors_df = (
            df[df['doctor_id'].isin(doctor_ids)]
            .groupby('doctor_id')[['visits', 'click_count']]
            .sum()
            .sum(axis=1)
            .sort_values(ascending=False)
            .head(8)
        )
        results = [{'doctor_id': int(doc_id), 'predicted_rating': 4.0} for doc_id in top_doctors_df.index]
        print ("Check kết quả 2:", results)
        return jsonify({
            "status": 200,
            "message": "Gợi ý theo chuyên khoa cho người dùng mới",
            "data": results
        })

    # Trường hợp 3: Người dùng cũ
    print("=> Case: Người dùng cũ")
    specialty_df = df[df['patient_id'] == patient_id]
    top_specialty = None
    if not specialty_df.empty:
        top_specialty = specialty_df.groupby('specialty_id')[['visits','click_count']].sum().sum(axis=1).idxmax()
    top_specialty_doctors = df[df['specialty_id'] == top_specialty]['doctor_id'].unique().tolist()

    predictions = []
    for doc_id in all_doctor_ids:
        pred = model.predict(patient_id, doc_id)
        bonus = 0.0

        # Ưu tiên bác sĩ cùng chuyên khoa tương tác nhiều gần đây
        if doc_id in top_specialty_doctors:
            bonus += 0.3

        # Ưu tiên bác sĩ có chuyên khoa đang quan tâm
        doctor_specialty = df[df['doctor_id'] == doc_id]['specialty_id'].iloc[0]
        if doctor_specialty in specialist_symptoms:
            bonus += 2

        # Ưu tiên bác sĩ có khám gần đây
        recent_date = df[df['doctor_id'] == doc_id]['last_visit_date'].max()
        if pd.notnull(recent_date):
            recent_date = recent_date.tz_localize(None) 
            days_ago = (datetime.now() - recent_date).days
            recent_bonus = max(0, (30 - days_ago) / 30) * 0.1
            bonus += recent_bonus

        # Ưu tiên bác sĩ chưa từng khám
        if doc_id not in known_doctor_ids:
            bonus += 0.15

        predictions.append((doc_id, pred.est + bonus))

    predictions.sort(key=lambda x: x[1], reverse=True)
    results = [{'doctor_id': doc_id, 'predicted_rating': round(score, 2)} for doc_id, score in predictions[:8]]

    print ("Check kết quả 3:", results)

    return jsonify({
        "status": 200,
        "message": "Gợi ý cho người dùng cũ có lịch sử",
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

    reader = Reader(rating_scale=(1, 5))
    df = df.dropna(subset=['final_rating'])
    dataset = Dataset.load_from_df(df[['patient_id', 'doctor_id', 'final_rating']], reader)
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
