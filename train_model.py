# train_model.py
from surprise import Dataset, Reader, SVD
from utils import preprocess_data
import requests
import json
import pickle
# url = "http://localhost:9000/user/suggestions"
# response = requests.get(url)
# if response:
#     print("Data fetched successfully from the external service.")
# else:
#     print("Failed to fetch data from the external service.")


# data = response.json()['data']
# data = [
#     {'patient_id': 1, 'doctor_id': 101, 'specialty_id': 1, 'visits': 3, 'rating': 4.5, 'last_visit_date': '2025-04-11', 'click_count': 20},
#     {'patient_id': 1, 'doctor_id': 102, 'specialty_id': 2, 'visits': 1, 'rating': 4.0, 'last_visit_date': '2025-04-11', 'click_count': 10},
#     {'patient_id': 2, 'doctor_id': 105, 'specialty_id': 1, 'visits': 2, 'rating': 5.0, 'last_visit_date': '2025-01-20', 'click_count': 1},
#     {'patient_id': 3, 'doctor_id': 103, 'specialty_id': 2, 'visits': 4, 'rating': 4.8, 'last_visit_date': '2025-03-24', 'click_count': 7},
#     {'patient_id': 3, 'doctor_id': 104, 'specialty_id': 3, 'visits': 1, 'rating': 4.8, 'last_visit_date': '2025-03-25', 'click_count': 10},
# ]

with open("patient_doctor_data.json","r") as f:
    data = json.load(f)

df = preprocess_data(data)

reader = Reader(rating_scale=(1, 5))
dataset = Dataset.load_from_df(df[['patient_id', 'doctor_id', 'final_rating']], reader)
trainset = dataset.build_full_trainset()
model = SVD()
model.fit(trainset)

with open("svd_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model training completed and saved as 'svd_model.pkl'.")