import requests

url = 'http://127.0.0.1:5000/predict'
data = {
    "Pregnancies": 2,
    "Glucose": 120,
    "BloodPressure": 70,
    "SkinThickness": 30,
    "Insulin": 100,
    "BMI": 25.0,
    "DiabetesPedigreeFunction": 0.5,
    "Age": 28
}

response = requests.post(url, json=data)
print(response.json())
