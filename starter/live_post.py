import requests


url = "https://ml-fastapi-lingjia-5e077637aff3.herokuapp.com/inference"

payload = {
    "age": 37,
    "capital-gain": 2174,
    "capital-loss": 0,
    "education": "Bachelors",
    "education-num": 13,
    "fnlgt": 77516,
    "hours-per-week": 40,
    "marital-status": "Married-civ-spouse",
    "native-country": "United-States",
    "occupation": "Tech-support",
    "race": "White",
    "relationship": "Husband",
    "sex": "Male",
    "workclass": "Private",
}

response = requests.post(url, json=payload)

print("Status code:", response.status_code)
print("Response:", response.json())
