import pandas as pd
import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field, ConfigDict

from starter.starter.ml.data import process_data
from starter.starter.ml.model import inference

app = FastAPI()

model = joblib.load("starter/model/model.pkl")
encoder = joblib.load("starter/model/encoder.pkl")
lb = joblib.load("starter/model/lb.pkl")

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


class CensusData(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "age": 37,
                "workclass": "Private",
                "fnlgt": 77516,
                "education": "Bachelors",
                "education-num": 13,
                "marital-status": "Married-civ-spouse",
                "occupation": "Tech-support",
                "relationship": "Husband",
                "race": "White",
                "sex": "Male",
                "capital-gain": 2174,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States",
            }
        },
    )

    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")


@app.get("/")
def welcome() -> dict[str, str]:
    return {"message": "Welcome to the census income prediction API"}


@app.post("/inference")
def predict(data: CensusData) -> dict[str, str]:
    input_df = pd.DataFrame([data.model_dump(by_alias=True)])

    X, _, _, _ = process_data(
        input_df,
        categorical_features=cat_features,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    pred = inference(model, X)[0]
    prediction = ">50K" if pred == 1 else "<=50K"

    return {"prediction": prediction}
