# Script to train machine learning model.

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd

from starter.starter.ml.data import process_data
from starter.starter.ml.model import compute_model_metrics, inference


def compute_slice_metrics(data, categorical_features, model, encoder, lb):
    for feature in categorical_features:
        for cls in data[feature].unique():
            data_slice = data[data[feature] == cls]

            X_slice, y_slice, _, _ = process_data(
                data_slice,
                categorical_features=categorical_features,
                label="salary",
                training=False,
                encoder=encoder,
                lb=lb,
            )

            preds = inference(model, X_slice)
            precision, recall, fbeta = compute_model_metrics(y_slice, preds)

            print(f"{feature} = {cls}")
            print(
                f"Precision: {precision:.3f}, "
                f"Recall: {recall:.3f}, "
                f"Fbeta: {fbeta:.3f}"
            )


data = pd.read_csv("starter/data/census.csv")
data.columns = data.columns.str.strip()

for col in data.select_dtypes(include=["object"]).columns:
    data[col] = data[col].str.strip()

print(data.columns.tolist())
print(data.head())

train, test = train_test_split(data, test_size=0.20, random_state=42)

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

X_train, y_train, encoder, lb = process_data(
    train,
    categorical_features=cat_features,
    label="salary",
    training=True,
)

X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

model = RandomForestClassifier(
    n_estimators=5,
    max_depth=3,
    random_state=42,
)
model.fit(X_train, y_train)

compute_slice_metrics(test, cat_features, model, encoder, lb)

joblib.dump(model, "starter/model/model.pkl")
joblib.dump(encoder, "starter/model/encoder.pkl")
joblib.dump(lb, "starter/model/lb.pkl")
