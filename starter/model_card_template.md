# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This project uses a Random Forest Classifier (scikit-learn) to predict whether a person earns more than $50K per year based on census data.

Model: Random Forest
Framework: scikit-learn
Deployment: FastAPI API

## Intended Use
This model is built for:

Educational purposes (Udacity ML project)
Demonstrating an end-to-end ML pipeline

❗ Not intended for real-world decision-making such as hiring or lending.


## Training Data
TThe model is trained on the Census Income (Adult) dataset.

Features include demographic and work-related attributes such as:

age
education
occupation
marital status
race
sex

Target:

income (<=50K or >50K)

Categorical features are one-hot encoded.

## Evaluation Data
The dataset is split into:

Training set: 80%
Test set: 20%

using train_test_split.


## Metrics
_Please include the metrics used and your model's performance on those metrics._
The model is evaluated using:

Precision
Recall
F1 Score

Example performance (may vary depending on split):

Precision: ~0.72
Recall: ~0.66
F1 Score: ~0.69

These metrics are computed using functions from scikit-learn.

## Ethical Considerations
The model uses sensitive features such as race and sex, which may introduce bias.

It should not be used in high-stakes decisions.

## Caveats and Recommendations
Limitations
No bias mitigation applied
No hyperparameter tuning
Dataset may be outdated

Recommendations
Perform fairness analysis
Improve model tuning
Add monitoring for production use