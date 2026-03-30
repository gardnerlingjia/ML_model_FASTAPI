# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model is a supervised machine learning classification model based on a Random Forest Classifier implemented using scikit-learn.

The model predicts whether an individual’s income exceeds $50K per year based on demographic and employment-related features.

Model type: Random Forest Classifier
Framework: scikit-learn
Pipeline components:
OneHotEncoder for categorical features
LabelBinarizer for target encoding
Deployment: Designed to be served via a FastAPI REST API

## Intended Use
This model is intended for:

Educational purposes (Udacity ML pipeline project)
Demonstrating end-to-end ML workflows:
data processing
model training
evaluation
API deployment

It may also be used as a reference implementation for:

structured tabular classification problems
MLOps pipelines

❗ Not intended for:

real-world decision-making (e.g., hiring, lending, or legal decisions)
production use without further validation, fairness checks, and monitoring

## Training Data
The model is trained on the Census Income dataset (also known as Adult dataset), originally from the UCI Machine Learning Repository.

The dataset includes features such as:

age
workclass
education
marital-status
occupation
relationship
race
sex
native-country

The target variable is:

income (<=50K or >50K)

Data preprocessing includes:

One-hot encoding of categorical variables
No explicit scaling of numerical features

## Evaluation Data
The dataset is split into:

Training set: 80%
Test set: 20%

using train_test_split.

The test set is used to evaluate model generalization performance.

## Metrics
_Please include the metrics used and your model's performance on those metrics._
The model is evaluated using:

Precision
Recall
F1 Score (fbeta with β=1)

Example performance (may vary depending on split):

Precision: ~0.70–0.75
Recall: ~0.60–0.70
F1 Score: ~0.65–0.72

These metrics are computed using functions from scikit-learn.

## Ethical Considerations
This model uses demographic features such as:

race
sex
marital status

which may introduce or reinforce bias.

Potential risks include:

discrimination if used in real-world decision systems
amplification of historical societal biases present in the dataset

⚠️ The model should not be used in sensitive contexts such as:

hiring decisions
financial approvals
legal or governmental systems

## Caveats and Recommendations
Limitations
The dataset is outdated and may not reflect current socioeconomic conditions
No fairness or bias mitigation techniques are applied
No feature scaling or hyperparameter tuning performed
Model performance may vary across subgroups
Recommendations
Perform fairness analysis (e.g., across gender and race)
Apply bias mitigation techniques if used beyond demonstration
Tune hyperparameters for improved performance
Add monitoring if deployed in production
Consider more interpretable models for sensitive use cases