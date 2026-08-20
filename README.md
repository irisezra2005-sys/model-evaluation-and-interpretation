# model-evaluation-and-interpretation
Model Evaluation and Interpretation

1. Project Overview

This project focuses on evaluating and interpreting a machine learning classification model. A Random Forest Classifier is trained on the Breast Cancer Wisconsin dataset and evaluated using multiple classification metrics and visualizations.

The project also uses SHAP (SHapley Additive exPlanations) to understand which features contribute most to the model's predictions.

Objectives

- Train a classification model.
- Evaluate model performance using Accuracy, Precision, Recall, and F1-score.
- Generate a Classification Report.
- Analyze predictions using a Confusion Matrix.
- Evaluate classification performance using an ROC Curve and AUC.
- Interpret model predictions using SHAP.
- Generate SHAP Summary and Feature Importance plots.
- Generate a SHAP Force Plot for an individual prediction.

---

2. Dataset

The project uses the Breast Cancer Wisconsin dataset available through Scikit-learn.

The dataset contains:

- 569 samples
- 30 numerical features
- 2 target classes

The features describe characteristics of breast cell nuclei obtained from digitized images.

The dataset is loaded directly using:

from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()

---

3. Project Structure

Model-Evaluation-Interpretation/
│
├── README.md
├── requirements.txt
│
├── notebooks/
│   └── model_interpretation.ipynb
│
├── docs/
│   └── evaluation_report.md
│
└── assets/
    ├── confusion_matrix.png
    ├── roc_curve.png
    ├── shap_summary.png
    ├── shap_feature_importance.png
    └── shap_force.html

---

4. Evaluation Methodology

Step 1: Data Loading

The Breast Cancer Wisconsin dataset is loaded using Scikit-learn.

The feature matrix is stored in "X", while the target variable is stored in "y".

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

The dataset is also checked for missing values and class distribution.

---

Step 2: Train-Test Split

The dataset is divided into training and testing sets.

An 80:20 split is used:

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

The training set is used to train the model, while the test set is reserved for evaluating its performance on unseen data.

---

Step 3: Model Training

A Random Forest Classifier is used as the classification model.

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

The model consists of multiple decision trees whose predictions are combined to produce the final classification.

---

Step 4: Prediction

The trained model generates class predictions for the test data:

y_pred = model.predict(X_test)

Prediction probabilities are also generated for ROC-AUC analysis:

y_prob = model.predict_proba(X_test)[:, 1]

---

5. Model Evaluation Metrics

The following metrics are calculated.

Accuracy

Accuracy measures the proportion of correctly classified samples.

Accuracy = Correct Predictions / Total Predictions

---

Precision

Precision measures how many of the samples predicted as positive are actually positive.

Precision = TP / (TP + FP)

---

Recall

Recall measures how many actual positive samples were correctly identified.

Recall = TP / (TP + FN)

---

F1-Score

F1-score combines Precision and Recall using their harmonic mean.

F1 = 2 × (Precision × Recall) / (Precision + Recall)

---

Classification Report

Scikit-learn's "classification_report()" is used to display:

- Precision
- Recall
- F1-score
- Support
- Accuracy
- Macro average
- Weighted average

---

6. Confusion Matrix

A confusion matrix is generated to examine the individual prediction outcomes.

The four possible outcomes are:

- True Positive (TP)
- True Negative (TN)
- False Positive (FP)
- False Negative (FN)

The visualization is saved as:

assets/confusion_matrix.png

The confusion matrix helps identify which classes are being correctly classified and where the model makes errors.

---

7. ROC Curve and AUC

The Receiver Operating Characteristic (ROC) curve evaluates the model across different classification thresholds.

The curve uses:

- True Positive Rate (TPR)
- False Positive Rate (FPR)

The Area Under the ROC Curve (AUC) is calculated using:

auc_score = roc_auc_score(y_test, y_prob)

The ROC curve is saved as:

assets/roc_curve.png

A higher AUC indicates better ability to distinguish between the two classes.

---

8. SHAP Model Interpretation

SHAP is used to interpret the predictions made by the Random Forest model.

A TreeExplainer is created:

explainer = shap.TreeExplainer(model)

SHAP values are then calculated to measure the contribution of individual features to model predictions.

---

SHAP Summary Plot

The SHAP Summary Plot provides a global view of feature importance.

It helps identify:

- Which features have the greatest influence.
- How feature values affect predictions.
- The overall distribution of feature contributions.

The plot is saved as:

assets/shap_summary.png

---

SHAP Feature Importance

A SHAP bar plot is also generated to provide a simplified ranking of the most influential features.

The output is saved as:

assets/shap_feature_importance.png

---

SHAP Force Plot

A SHAP Force Plot is used for local interpretation.

It explains the contribution of individual features to a particular prediction.

The generated interactive visualization is saved as:

assets/shap_force.html

The force plot helps answer:

«Why did the model make this particular prediction?»

---

9. Reproducibility

A fixed "random_state=42" is used during the train-test split and Random Forest training.

This helps make the experiment reproducible when the notebook is executed again under the same software environment.

---

10. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- SHAP
- Jupyter Notebook

---

11. Evaluation Summary

The project evaluates the Random Forest classifier from multiple perspectives:

Model Performance
        │
        ├── Accuracy
        ├── Precision
        ├── Recall
        ├── F1-score
        ├── Classification Report
        ├── Confusion Matrix
        └── ROC-AUC
                │
                ▼
       Model Interpretation
                │
                ├── SHAP Summary Plot
                ├── SHAP Feature Importance
                └── SHAP Force Plot

This combination provides both quantitative evaluation of the classifier and interpretability of its predictions.
