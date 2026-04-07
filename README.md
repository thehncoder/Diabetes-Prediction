# Diabetes-Prediction

# 🧠 Diabetes Prediction using Machine Learning

## 📌 Project Overview

This project predicts whether a patient is diabetic or not using machine learning. It covers the full data science pipeline: data analysis, preprocessing, model training, evaluation, and visualization.

---

## 🎯 Objective

To analyze medical data and build a model that classifies patients as:

* **0 → Non-Diabetic**
* **1 → Diabetic**

---

## 📊 Dataset

* **Name:** Pima Indians Diabetes Dataset
* **Records:** 768 patients
* **Features:**

  * Glucose
  * BMI
  * Age
  * Pregnancies
  * Blood Pressure
  * Skin Thickness
  * Insulin
  * Diabetes Pedigree Function
* **Target:** Outcome (0 / 1)

---

## ⚙️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* HTML, CSS, JavaScript (Dashboard)

---

## 🔍 Project Workflow

### 1. Data Collection

* Loaded dataset using Pandas

### 2. Exploratory Data Analysis (EDA)

* Feature distributions
* Class comparison (Diabetic vs Non-Diabetic)
* Correlation heatmap

### 3. Data Preprocessing

* Handled missing values
* Train-test split (80/20)
* Feature scaling (StandardScaler)

### 4. Model Building

* Logistic Regression
* Random Forest
* Gradient Boosting

### 5. Model Evaluation

* Accuracy
* Confusion Matrix
* ROC Curve
* Precision, Recall, F1-score

---

## 🏆 Results

| Model               | Accuracy  | AUC-ROC   |
| ------------------- | --------- | --------- |
| Logistic Regression | 87.0%     | 92.9%     |
| Gradient Boosting   | 87.7%     | 94.7%     |
| Random Forest       | **90.3%** | **95.4%** |

👉 **Best Model: Random Forest**

---

## 📈 Key Insights

* Glucose and Age are the most important features
* Higher BMI increases diabetes risk
* Random Forest gives best performance

---

## 🖥️ Features

* Data analysis & visualization
* Multiple ML models comparison
* Confusion matrix & ROC curve
* Feature importance analysis
* Interactive dashboard

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
python diabetes_project.py
```

---

## 👨‍💻 Developers

* Asim Siddiqui
* Asmit Raj
* Naman Nigam
* Harshit Nigam

---

## 📌 Conclusion

This project demonstrates how data science and machine learning can be used to predict diseases and extract meaningful insights from medical data.

---

MR HN❤️
