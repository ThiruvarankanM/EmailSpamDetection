# Spam Email Classification

## Overview
This project implements a machine learning solution to classify emails as **Spam** or **Ham** (not spam) using **Naive Bayes** and **Linear Support Vector Machine (Linear SVM)** algorithms. The goal is to automatically detect unwanted emails and filter them effectively.

---

## Tech Stack / Tools Used
- **Programming Language:** Python  
- **Libraries / Frameworks:**  
  - `scikit-learn` (Naive Bayes, Linear SVM, TF-IDF vectorizer, metrics)  
  - `pandas` (data manipulation)  
  - `numpy` (numerical operations)  
  - `matplotlib` & `seaborn` (visualization)  
  - `joblib` (model saving & loading)  

---

## Features
- **Text Preprocessing:** TF-IDF vectorization converts email text into numerical features.  
- **Machine Learning Models:**  
  - **Multinomial Naive Bayes** – fast and efficient for text data.  
  - **Linear SVM (LinearSVC)** – highly accurate for spam detection.  
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1 Score, Confusion Matrix, and Classification Report.  
- **Model Persistence:** Trained models and vectorizer are saved for predictions on new emails.

---

## How it Works
1. **Data Loading:** Emails labeled as `ham` or `spam`.  
2. **Feature Extraction:** TF-IDF vectorizer transforms email text into numerical features.  
3. **Model Training:** Naive Bayes and Linear SVM models are trained.  
4. **Evaluation:** Metrics and confusion matrices assess performance.  
5. **Prediction:** Saved models classify new emails instantly.

---

## Example
```python
import joblib

# Load trained model and vectorizer
svm_model = joblib.load("linear_svm_spam_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Predict new emails
new_emails = ["Win a free iPhone now!", "Please review the report."]
X_new = vectorizer.transform(new_emails)
predictions = svm_model.predict(X_new)

for email, label in zip(new_emails, predictions):
    print(email, "->", "Spam" if label == 1 else "Ham")
````

---

## Results

* **Naive Bayes:** Accuracy \~ 94.9%, F1 Score \~ 0.91
* **Linear SVM:** Accuracy \~ 98.4%, Precision \~ 0.98, Recall \~ 0.97

**Linear SVM** performs better on this dataset and is recommended for deployment.

---

## Notes

* TF-IDF is fitted only on training data; test/new data is transformed using the fitted vectorizer.
* Models are ready for real-time predictions on new email content.

---

