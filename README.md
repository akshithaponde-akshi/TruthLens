# 📰 TruthLens

### See Beyond the Headlines

## 🚀 Live Demo

👉 [Try TruthLens Live](https://truthlens-fakenews.streamlit.app/)

TruthLens is a Machine Learning based Fake News Detection System that classifies a news article as **Real** or **Fake** using Natural Language Processing (NLP) and Machine Learning techniques.

> ⚠️ Educational Project: TruthLens makes predictions based on patterns learned from its training dataset. A prediction is not proof that a news article is factually true or false.

---

## 📌 Project Overview

Fake news can spread rapidly through online platforms and social media.

TruthLens attempts to identify whether a news article is likely to be Real or Fake by analyzing the textual patterns present in the article.

The project uses:

- Natural Language Processing (NLP)
- TF-IDF Vectorization
- Machine Learning Classification
- Streamlit for Web Application Development

Three Machine Learning algorithms are implemented:

1. Logistic Regression
2. Naive Bayes
3. Support Vector Machine (SVM)

---

## 🚀 Features

- 📰 Fake News Classification
- 🤖 Multiple Machine Learning Models
- 📊 Model Performance Comparison
- 📈 Confusion Matrix
- 🎯 Prediction Confidence
- 🗂️ Prediction History
- 📂 News Category Analysis
- 🌐 Interactive Streamlit Dashboard

---

## 🧠 Machine Learning Workflow

```text
News Article
     ↓
Title + Content
     ↓
Text Preprocessing
     ↓
TF-IDF Vectorization
     ↓
Machine Learning Model
     ↓
Real / Fake Prediction
     ↓
Streamlit Dashboard
```

---

## 🔬 Models Used

### 1. Logistic Regression

Logistic Regression is used as one of the classification algorithms to predict whether the given news article belongs to the Real or Fake class.

### 2. Naive Bayes

Multinomial Naive Bayes is used for text classification and is included in TruthLens for comparison with the other models.

### 3. Support Vector Machine

Linear SVM is used to classify the TF-IDF feature representation of the news article.

---

````markdown
## 📊 Dataset

The current TruthLens dataset contains:

- **1,000 news records**
- **516 Real**
- **484 Fake**
- **10 news categories**

The dataset contains the following fields:

```text
id
title
content
category
source
date
label
verification_source
````

The `verification_source` field is used as reference/audit information and is not used as a model input feature.

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Streamlit
- Joblib
- Matplotlib

---

## 📁 Project Structure

```text
TruthLens/
│
├── data/
│   └── truthlens_1000_final.csv
│
├── model/
│   ├── logistic_regression_model_1000.pkl
│   ├── naive_bayes_model_1000.pkl
│   ├── svm_model_1000.pkl
│   ├── tfidf_vectorizer_1000.pkl
│   ├── model_performance_1000.csv
│   ├── logistic_regression_confusion_matrix_1000.csv
│   ├── naive_bayes_confusion_matrix_1000.csv
│   └── svm_confusion_matrix_1000.csv
│
├── app.py
├── train.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/akshithaponde-akshi/TruthLens.git
```

### 2. Open the project folder

```bash
cd TruthLens
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

Windows CMD:

```bash
venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Start the Streamlit application using:

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 📊 Model Performance

The models were evaluated on a held-out test set of **200 records**.

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 100.00% | 100.00% | 100.00% | 100.00% |
| Naive Bayes | 98.50% | 100.00% | 97.09% | 98.52% |
| SVM | 99.50% | 99.04% | 100.00% | 99.52% |

> These results are based on the project's current dataset and held-out test split. They should not be interpreted as proof of real-world fact-checking accuracy.

---

## 📈 Confusion Matrix

TruthLens provides confusion matrices for all three models.

A confusion matrix helps analyze:

- Correctly classified Real news
- Correctly classified Fake news
- Real news classified as Fake
- Fake news classified as Real

---

## 💻 Application

The Streamlit application provides:

### Dashboard

Displays:

- Total number of news articles
- Real/Fake distribution
- News category distribution

### News Prediction

Users can:

1. Select a Machine Learning model
2. Enter a news article
3. Generate a prediction
4. View the model used
5. View the model's confidence score

### Prediction History

The application keeps track of predictions made during the current session.

---

## 🌐 Deployment

TruthLens is deployed using Streamlit Community Cloud.

👉 [Open the Live Application](https://truthlens-fakenews.streamlit.app/)

---

## ⚠️ Limitations

TruthLens has several limitations:

- The dataset is relatively small for a real-world fact-checking system.
- Machine Learning models learn patterns from the available training data.
- The system cannot independently verify whether a news claim is factually true.
- Predictions may be affected by dataset quality and distribution.
- Performance on the test set may not represent performance on unseen real-world news.
- The confidence value should not be interpreted as factual certainty.

---

## 🔮 Future Improvements

Possible future improvements include:

- Increasing the size and diversity of the dataset
- Using larger real-world news datasets
- Adding advanced NLP techniques
- Experimenting with deep learning models
- Using transformer-based models such as BERT
- Adding explainable AI techniques
- Adding external fact-checking sources
- Improving confidence calibration

---

## 🎯 Learning Outcomes

Through this project, the following concepts were practiced:

- Python
- Data preprocessing
- Exploratory Data Analysis
- Natural Language Processing
- TF-IDF
- Train/Test Split
- Classification Algorithms
- Model Evaluation
- Confusion Matrix
- Model Serialization
- Streamlit
- Machine Learning Deployment

---

## 👩‍💻 Author

**Akshitha**

B.Tech – Computer Science & Engineering (AI)

---

## 📌 Disclaimer

TruthLens is an educational Machine Learning project.

The predictions generated by this application should not be treated as authoritative fact-checking results. Always verify important information using reliable sources.
