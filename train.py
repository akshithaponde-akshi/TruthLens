import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("data/truthlens_1000_final.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# ==========================================
# 2. PREPARE INPUT AND OUTPUT
# ==========================================

X = df["title"] + " " + df["content"]
y = df["label"]


# ==========================================
# 3. TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# ==========================================
# 4. TF-IDF VECTORIZATION
# ==========================================

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("\nTF-IDF conversion completed!")
print("Training TF-IDF shape:", X_train_tfidf.shape)
print("Testing TF-IDF shape:", X_test_tfidf.shape)


# ==========================================
# 5. CREATE MODELS
# ==========================================

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "SVM": LinearSVC()
}


# ==========================================
# 6. TRAIN AND EVALUATE MODELS
# ==========================================

results = []

for name, model in models.items():

    print("\n===================================")
    print("Training:", name)
    print("===================================")

    # Train model
    model.fit(
        X_train_tfidf,
        y_train
    )

    # Make predictions
    y_pred = model.predict(
        X_test_tfidf
    )

    # Calculate metrics
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        pos_label="Real"
    )

    recall = recall_score(
        y_test,
        y_pred,
        pos_label="Real"
    )

    f1 = f1_score(
        y_test,
        y_pred,
        pos_label="Real"
    )

    # Confusion matrix
    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=["Real", "Fake"]
    )

    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)

    print("\nConfusion Matrix:")
    print(cm)

    # ======================================
    # SAVE CONFUSION MATRIX
    # ======================================

    cm_df = pd.DataFrame(
        cm,
        index=["Real", "Fake"],
        columns=["Real", "Fake"]
    )

    if name == "Logistic Regression":

        cm_df.to_csv(
            "model/logistic_regression_confusion_matrix_1000.csv"
        )

    elif name == "Naive Bayes":

        cm_df.to_csv(
            "model/naive_bayes_confusion_matrix_1000.csv"
        )

    elif name == "SVM":

        cm_df.to_csv(
            "model/svm_confusion_matrix_1000.csv"
        )

    # ======================================
    # STORE PERFORMANCE RESULTS
    # ======================================

    results.append(
        {
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        }
    )

    # ======================================
    # SAVE TRAINED MODEL
    # ======================================

    if name == "Logistic Regression":

        joblib.dump(
            model,
            "model/logistic_regression_model_1000.pkl"
        )

    elif name == "Naive Bayes":

        joblib.dump(
            model,
            "model/naive_bayes_model_1000.pkl"
        )

    elif name == "SVM":

        joblib.dump(
            model,
            "model/svm_model_1000.pkl"
        )


# ==========================================
# 7. SAVE TF-IDF VECTORIZER
# ==========================================

joblib.dump(
    vectorizer,
    "model/tfidf_vectorizer_1000.pkl"
)


# ==========================================
# 8. SAVE MODEL PERFORMANCE
# ==========================================

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    "model/model_performance_1000.csv",
    index=False
)


# ==========================================
# 9. DISPLAY FINAL RESULTS
# ==========================================

print("\n\n===================================")
print("FINAL MODEL PERFORMANCE")
print("===================================")

print(results_df)


# ==========================================
# 10. FINAL MESSAGE
# ==========================================

print("\n===================================")
print("1000-RECORD TRAINING COMPLETED!")
print("===================================")

print("\nSaved files:")

print(
    "1. model/logistic_regression_model_1000.pkl"
)

print(
    "2. model/naive_bayes_model_1000.pkl"
)

print(
    "3. model/svm_model_1000.pkl"
)

print(
    "4. model/tfidf_vectorizer_1000.pkl"
)

print(
    "5. model/model_performance_1000.csv"
)

print(
    "6. model/logistic_regression_confusion_matrix_1000.csv"
)

print(
    "7. model/naive_bayes_confusion_matrix_1000.csv"
)

print(
    "8. model/svm_confusion_matrix_1000.csv"
)