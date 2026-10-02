import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="TruthLens",
    page_icon="📰",
    layout="centered"
)


# ==========================================
# LOAD MODELS
# ==========================================

models = {
    "Logistic Regression": joblib.load(
        "model/logistic_regression_model_1000.pkl"
    ),

    "Naive Bayes": joblib.load(
        "model/naive_bayes_model_1000.pkl"
    ),

    "SVM": joblib.load(
        "model/svm_model_1000.pkl"
    )
}


# ==========================================
# LOAD TF-IDF VECTORIZER
# ==========================================

vectorizer = joblib.load(
    "model/tfidf_vectorizer_1000.pkl"
)


# ==========================================
# LOAD MODEL PERFORMANCE
# ==========================================

performance = pd.read_csv(
    "model/model_performance_1000.csv"
)


# ==========================================
# LOAD DATASET
# ==========================================

news_data = pd.read_csv(
    "data/truthlens_1000_final.csv"
)


# ==========================================
# DATASET STATISTICS
# ==========================================

total_news = len(news_data)

real_count = (
    news_data["label"] == "Real"
).sum()

fake_count = (
    news_data["label"] == "Fake"
).sum()


# ==========================================
# PREDICTION HISTORY
# ==========================================

if "history" not in st.session_state:
    st.session_state.history = []


# ==========================================
# TITLE
# ==========================================

st.title("📰 TruthLens")

st.subheader("See Beyond the Headlines")

st.write(
    "TruthLens uses Machine Learning to classify "
    "a news article as Real or Fake."
)

st.info(
    "⚠️ Educational project: This prediction is based "
    "on patterns learned from the training dataset. "
    "It is not proof that a news article is factually "
    "true or false."
)

st.divider()


# ==========================================
# DASHBOARD
# ==========================================

st.subheader("📊 Dashboard")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total News",
        total_news
    )

with col2:
    st.metric(
        "Real",
        real_count
    )

with col3:
    st.metric(
        "Fake",
        fake_count
    )


# ==========================================
# REAL VS FAKE CHART
# ==========================================

chart_data = pd.DataFrame(
    {
        "Prediction": ["REAL", "FAKE"],
        "Count": [real_count, fake_count]
    }
)

st.bar_chart(
    chart_data.set_index("Prediction")
)


# ==========================================
# NEWS BY CATEGORY
# ==========================================

st.subheader("📂 News by Category")

category_data = (
    news_data["category"]
    .value_counts()
)

st.bar_chart(
    category_data
)

st.divider()


# ==========================================
# CHECK NEWS
# ==========================================

st.subheader("🔍 Check News")


selected_model = st.selectbox(
    "🤖 Choose Machine Learning Model",
    [
        "Logistic Regression",
        "Naive Bayes",
        "SVM"
    ]
)


news = st.text_area(
    "Enter News Article",
    height=200,
    placeholder="Paste or type a news article here..."
)


if st.button("🔍 Check News"):

    if news.strip() == "":

        st.warning(
            "Please enter a news article."
        )

    else:

        # Convert article into TF-IDF
        news_tfidf = vectorizer.transform(
            [news]
        )


        # Select model
        model = models[selected_model]


        # Make prediction
        prediction = model.predict(
            news_tfidf
        )[0]


        # ==================================
        # CONFIDENCE
        # ==================================

        if hasattr(model, "predict_proba"):

            probability = model.predict_proba(
                news_tfidf
            )[0]

            confidence = (
                np.max(probability) * 100
            )

        else:

            decision_score = model.decision_function(
                news_tfidf
            )[0]

            confidence = (
                1 /
                (
                    1 +
                    np.exp(-abs(decision_score))
                )
            ) * 100


        # ==================================
        # DISPLAY RESULT
        # ==================================

        if prediction == "Real":

            result = "REAL"

            st.success(
                "🟢 LIKELY REAL NEWS"
            )

            st.write(
                "🧠 The model found text patterns "
                "that are more similar to Real news "
                "examples in its training data."
            )

        else:

            result = "FAKE"

            st.error(
                "🔴 LIKELY FAKE NEWS"
            )

            st.write(
                "🧠 The model found text patterns "
                "that are more similar to Fake news "
                "examples in its training data."
            )


        # ==================================
        # MODEL USED
        # ==================================

        st.info(
            f"🤖 Model Used: **{selected_model}**"
        )


        # ==================================
        # CONFIDENCE DISPLAY
        # ==================================

        st.metric(
            "📊 Model Confidence",
            f"{confidence:.2f}%"
        )


        st.caption(
            "Confidence indicates how strongly the "
            "model favors its prediction; it does not "
            "represent factual certainty."
        )


        # ==================================
        # SAVE HISTORY
        # ==================================

        st.session_state.history.append(
            {
                "News": news,
                "Model": selected_model,
                "Prediction": result,
                "Confidence": f"{confidence:.2f}%"
            }
        )


st.divider()


# ==========================================
# MODEL PERFORMANCE
# ==========================================

st.subheader("🤖 Model Performance")

st.write(
    "Comparison of the machine learning algorithms "
    "used in TruthLens."
)


# Accuracy chart

accuracy_data = performance[
    ["Model", "Accuracy"]
].copy()

accuracy_data["Accuracy"] = (
    accuracy_data["Accuracy"] * 100
)

accuracy_data = accuracy_data.set_index(
    "Model"
)

st.bar_chart(
    accuracy_data
)


# ==========================================
# PERFORMANCE TABLE
# ==========================================

performance_display = performance.copy()


for column in [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score"
]:

    performance_display[column] = (
        performance_display[column] * 100
    ).round(2).astype(str) + "%"


st.dataframe(
    performance_display,
    use_container_width=True,
    hide_index=True
)


st.divider()


# ==========================================
# CONFUSION MATRIX
# ==========================================

st.subheader("📊 Confusion Matrix")

st.write(
    "The confusion matrix shows how many Real and Fake "
    "news articles were classified correctly or incorrectly."
)


confusion_model = st.selectbox(
    "🤖 Select Model for Confusion Matrix",
    [
        "Logistic Regression",
        "Naive Bayes",
        "SVM"
    ],
    key="confusion_model"
)


# ==========================================
# CONFUSION MATRIX FILES
# ==========================================

confusion_files = {

    "Logistic Regression":
        "model/logistic_regression_confusion_matrix_1000.csv",

    "Naive Bayes":
        "model/naive_bayes_confusion_matrix_1000.csv",

    "SVM":
        "model/svm_confusion_matrix_1000.csv"
}


# Select corresponding file

confusion_file = confusion_files[
    confusion_model
]


# Read confusion matrix

confusion_matrix_data = pd.read_csv(
    confusion_file,
    index_col=0
)


# Display confusion matrix

st.dataframe(
    confusion_matrix_data,
    use_container_width=True
)


st.divider()


# ==========================================
# PREDICTION HISTORY
# ==========================================

st.subheader("🗂️ Prediction History")


if len(st.session_state.history) == 0:

    st.info(
        "No predictions yet."
    )

else:

    history_data = pd.DataFrame(
        st.session_state.history
    )


    history_data = history_data[
        [
            "News",
            "Model",
            "Prediction",
            "Confidence"
        ]
    ]


    st.dataframe(
        history_data,
        use_container_width=True,
        hide_index=True
    )


    if st.button(
        "🗑️ Clear Prediction History"
    ):

        st.session_state.history = []

        st.rerun()


# ==========================================
# FOOTER
# ==========================================

st.caption(
    "TruthLens | Machine Learning Fake News Detection Project"
)