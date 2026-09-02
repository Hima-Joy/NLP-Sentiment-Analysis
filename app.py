import streamlit as st
import joblib
import re
import nltk

from nltk.corpus import stopwords

# Download NLTK stopwords
nltk.download('stopwords')

# Load trained model and TF-IDF vectorizer
model = joblib.load("sentiment_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# Get English stopwords
stop_words = set(stopwords.words("english"))

# Keep important negative words
stop_words = stop_words - {"not", "no", "nor", "never"}


# Text preprocessing function
def preprocess_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove punctuation and special characters
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Remove stopwords
    words = text.split()
    words = [word for word in words if word not in stop_words]

    return " ".join(words)


# Application title
st.title("🛍️ Product Review Sentiment Analysis")

st.write(
    "Enter a product review below to predict whether the sentiment "
    "is Positive or Negative."
)

# User input
review = st.text_area(
    "Enter your product review:",
    placeholder="Example: This product is excellent and I am very happy with it."
)

# Analyze button
if st.button("Analyze Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a product review.")

    else:

        # Preprocess review
        cleaned_review = preprocess_text(review)

        # Convert review into TF-IDF features
        review_tfidf = tfidf.transform([cleaned_review])

        # Predict sentiment
        prediction = model.predict(review_tfidf)[0]

        # Display result
        st.subheader("Prediction")

        if prediction == "Positive":
            st.success("😊 Sentiment: POSITIVE")

        else:
            st.error("😞 Sentiment: NEGATIVE")