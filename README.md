# NLP-Based Sentiment Analysis of Product Reviews Using TF-IDF and Machine Learning

## 1. Project Overview

This project presents an NLP-based sentiment analysis system for classifying product reviews into **Positive** and **Negative** sentiments.

The project uses **Natural Language Processing (NLP)** techniques to preprocess review text and **TF-IDF (Term Frequency-Inverse Document Frequency)** to convert text into numerical features.

A **Logistic Regression** machine learning model is then trained to classify the reviews.

The project also includes a **Streamlit web application** that allows users to enter a product review and obtain a predicted sentiment.

---

## 2. Problem Statement

Online product reviews contain valuable information about customer opinions and experiences. However, manually analyzing a large number of reviews is time-consuming.

The objective of this project is to develop an NLP-based machine learning system that automatically analyzes product reviews and classifies them as either **Positive** or **Negative**.

The system uses text preprocessing, TF-IDF feature extraction, and Logistic Regression for sentiment classification.

---

## 3. Objectives

The main objectives of this project are:

- To collect and analyze product review data.
- To preprocess textual review data using NLP techniques.
- To create sentiment labels from product ratings.
- To convert text into numerical features using TF-IDF.
- To train a machine learning model for sentiment classification.
- To evaluate the model using accuracy, precision, recall, F1-score, and confusion matrix.
- To develop a simple Streamlit application for sentiment prediction.

---

## 4. Dataset

The dataset used in this project is the **Indian Products on Amazon** dataset.

The original dataset contains **2,782 product reviews**.

### Dataset Columns

| Column | Description |
|---|---|
| `asin` | Product identifier |
| `name` | Product name |
| `date` | Review date |
| `rating` | Product rating |
| `review` | Review text |

### Dataset File

The dataset used in the project is:

`amazon_vfl_reviews.csv`

---

## 5. Dataset Analysis

The original dataset contains **2,782 reviews**.

### Rating Distribution

| Rating | Number of Reviews |
|---:|---:|
| 1 | 546 |
| 2 | 130 |
| 3 | 198 |
| 4 | 464 |
| 5 | 1,444 |

The rating distribution shows that 5-star reviews are the most common in the dataset.

### Missing Values

There were **6 missing values** in the `review` column.

### Duplicate Reviews

The dataset contained **1,522 duplicate reviews**.

Data cleaning and sentiment labelling were performed before training the machine learning model.

---

## 6. Sentiment Labelling

The product ratings were used to create sentiment labels.

Reviews were grouped into two sentiment classes:

- **Positive**
- **Negative**

After the initial sentiment labelling and cleaning:

| Sentiment | Number of Reviews |
|---|---:|
| Positive | 767 |
| Negative | 289 |
| Total | 1,056 |

After removing empty reviews, the final dataset contained **1,051 reviews**.

### Final Sentiment Distribution

| Sentiment | Number of Reviews |
|---|---:|
| Positive | 763 |
| Negative | 288 |
| Total | 1,051 |

---

## 7. Text Preprocessing

Text preprocessing was performed to prepare the review text for machine learning.

The preprocessing steps include:

1. Converting text to lowercase.
2. Removing unnecessary characters and punctuation.
3. Removing unwanted spaces.
4. Removing stopwords where applicable.
5. Preparing the cleaned text for feature extraction.

Five empty reviews remained after preprocessing and were removed.

The final dataset contained **0 empty reviews**.

---

## 8. Train-Test Split

The final dataset was divided into training and testing sets.

The following settings were used:

- Training data: **840 reviews**
- Testing data: **211 reviews**
- Test size: **20%**
- Random state: **42**
- Stratification: Applied

The training and testing data were separated before applying TF-IDF to avoid data leakage.

---

## 9. TF-IDF Feature Extraction

TF-IDF was used to convert the cleaned review text into numerical feature vectors.

TF-IDF assigns importance to words based on their frequency in a document and their frequency across the complete collection of documents.

### TF-IDF Parameters

- `max_features = 5000`
- `ngram_range = (1, 2)`

The model uses both:

- Unigrams: single words
- Bigrams: pairs of words

### TF-IDF Output

Training data shape:

```text
(840, 5000)
```

Testing data shape:

```text
(211, 5000)
```

Therefore, the TF-IDF vectorizer generated up to **5,000 features**.

---

## 10. Machine Learning Model

### Logistic Regression

The machine learning algorithm used in this project is **Logistic Regression**.

Logistic Regression is a classification algorithm that is suitable for binary classification problems such as Positive vs Negative sentiment classification.

The model was implemented using:

```python
LogisticRegression(max_iter=1000)
```

The model was trained using the TF-IDF features extracted from the training reviews.

---

## 11. Project Workflow

The overall workflow of the project is:

```text
Product Review Dataset
        ↓
Data Cleaning
        ↓
Sentiment Labelling
        ↓
Text Preprocessing
        ↓
Train-Test Split
        ↓
TF-IDF Feature Extraction
        ↓
Logistic Regression
        ↓
Sentiment Prediction
        ↓
Model Evaluation
        ↓
Streamlit Application
```

---

## 12. Model Evaluation

The trained model was evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

### Accuracy

The Logistic Regression model achieved an overall accuracy of:

**85.31%**

This means that the model correctly classified approximately 85% of the reviews in the test dataset.

---

## 13. Classification Report

The classification report obtained from the model is:

```text
              precision    recall  f1-score   support

Negative       0.97      0.48      0.64        58
Positive       0.84      0.99      0.91       153

accuracy                           0.85       211
macro avg       0.90      0.74      0.78       211
weighted avg    0.87      0.85      0.83       211
```

### Interpretation

The model achieved a high recall for Positive reviews (**0.99**), meaning that most positive reviews in the test set were correctly identified.

The recall for Negative reviews was lower (**0.48**). This can be partly associated with the imbalance between positive and negative reviews in the final dataset.

Therefore, although the overall accuracy is **85.31%**, the model performs differently across the two sentiment classes.

---

## 14. Confusion Matrix

A confusion matrix was used to visualize the correct and incorrect predictions made by the model.

It helps identify:

- Correct Positive predictions
- Correct Negative predictions
- Positive reviews classified as Negative
- Negative reviews classified as Positive

The confusion matrix is included in the project screenshots.

---

## 15. Streamlit Application

A Streamlit-based web application was developed for the project.

The application allows users to enter a product review and receive a predicted sentiment.

### Application Process

```text
User enters product review
          ↓
Text preprocessing
          ↓
TF-IDF transformation
          ↓
Trained Logistic Regression model
          ↓
Predicted sentiment
          ↓
Positive / Negative result
```

The trained model and TF-IDF vectorizer are saved as:

```text
sentiment_model.pkl
tfidf_vectorizer.pkl
```

The Streamlit application is implemented in:

```text
app.py
```

---

## 16. Technologies Used

### Programming Language

- Python

### Libraries and Frameworks

- Pandas
- NumPy
- Scikit-learn
- NLTK
- Matplotlib
- Seaborn
- Streamlit
- Pickle

### NLP Techniques

- Text Cleaning
- Text Preprocessing
- Stopword Removal
- TF-IDF Feature Extraction
- Sentiment Classification

### Machine Learning Algorithm

- Logistic Regression

---

## 17. Project Structure

```text
NLP-Sentiment-Analysis/
│
├── app.py
├── sentiment_analysis.ipynb
├── sentiment_model.pkl
├── tfidf_vectorizer.pkl
├── README.md
├── requirements.txt
├── amazon_vfl_reviews.csv
│
├── dataset/
│
├── screenshots/
│   ├── rating_distribution.png
│   ├── sentiment_distribution.png
│   ├── preprocessing.png
│   ├── tfidf.png
│   ├── classification_report.png
│   ├── confusion_matrix.png
│   ├── application.png
│   ├── positive_prediction.png
│   └── negative_prediction.png
│
└── report/
    └── NLP.docx
```

---

## 18. Installation

Clone the repository:

```bash
git clone https://github.com/Hima-Joy/NLP-Sentiment-Analysis.git
```

Open the project folder:

```bash
cd NLP-Sentiment-Analysis
```

Install the required Python libraries:

```bash
pip install -r requirements.txt
```

If the project files are inside the nested `NLP-Sentiment-Analysis` folder, enter that folder before running the application:

```bash
cd NLP-Sentiment-Analysis
```

---

## 19. Running the Streamlit Application

Run the following command:

```bash
streamlit run app.py
```

After running the command, Streamlit will provide a local URL in the terminal.

Open the provided URL in a web browser to use the application.

---

## 20. Running the Notebook

The complete NLP and machine learning implementation is available in:

```text
sentiment_analysis.ipynb
```

The notebook contains:

- Dataset loading
- Dataset analysis
- Data cleaning
- Sentiment labelling
- Text preprocessing
- TF-IDF feature extraction
- Model training
- Model prediction
- Model evaluation
- Confusion matrix

The notebook can be executed using **Google Colab** or **Jupyter Notebook**.

---

## 21. Results Summary

| Parameter | Result |
|---|---:|
| Original Reviews | 2,782 |
| Final Reviews | 1,051 |
| Positive Reviews | 763 |
| Negative Reviews | 288 |
| Training Reviews | 840 |
| Testing Reviews | 211 |
| TF-IDF Features | 5,000 |
| N-gram Range | 1–2 |
| Machine Learning Model | Logistic Regression |
| Accuracy | 85.31% |

---

## 22. Advantages

- Simple and easy-to-understand NLP workflow.
- Uses an explainable machine learning algorithm.
- TF-IDF is effective for converting text into numerical features.
- Provides multiple evaluation metrics.
- Includes a confusion matrix for performance analysis.
- Provides a simple Streamlit user interface.
- Can be extended to larger review datasets.

---

## 23. Limitations

- The dataset contains more positive reviews than negative reviews.
- The model is limited to Positive and Negative sentiment classes.
- TF-IDF does not fully understand the semantic meaning of sentences.
- Sarcasm and context-dependent expressions may be difficult to classify.
- The model may not perform equally well on reviews from completely different domains.

---

## 24. Future Scope

The project can be improved in the future by:

- Using a larger and more balanced dataset.
- Adding a Neutral sentiment class.
- Applying advanced NLP models such as BERT.
- Using word embeddings such as Word2Vec or GloVe.
- Performing hyperparameter tuning.
- Applying techniques to handle class imbalance.
- Improving the Streamlit interface.
- Deploying the application online.
- Supporting multilingual product reviews.

---

## 25. Conclusion

This project demonstrates the use of Natural Language Processing and Machine Learning for sentiment analysis of product reviews.

The review text was cleaned and preprocessed before being converted into numerical features using TF-IDF. A Logistic Regression classifier was then trained to classify reviews as Positive or Negative.

The model achieved an overall accuracy of **85.31%** on the test dataset.

The project also includes a Streamlit application that provides a simple interface for entering reviews and obtaining sentiment predictions.

Overall, the project demonstrates a complete NLP pipeline from raw product reviews to machine learning-based sentiment prediction.

---

## 26. Author

**Hima Joy**

GitHub Repository:

https://github.com/Hima-Joy/NLP-Sentiment-Analysis
