# Email Spam Detection Web App

A Machine Learning-powered web application that classifies email messages as **Spam** or **Ham** (Legitimate). This project utilizes Natural Language Processing (NLP) techniques and a Logistic Regression model to provide real-time predictions through a user-friendly Streamlit interface.

## 🚀 Features
* **Real-time Prediction:** Enter any email text to instantly check if it's spam.
* **NLP Pipeline:** Uses TF-IDF (Term Frequency-Inverse Document Frequency) vectorization for text processing.
* **Interactive UI:** Built with Streamlit, featuring a multi-page navigation system (Home and Prediction).
* **High Accuracy:** The underlying model achieves a high accuracy rate (approx. 96%) on the training dataset.

## 🛠️ Tech Stack
* **Language:** Python
* **Machine Learning:** Scikit-learn
* **Web Framework:** Streamlit
* **Data Handling:** Pandas, NumPy
* **Natural Language Processing:** NLTK

## 📁 Project Structure
```text
├── app.py                   # Simple Streamlit entry point
├── web2.py                  # Enhanced Streamlit app with multi-page navigation
├── email_spam_detection.py  # Script for data analysis, training, and evaluation
├── model.pkl                # Trained Logistic Regression model
├── tfidfvectorizer.pkl      # Pre-trained TF-IDF Vectorizer
├── spam.csv                 # Dataset used for training
└── README.md                # Project documentation
