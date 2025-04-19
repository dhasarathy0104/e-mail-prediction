import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the saved model
tfidf_vectorizer = pickle.load(open('tfidfvectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Function to predict email spam or ham
def emailspam_prediction(message):
    input_mail = [message]
    input_data_feature = tfidf_vectorizer.transform(input_mail)
    prediction = model.predict(input_data_feature)
    if prediction[0] == 1:
        return "Ham Mail"
    else:
        return "Spam Mail"

# Function to display welcome page
def welcome_page():
    st.title('Welcome to Email Spam or Ham Detection Web App!')
    st.markdown(
        """
        <style>
        .big-font {
            font-size: 24px !important;
        }
        .sub-font {
            font-size: 18px !important;
        }
        .highlight {
            background-color: #f0f0f0;
            padding: 10px;
            border-radius: 5px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.write('This web app helps you classify emails as spam or ham (not spam) using a machine learning model.')
    st.markdown('<hr class="big-font">', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 3])
    with col1:
        st.header('Project Overview')
        st.markdown(
            """
            - The machine learning model used in this app is trained on a dataset of emails labeled as spam or ham.
            - The model utilizes TF-IDF vectorization for feature extraction.
            """
        )

    with col2:
        st.header('How to Use')
        st.markdown(
            """
            1. Enter the email message you want to classify in the text area provided.
            2. Click the 'Predict Email' button.
            """
        )

    st.markdown('<hr class="big-font">', unsafe_allow_html=True)

    st.header('About the Model')
    st.markdown(
        """
        - The machine learning model used in this app is trained on a dataset of labeled emails.
        - It employs a TF-IDF vectorizer to transform text data into numerical features.
        """
    )

    st.markdown('<hr class="big-font">', unsafe_allow_html=True)

    st.header('Disclaimer')
    st.markdown(
        """
        - This web app is for demonstration purposes only and may not be suitable for real-world applications.
        - The predictions made by the model may not always be accurate.
        """
    )

    st.markdown('<hr class="big-font">', unsafe_allow_html=True)

    st.header('Get Started')
    st.write("To get started, enter your email message in the text area below and click 'Predict Email'.")


# Function to display prediction page
def prediction_page():
    st.title('Email Prediction')
    message = st.text_area('Enter your email message here')

    # Creating a button for prediction
    if st.button('Predict Email'):
        prediction = emailspam_prediction(message)
        if prediction == "Ham Mail":
            st.success("Prediction: " + prediction)
        else:
            st.error("Prediction: " + prediction)

# Main function to manage page navigation
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("", ["Home", "Prediction"])

    if page == "Home":
        welcome_page()
    elif page == "Prediction":
        prediction_page()

if __name__ == '__main__':
    main()
