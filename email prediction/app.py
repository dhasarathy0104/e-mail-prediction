import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import streamlit as st
feature_extraction = TfidfVectorizer(min_df=1, stop_words="english", lowercase=True)

# Load the saved model
tfidf_vectorizer=pickle.load(open('tfidfvectorizer.pkl', 'rb'))

model = pickle.load(open('model.pkl', 'rb'))

def emailspam_prediction(message):
    input_mail=[message]
    input_data_feature=tfidf_vectorizer.transform(input_mail)

    prediction=model.predict(input_data_feature)

    if prediction[0] == 1:
        st.write("ham mail")
    else:
        st.write ("spam Mail")

def main():
    # Giving the title
    st.title('Email Spam Prediction Web App')

    # Code for prediction
    message = st.text_area('give yor mail here')

    # Creating a button for prediction
    if st.button('Predict Email'):
         emailspam_prediction(message)

    
if __name__ == '__main__':
    main()
