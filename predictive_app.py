# pip install streamlit transformers nltk
# pip install tensorflow
# pip install tf-keras
# pip install torch torchvision torchaudio
# import tensorflow as tf
# import torch

import streamlit as st
import pandas as pd
from transformers import pipeline
import string
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

def clean_text(text):
    text = ''.join([char for char in text if char not in string.punctuation])
    text = text.lower()
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

def map_sentiment_to_risk(sentiment_score):
    if sentiment_score == "5 stars":
        return "Low"
    elif sentiment_score == "4 stars":
        return "Low-Medium"
    elif sentiment_score == "3 stars":
        return "Medium"
    elif sentiment_score == "2 stars":
        return "Medium-High"
    elif sentiment_score == "1 star":
        return "High"
    else:
        return "Unknown"

sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

st.title("Predictive Mental Health Risk Level")

dataset = st.file_uploader(r"C:\Users\Namitha\Desktop\Medical_Prediction\archive\mental_health_data.csv", type=["csv"])

if dataset is not None:
    try:
        df = pd.read_csv(dataset)

        st.write("Dataset preview:")
        st.write(df.head())

        user_input = st.text_area("What's on your mind today? 💬")

        if st.button("Analyze"):
            if user_input:
                # Clean the input text
                cleaned_text = clean_text(user_input)

                # Perform sentiment analysis using BERT
                sentiment_result = sentiment_pipeline(cleaned_text)[0]['label']

                # Map sentiment to risk level
                risk_level = map_sentiment_to_risk(sentiment_result)

                # Display results
                st.subheader("Sentiment Score:")
                st.write(f"Sentiment: {sentiment_result}")
                st.subheader("Predicted Mental Health Risk Level:")
                st.write(f"Risk Level: {risk_level}")
            else:
                st.warning("Please enter some text to analyze.")
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
else:
    st.warning("Please upload the dataset to proceed.")
