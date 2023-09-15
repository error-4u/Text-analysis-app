from tracemalloc import stop
import streamlit as st
import numpy as np
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier

# Designing side bar
rad=st.sidebar.radio("Navigation",["Home","Spam or Ham Detection","Sentiment Analysis","Stress Detection","Hate and Offensive Content Detection","Sarcasm Detection"])

# Home page designing
if rad=="Home":
    st.title("Complete Text Analysis App")
    st.image("Complete Text Analysis Home Page.jpg")
    st.text(" ")
    st.text("The Following Text Analysis Options Are Available->")
    st.text(" ")
    st.text("1. Spam or Ham Detection")
    st.text("2. Sentiment Analysis")
    st.text("3. Stress Detection")
    st.text("4. Hate and Offensive Content Detection")
    st.text("5. Sarcasm Detection")
