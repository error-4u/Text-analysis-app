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

# cleaning and transforming the user input data
def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text=y[:]
    y.clear()
    ps=PorterStemmer()
    for i in text:
        y.append(ps.stem(i))
        return " ".join(y)

# this is for span detection 

tfidf1=TfidfVectorizer(stop_words=sw,max_features=20)
def transform1(txt1):
    txt2=tfidf1.fit_transform(txt1)
    return txt2.toarray()

df1=pd.read_csv("Spam Detection.csv")
df1.columns=["Label","Text"]
x=transform1(df1["Text"])
y=df1["Label"]
x_train1,x_test1,y_train1,y_test1=train_test_split(x,y,test_size=0.1,random_state=0)
model1=LogisticRegression()
model1.fit(x_train1,y_train1)

tfidf1=TfidfVectorizer(stop_words=sw,max_features=20)
def transform1(txt1):
    txt2=tfidf1.fit_transform(txt1)
    return txt2.toarray()

df1=pd.read_csv("Spam Detection.csv")
df1.columns=["Label","Text"]
x=transform1(df1["Text"])
y=df1["Label"]
x_train1,x_test1,y_train1,y_test1=train_test_split(x,y,test_size=0.1,random_state=0)
model1=LogisticRegression()
model1.fit(x_train1,y_train1)

#Spam Detection Analysis 
if rad=="Spam or Ham Detection":
    st.header("Detect Whether A Text Is Spam Or Ham??")
    sent1=st.text_area("Enter The Text")
    transformed_sent1=transform_text(sent1)
    vector_sent1=tfidf1.transform([transformed_sent1])
    prediction1=model1.predict(vector_sent1)[0]

    if st.button("Predict"):
        if prediction1=="spam":
            st.warning("Spam Text!!")
        elif prediction1=="ham":
            st.success("Ham Text!!")
#Sentiment Analysis Prediction 
tfidf2=TfidfVectorizer(stop_words=sw,max_features=20)
def transform2(txt1):
    txt2=tfidf2.fit_transform(txt1)
    return txt2.toarray()

df2=pd.read_csv("Sentiment Analysis.csv")
df2.columns=["Text","Label"]
x=transform2(df2["Text"])
y=df2["Label"]
x_train2,x_test2,y_train2,y_test2=train_test_split(x,y,test_size=0.1,random_state=0)
model2=LogisticRegression()
model2.fit(x_train2,y_train2)

#Sentiment Analysis Page
if rad=="Sentiment Analysis":
    st.header("Detect The Sentiment Of The Text!!")
    sent2=st.text_area("Enter The Text")
    transformed_sent2=transform_text(sent2)
    vector_sent2=tfidf2.transform([transformed_sent2])
    prediction2=model2.predict(vector_sent2)[0]

    if st.button("Predict"):
        if prediction2==0:
            st.warning("Negetive Text!!")
        elif prediction2==1:
            st.success("Positive Text!!")
