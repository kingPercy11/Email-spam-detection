import streamlit as st
from streamlit.components.v1 import html
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer


ps = PorterStemmer()
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))


#function to transform text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f6;
        color: #262730;
        padding: 20px;
        border-radius: 10px;
    }
    .stTextArea textarea {
        background-color: #fff;
        border: 1px solid #ccc;
        border-radius: 5px;
        color: #262730;
    }
    .stTextArea textarea::placeholder {
        color: grey;
        opacity: 1; /* Make sure the placeholder is fully visible */
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 0.5em 1em;
        margin-top: 10px;
        border: none;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    .stHeader {
        color: #4CAF50;
        font-size: 24px;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: white;'>Email/SMS Spam Classifier</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: #4CAF50;'>Check if your message is Spam or Not </h2>", unsafe_allow_html=True)

input_sms = st.text_area("yo", placeholder="Enter the message here", height=150, label_visibility="collapsed")

if st.button('Predict'):
    transformed_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]
    if result == 1:
        st.markdown("<h3 style='color: red;'>🚫 This is Spam!</h3>", unsafe_allow_html=True)
    else:
        st.markdown("<h3 style='color: green;'>✅ This is Not Spam!</h3>", unsafe_allow_html=True)