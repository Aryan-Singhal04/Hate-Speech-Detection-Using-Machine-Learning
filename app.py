import streamlit as st
import re
import pickle

st.title("Hate Speech Detection Model")
data=st.text_input("write the text")
bt=st.button("Detect")


classifier=pickle.load(open("classifier.pkl",'rb'))
cv=pickle.load(open("CountVect.pkl","rb"))

import re
import nltk
from nltk.util import pr
stop=nltk.SnowballStemmer("english")
from nltk.corpus import stopwords
import string
stopword=set(stopwords.words("english"))


def clean(text):
    text=str(text).lower()
    text=re.sub(r'\[.*?\]','',text)
    text=re.sub(r'https?://\S+|www\.\S+','',text)
    text=re.sub(r'<.*?>+','',text)
    text=re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text=re.sub(r'\n','',text)
    text=re.sub(r'\w*\d\w*','',text)
    text=[stop.stem(word) for word in text.split() if word not in stopword]
    text=" ".join(text)
    return text


dt=clean(data)
dtt=cv.transform([dt]).toarray()
pred=classifier.predict(dtt)[0]
if(bt):
    st.write(pred)