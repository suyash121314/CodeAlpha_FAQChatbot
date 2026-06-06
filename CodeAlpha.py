import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_csv("faq.csv")

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['question'])

st.title("🤖 Smart FAQ Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.write(msg)

user_input = st.text_input("You:")

if user_input:
    user_vec = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vec, X)
    score = similarity.max()
    index = similarity.argmax()

    if score > 0.3:
        answer = data['answer'][index]
        response = f"🤖: {answer} \n\n(confidence: {score:.2f})"
    else:
        response = "🤖: Sorry, I don't understand that question."

    st.session_state.messages.append(f"You: {user_input}")
    st.session_state.messages.append(response)

    st.experimental_rerun()
