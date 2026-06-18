import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@st.cache_data
def load_data(path):
    return pd.read_csv(path)


df = load_data("banking_knowledge_base_1000.csv")

questions = df['Question'].fillna("").astype(str)
answers = df['Answer'].fillna("").astype(str)


@st.cache_resource
def build_vectorizer(qs):
    vec = TfidfVectorizer(stop_words='english')
    qv = vec.fit_transform(qs)
    return vec, qv


vectorizer, question_vectors = build_vectorizer(questions)


def chatbot_response(user_input):
    user_vector = vectorizer.transform([user_input])
    similarity_scores = cosine_similarity(user_vector, question_vectors)
    most_similar_index = int(similarity_scores.argmax())
    score = float(similarity_scores[0, most_similar_index])
    return answers.iloc[most_similar_index], score


st.set_page_config(page_title="Banking FAQ Chat", layout="wide", page_icon="💬")

CSS = """
<style>
:root{--bg:#0f1724;--card:#0b1220;--accent:#06b6d4}
body {background: linear-gradient(180deg,#021124 0%, #061826 100%);}
.header{background:linear-gradient(90deg, rgba(6,182,212,0.12), rgba(99,102,241,0.06)); padding:18px; border-radius:12px}
.title{font-family: 'Segoe UI', Roboto, sans-serif; color: white;}
.subtitle{color: #cbd5e1}
.chat-bubble.user{background:linear-gradient(90deg,#7dd3fc55,#60a5fa55); padding:12px; border-radius:12px; color:#021124}
.chat-bubble.bot{background:#0b122026; border:1px solid rgba(255,255,255,0.03); padding:12px; border-radius:12px; color:#e6eef8}
.meta{color:#94a3b8; font-size:12px}
.card{background:rgba(255,255,255,0.02); padding:14px; border-radius:10px}
.examples button{margin:6px}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

if 'history' not in st.session_state:
    st.session_state.history = []


with st.container():
    col1, col2 = st.columns([3,1])
    with col1:
        st.markdown("<div class='header'>", unsafe_allow_html=True)
        st.markdown("<h1 class='title'>💬 Banking FAQ Chat</h1>", unsafe_allow_html=True)
        st.markdown("<div class='subtitle'>Ask common banking questions — smart, fast, and friendly.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div style='text-align:right'><small class='meta'>Creative UI by Streamlit</small></div>", unsafe_allow_html=True)

st.write("")

with st.container():
    left, right = st.columns([3,1])
    with left:
        query = st.text_area("Ask your question", height=100, key='query_input', placeholder='e.g. How do I open a savings account?')
        ask = st.button("Ask")
        clear = st.button("Clear chat")
        if clear:
            st.session_state.history = []

        if ask and query:
            answer, score = chatbot_response(query)
            st.session_state.history.append({'role':'user','text':query})
            st.session_state.history.append({'role':'bot','text':answer,'score':score})

        for item in st.session_state.history[::-1]:
            if item['role'] == 'user':
                st.markdown(f"<div class='chat-bubble user'>{item['text']}</div>", unsafe_allow_html=True)
            else:
                sc = item.get('score', None)
                meta = f"<div class='meta'>Confidence: {sc:.2f}</div>" if sc is not None else ""
                st.markdown(f"<div class='chat-bubble bot'>{item['text']}{meta}</div>", unsafe_allow_html=True)

   


if __name__ == "__main__":
    pass



            

