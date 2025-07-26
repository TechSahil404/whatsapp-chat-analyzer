import streamlit as st
import pandas as pd
import emoji
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import plotly.express as px
import re
from datetime import datetime
from textblob import TextBlob
from transformers import pipeline
import gensim
from gensim import corpora
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# --------- Chat preprocessing function (same as you have) -----------
def preprocess_chat(data):
    pattern = r"(\d{1,2}/\d{1,2}/\d{2}), (\d{1,2}:\d{2})\s(am|pm) - (.*?): (.*)"
    messages = []
    for line in data.split('\n'):
        match = re.match(pattern, line)
        if match:
            date_str = match.group(1) + ", " + match.group(2) + " " + match.group(3)
            try:
                date = datetime.strptime(date_str, "%d/%m/%y, %I:%M %p")
            except:
                continue
            user = match.group(4)
            message = match.group(5)
            messages.append([date, user, message])
    df = pd.DataFrame(messages, columns=["date", "user", "message"])
    return df

# --------- Sentiment Analysis Functions ---------
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

def sentiment_category(polarity):
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"

# --------- Emotion Detection Pipeline ---------
@st.cache_resource  # cache model loading
def load_emotion_model():
    return pipeline('text-classification', model='j-hartmann/emotion-english-distilroberta-base', return_all_scores=True)

emotion_classifier = load_emotion_model()

def get_emotion(text):
    results = emotion_classifier(text[:512])
    top_emotion = max(results[0], key=lambda x: x['score'])['label']
    return top_emotion

# --------- Topic Modeling Helpers ---------
def preprocess_texts(texts):
    processed_texts = []
    for doc in texts:
        tokens = gensim.utils.simple_preprocess(doc, deacc=True)
        tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
        processed_texts.append(tokens)
    return processed_texts

def build_lda_model(texts, num_topics=5):
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    lda = gensim.models.LdaModel(corpus=corpus,
                                 id2word=dictionary,
                                 num_topics=num_topics,
                                 random_state=42,
                                 passes=10,
                                 alpha='auto',
                                 per_word_topics=True)
    return lda, dictionary

def plot_topic_wordcloud(lda_model, topic_num):
    plt.figure(figsize=(8,6))
    plt.imshow(WordCloud(background_color='white').fit_words(dict(lda_model.show_topic(topic_num, 30))))
    plt.axis("off")
    plt.title(f"WordCloud for Topic #{topic_num+1}")
    st.pyplot(plt)

def generate_wordcloud(text, title="WordCloud"):
    wc = WordCloud(stopwords=stop_words, background_color='white', width=800, height=400).generate(text)
    plt.figure(figsize=(10,5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    st.pyplot(plt)

# --------- Main Streamlit App ---------

st.set_page_config(page_title="WhatsApp Chat Analyzer", layout="wide")
st.title("📱 WhatsApp Chat Analyzer")

uploaded_file = st.file_uploader("Upload your WhatsApp chat text file", type="txt")

if uploaded_file is not None:
    raw_data = uploaded_file.read().decode("utf-8")
    df = preprocess_chat(raw_data)

    # Sidebar filters
    st.sidebar.header("Filter Options")
    users = df["user"].unique().tolist()
    users.sort()
    selected_user = st.sidebar.selectbox("Select User (optional)", ["All"] + users)
    min_date = df['date'].min().date()
    max_date = df['date'].max().date()
    date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)

    # Filter DataFrame
    filtered_df = df.copy()
    filtered_df = filtered_df[(filtered_df['date'].dt.date >= date_range[0]) & (filtered_df['date'].dt.date <= date_range[1])]
    if selected_user != "All":
        filtered_df = filtered_df[filtered_df['user'] == selected_user]

    # Search box filter
    search_term = st.text_input("🔍 Search messages")
    if search_term:
        filtered_df = filtered_df[filtered_df["message"].str.contains(search_term, case=False, na=False)]

    st.subheader(f"📊 Analysis from {date_range[0]} to {date_range[1]}")
    st.markdown(f"**Total Messages:** {filtered_df.shape[0]}")

    # --- Basic Stats and Plots (your existing code can go here) ---

    # ---- SENTIMENT ANALYSIS ----
    st.subheader("😊 Sentiment Analysis")
    filtered_df['polarity'] = filtered_df['message'].apply(get_sentiment)
    filtered_df['sentiment'] = filtered_df['polarity'].apply(sentiment_category)

    sentiment_counts = filtered_df['sentiment'].value_counts()
    fig1 = px.pie(names=sentiment_counts.index, values=sentiment_counts.values, title='Overall Sentiment Distribution')
    st.plotly_chart(fig1, use_container_width=True)

    # Daily sentiment trend
    filtered_df['date_only'] = filtered_df['date'].dt.date
    daily_sentiment = filtered_df.groupby('date_only')['polarity'].mean().reset_index()
    fig2 = px.line(daily_sentiment, x='date_only', y='polarity', title='Daily Average Sentiment')
    st.plotly_chart(fig2, use_container_width=True)

    # ---- EMOTION DETECTION ----
    st.subheader("😃 Emotion Detection")
    with st.spinner("Running emotion detection (may take some time)..."):
        filtered_df['emotion'] = filtered_df['message'].apply(get_emotion)
    emotion_counts = filtered_df['emotion'].value_counts()
    fig3 = px.bar(x=emotion_counts.index, y=emotion_counts.values, title='Overall Emotion Distribution')
    st.plotly_chart(fig3, use_container_width=True)

    # Emotion by day stacked bar
    emotion_daily = filtered_df.groupby(['date_only', 'emotion']).size().unstack(fill_value=0)
    fig4 = px.bar(emotion_daily, x=emotion_daily.index, y=emotion_daily.columns.tolist(),
                  title='Daily Emotion Distribution', barmode='stack')
    st.plotly_chart(fig4, use_container_width=True)

    # ---- TOPIC MODELING & WORDCLOUDS ----
    st.subheader("🧠 Topic Modeling & Wordclouds")

    # Prepare texts for LDA
    texts = preprocess_texts(filtered_df['message'].tolist())
    if len(texts) > 0:
        lda_model, dictionary = build_lda_model(texts, num_topics=5)

        st.markdown("### Top 5 Topics")
        for i in range(5):
            st.markdown(f"**Topic #{i+1}:** {lda_model.print_topic(i, topn=8)}")
            plot_topic_wordcloud(lda_model, i)

    # Wordcloud for selected user or all messages
    wc_text = " ".join(filtered_df['message'].tolist())
    if selected_user != "All":
        wc_text = " ".join(filtered_df['message'].tolist())
    generate_wordcloud(wc_text, title=f"WordCloud {'for ' + selected_user if selected_user != 'All' else 'for All Users'}")
