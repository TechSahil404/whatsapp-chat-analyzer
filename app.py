import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import plotly.express as px
import emoji
import re
from datetime import datetime
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords

# Download the VADER lexicon and stopwords for sentiment analysis
nltk.download('vader_lexicon')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# -------------------- Helper Functions --------------------
def preprocess_chat(data):
    pattern = r"(\d{1,2}/\d{1,2}/\d{2}), (\d{1,2}:\d{2})\s(am|pm) - (.*?): (.*)"
    messages = []
    for line in data.split('\n'):
        match = re.match(pattern, line)
        if match:
            date_str = match.group(1) + ", " + match.group(2) + " " + match.group(3)
            try:
                date = datetime.strptime(date_str, "%d/%m/%y, %I:%M %p")
                user = match.group(4)
                message = match.group(5)
                messages.append([date, user, message])
            except ValueError:
                continue
    return pd.DataFrame(messages, columns=["date", "user", "message"])

def get_response_times(df, user1, user2):
    df = df[df["user"].isin([user1, user2])].sort_values("date")
    df = df.reset_index(drop=True)
    response_times = []
    for i in range(1, len(df)):
        if df.loc[i-1, "user"] == user1 and df.loc[i, "user"] == user2:
            delta = (df.loc[i, "date"] - df.loc[i-1, "date"]).total_seconds() / 60  # in minutes
            response_times.append(delta)
    return response_times

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="WhatsApp Chat Analyzer", layout="wide")
st.title("📱 WhatsApp Chat Analyzer")

uploaded_file = st.file_uploader("Upload your WhatsApp chat (.txt) file", type="txt")

if uploaded_file is not None:
    raw_data = uploaded_file.read().decode("utf-8")
    df = preprocess_chat(raw_data)

    if df.empty:
        st.error("❌ No valid messages parsed. Please check the file format.")
    else:
        # Sidebar filters
        st.sidebar.header("Filter Options")
        users = df["user"].unique().tolist()
        users.sort()
        selected_user = st.sidebar.selectbox("Select User (optional)", ["All"] + users)
        min_date = df["date"].min().date()
        max_date = df["date"].max().date()
        date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)

        # Filter DataFrame
        filtered_df = df.copy()
        filtered_df = filtered_df[(filtered_df["date"].dt.date >= date_range[0]) & (filtered_df["date"].dt.date <= date_range[1])]
        if selected_user != "All":
            filtered_df = filtered_df[filtered_df["user"] == selected_user]

        st.subheader(f"📊 Analysis from {date_range[0]} to {date_range[1]}")
        st.markdown(f"**Total Messages:** {filtered_df.shape[0]}")

        # -------------------- Charts --------------------
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🔝 Most Active Users")
            user_counts = df["user"].value_counts().head(15)
            fig = px.bar(user_counts, x=user_counts.index, y=user_counts.values,
                         labels={"x": "User", "y": "Message Count"}, title="Top 15 Active Users")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### ⏰ Messages per Hour")
            filtered_df["hour"] = filtered_df["date"].dt.hour
            hourly = filtered_df["hour"].value_counts().sort_index()
            fig = px.bar(x=hourly.index, y=hourly.values,
                         labels={"x": "Hour", "y": "Messages"}, title="Messages by Hour")
            st.plotly_chart(fig, use_container_width=True)

        # Day-wise Distribution
        st.markdown("### 📅 Messages by Day of Week")
        filtered_df["day"] = filtered_df["date"].dt.day_name()
        daywise = filtered_df["day"].value_counts()
        fig = px.pie(values=daywise.values, names=daywise.index, title="Day-wise Message Distribution")
        st.plotly_chart(fig, use_container_width=True)

        # Top Words
        st.markdown("### 🧠 Top 15 Words Used (Meaningful)")
        text = " ".join(filtered_df["message"])
        words = re.findall(r'\b\w+\b', text.lower())
        # Remove stopwords and short words
        meaningful_words = [w for w in words if w not in stop_words and len(w) > 2]
        common_words = Counter(meaningful_words).most_common(15)
        if common_words:
            words_df = pd.DataFrame(common_words, columns=["word", "count"])
            fig = px.bar(words_df, x="word", y="count", title="Top 15 Meaningful Words Used")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No meaningful words found.")

        # Word Cloud
        st.markdown("### ☁️ Word Cloud")
        if words:
            wc = WordCloud(width=800, height=400, background_color="white").generate(" ".join(words))
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)
        else:
            st.info("Not enough data for word cloud.")

            

        # Search messages
        search_term = st.text_input("🔍 Search messages")
        if search_term:
            filtered_df = filtered_df[filtered_df["message"].str.contains(search_term, case=False, na=False)]

            st.markdown(f"### 🔍 Results for: `{search_term}` ({filtered_df.shape[0]} matches)")

            # Message Length Distribution
            st.markdown("### 🔠 Message Length Distribution")
            filtered_df["msg_length"] = filtered_df["message"].apply(len)
            fig = px.histogram(filtered_df, x="msg_length", nbins=30, title="Message Length")
            st.plotly_chart(fig, use_container_width=True)

            # Emoji Usage
            st.markdown("### 😀 Emoji Usage (Top 15)")
            all_emojis = [c for msg in filtered_df["message"] for c in msg if c in emoji.EMOJI_DATA]
            emoji_freq = Counter(all_emojis).most_common(15)

            if emoji_freq:
                emo_df = pd.DataFrame(emoji_freq, columns=["emoji", "count"])
                fig = px.bar(emo_df, x="emoji", y="count", title="Top Emojis")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No emojis found in this chat.")

            # Sentiment Analysis
            st.markdown("### 😊 Sentiment Analysis")
            def get_sentiment(text):
                return TextBlob(text).sentiment.polarity

            if not filtered_df.empty:
                filtered_df["sentiment"] = filtered_df["message"].apply(get_sentiment)
                sentiment_labels = filtered_df["sentiment"].apply(
                    lambda x: "Positive" if x > 0.1 else ("Negative" if x < -0.1 else "Neutral")
                )
                sentiment_counts = sentiment_labels.value_counts()
                fig = px.pie(
                    names=sentiment_counts.index,
                    values=sentiment_counts.values,
                    title="Sentiment Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No messages to analyze sentiment.")

            # Export Button
            st.markdown("### 📤 Export Filtered Data")
            csv = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", data=csv, file_name="filtered_chat.csv", mime='text/csv')

        # Dark Mode Info
        st.sidebar.markdown("---")
        st.sidebar.markdown("🌓 For Dark Mode, enable it from Streamlit theme settings.")
        
git init

