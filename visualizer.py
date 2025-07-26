import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import pandas as pd

def plot_messages_per_day(df):
    daily_counts = df.groupby('date').size().reset_index(name='counts')
    fig = px.line(daily_counts, x='date', y='counts', title='Messages Per Day')
    return fig

def plot_messages_per_hour(df):
    hourly_counts = df.groupby('hour').size().reset_index(name='counts')
    fig = px.bar(hourly_counts, x='hour', y='counts', title='Messages Per Hour')
    return fig

def plot_messages_by_user(df):
    user_counts = df['sender'].value_counts().reset_index()
    user_counts.columns = ['sender', 'counts']
    fig = px.bar(user_counts, x='sender', y='counts', title='Messages by User')
    return fig

def plot_avg_words_per_message(df):
    df['word_count'] = df['message'].apply(lambda x: len(str(x).split()))
    avg_words = df.groupby('sender')['word_count'].mean().reset_index()
    fig = px.bar(avg_words, x='sender', y='word_count', title='Average Words Per Message')
    return fig

def plot_top_words(df, top_n=15):
    import re
    from collections import Counter
    text = " ".join(df['message'].dropna().astype(str).tolist()).lower()
    words = re.findall(r'\b\w+\b', text)
    stopwords = set(['the', 'to', 'and', 'of', 'a', 'i', 'is', 'in', 'for', 'on', 'you', 'it', 'that', 'this'])
    filtered_words = [w for w in words if w not in stopwords]
    counts = Counter(filtered_words).most_common(top_n)
    words, counts = zip(*counts)
    fig = px.bar(x=words, y=counts, title=f'Top {top_n} Words')
    return fig

def plot_top_emojis(emoji_df, top_n=15):
    top_emojis = emoji_df.head(top_n)
    fig = px.bar(top_emojis, x='emoji', y='count', title=f'Top {top_n} Emojis')
    return fig

def plot_sentiment_per_user(sentiment_df):
    fig = px.bar(sentiment_df, x='sender', y='polarity', title='Average Sentiment per User')
    return fig

def plot_reply_time_distribution(df):
    df = df.sort_values('datetime')
    df['reply_time_min'] = df['datetime'].diff().dt.total_seconds() / 60
    reply_times = df['reply_time_min'].dropna()
    fig = px.histogram(reply_times, nbins=50, title='Reply Time Distribution (minutes)')
    return fig

def plot_day_hour_heatmap(df):
    df['day'] = df['datetime'].dt.day_name()
    heatmap_data = df.groupby(['day', 'hour']).size().unstack(fill_value=0)
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_data = heatmap_data.reindex(days_order)
    plt.figure(figsize=(12,6))
    sns.heatmap(heatmap_data, cmap='YlGnBu')
    plt.title('Messages Heatmap by Day and Hour')
    plt.tight_layout()
    return plt.gcf()

def plot_media_message_count(df):
    media_msgs = df['message'].str.contains('<Media omitted>', na=False)
    counts = df.groupby('sender')[media_msgs].sum().reset_index()
    counts.columns = ['sender', 'media_messages']
    fig = px.bar(counts, x='sender', y='media_messages', title='Media Messages Count')
    return fig

def plot_messages_with_links(df):
    links = df['message'].str.contains(r'http[s]?://', regex=True, na=False)
    counts = df.groupby('sender')[links].sum().reset_index()
    counts.columns = ['sender', 'link_messages']
    fig = px.bar(counts, x='sender', y='link_messages', title='Messages Containing Links')
    return fig

def plot_longest_message_length(df):
    df['msg_len'] = df['message'].apply(lambda x: len(str(x)))
    longest = df.groupby('sender')['msg_len'].max().reset_index()
    fig = px.bar(longest, x='sender', y='msg_len', title='Longest Message Length by User')
    return fig

def plot_wordcloud(df):
    text = " ".join(df['message'].dropna().astype(str).tolist())
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10,5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title("Word Cloud")
    return plt.gcf()

def plot_cumulative_messages(df):
    daily_counts = df.groupby('date').size().cumsum().reset_index(name='cumulative_messages')
    fig = px.line(daily_counts, x='date', y='cumulative_messages', title='Cumulative Messages Over Time')
    return fig
