import pandas as pd
import numpy as np
import re
from textblob import TextBlob

def analyze_chat(df):
    # General stats
    total_messages = len(df)
    total_users = df['sender'].nunique()

    # Extract emojis
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
    emojis = df['message'].apply(lambda x: emoji_pattern.findall(x))
    all_emojis = [e for sublist in emojis for e in sublist]
    emoji_counts = pd.Series(all_emojis).value_counts().reset_index()
    emoji_counts.columns = ['emoji', 'count']

    # Sentiment per user
    sentiment_data = []
    for sender in df['sender'].unique():
        messages = df[df['sender'] == sender]['message']
        polarity = messages.apply(lambda x: TextBlob(x).sentiment.polarity if x else 0).mean()
        sentiment_data.append({'sender': sender, 'polarity': polarity})
    sentiment_df = pd.DataFrame(sentiment_data)

    stats = {
        "Total Messages": total_messages,
        "Total Users": total_users,
        "Unique Emojis": len(emoji_counts),
    }

    return stats, emoji_counts, sentiment_df
