import numpy as np
import os
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Load pre-trained model tokenizer and model
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Initialize sentiment analysis pipeline
nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

def get_sentiment(transcript):
    # Split the transcript into manageable chunks if needed
    max_length = tokenizer.model_max_length
    transcript_chunks = [transcript[i:i + max_length] for i in range(0, len(transcript), max_length)]

    # Count the sentiments for each chunk
    sentiment_counts = {'POSITIVE': 0, 'NEGATIVE': 0, 'NEUTRAL': 0}
    for chunk in transcript_chunks:
        sentiment_result = nlp(chunk)[0]
        if sentiment_result['label'] in ('LABEL_4', 'LABEL_5'):  # Assuming these are positive labels
            sentiment_counts['POSITIVE'] += 1
        elif sentiment_result['label'] in ('LABEL_0', 'LABEL_1'):  # Assuming these are negative labels
            sentiment_counts['NEGATIVE'] += 1
        else:
            sentiment_counts['NEUTRAL'] += 1

    final_sentiment = max(sentiment_counts, key=sentiment_counts.get)
    return final_sentiment, sentiment_counts

# Directory paths
base_dir = '/Users/aryangarg/Documents/GitHub/VIP/transcripts'
csv_path = '/Users/aryangarg/Documents/GitHub/VIP/dataset.csv'
output_csv_path = '/Users/aryangarg/Documents/GitHub/VIP/dataset_with_sentiments.csv'

# Read the dataset
df = pd.read_csv(csv_path)

# Initialize new columns
df['Sentiment'] = None
df['Positive_Count'] = None
df['Negative_Count'] = None
df['Neutral_Count'] = None

# Process each row in the dataframe
for index, row in df.iterrows():
    # Extracting company name and quarter from the CSV
    company_name = row['Company']
    quarter = row['Quarter']
    transcript_file = f"{quarter}.txt"
    transcript_path = os.path.join(base_dir, company_name, transcript_file)

    # Check if the file exists
    if os.path.isfile(transcript_path):
        with open(transcript_path, 'r') as file:
            transcript = file.read()
        sentiment, counts = get_sentiment(transcript)
        df.at[index, 'Sentiment'] = sentiment
        df.at[index, 'Positive_Count'] = counts['POSITIVE']
        df.at[index, 'Negative_Count'] = counts['NEGATIVE']
        df.at[index, 'Neutral_Count'] = counts['NEUTRAL']
    else:
        df.at[index, 'Sentiment'] = "Not Found"

# Save the updated dataframe to a new CSV file
df.to_csv(output_csv_path, index=False)