import numpy as np
import os
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Define the model to use
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"

# Load pre-trained model tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Initialize sentiment analysis pipeline
nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)


def get_sentiment(transcript):
    # Split the transcript into manageable chunks if needed
    max_length = tokenizer.model_max_length
    transcript_chunks = [transcript[i:i + max_length]
                         for i in range(0, len(transcript), max_length)]

    # Aggregate the sentiment scores for each chunk
    sentiment_scores = []
    for chunk in transcript_chunks:
        sentiment_result = nlp(chunk)[0]
        # Parse the label to extract the sentiment score
        label = sentiment_result['label']
        if 'star' in label:
            score = int(label.split()[0])
            sentiment_scores.append(score)
        else:
            print(f"Unexpected label format: {label}")

    # Decide final sentiment based on the average score
    # Default to neutral if no scores
    average_score = np.mean(sentiment_scores) if sentiment_scores else 3
    final_sentiment = 'POSITIVE' if average_score > 3 else 'NEGATIVE' if average_score < 3 else 'NEUTRAL'

    return final_sentiment, sentiment_scores


def process_dataset(df, base_dir):
    for index, row in df.iterrows():
        company_name = row['Company'].strip()  # Trim any whitespace
        quarter = row['Quarter'].strip()  # Trim any whitespace
        folder_name = company_name_mapping.get(company_name, company_name)
        transcript_file = f"{quarter}.txt"
        transcript_path = os.path.join(base_dir, folder_name, transcript_file)

        print(f"Looking for file at path: {transcript_path}")

        if os.path.isfile(transcript_path):
            print("File found. Processing...")
            with open(transcript_path, 'r') as file:
                transcript = file.read()
            sentiment, scores = get_sentiment(transcript)
            df.at[index, 'Sentiment'] = sentiment
            df.at[index, 'Average_Score'] = np.mean(
                scores) if scores else 'No Score'
        else:
            print("File not found.")
            df.at[index, 'Sentiment'] = "Not Found"
            df.at[index, 'Average_Score'] = 'No File'


# Dictionary to map from CSV company names to folder names if they differ
company_name_mapping = {
    'JP Morgan': 'JP',
    'UnitedHealth Group': 'UnitedHealth'
}

# Directory paths

# /Users/sudeepdharanikota/Desktop/Spring 2024/NLP VIP/Replication
base_dir_23 = '/Users/sudeepdharanikota/Desktop/Spring 2024/NLP VIP/Replication/transcripts_23'
csv_path_23 = '/Users/sudeepdharanikota/Desktop/Spring 2024/NLP VIP/Replication/dataset_23.csv'
output_csv_path_23 = '/Users/sudeepdharanikota/Desktop/Spring 2024/NLP VIP/Replication/dataset_with_sentiments23.csv'

base_dir_22 = '/Users/sudeepdharanikota/Desktop/Spring 2024/NLP VIP/Replication/transcripts_22'
csv_path_22 = '/Users/sudeepdharanikota/Desktop/Spring 2024/NLP VIP/Replication/dataset_22.csv'
output_csv_path_22 = '/Users/sudeepdharanikota/Desktop/Spring 2024/NLP VIP/Replication/dataset_with_sentiments22.csv'

# Read the datasets
df_23 = pd.read_csv(csv_path_23)
df_22 = pd.read_csv(csv_path_22)

# Initialize new columns
df_23['Sentiment'] = None
df_23['Average_Score'] = None

df_22['Sentiment'] = None
df_22['Average_Score'] = None

# Process datasets
process_dataset(df_23, base_dir_23)
process_dataset(df_22, base_dir_22)

# Save the updated dataframes to new CSV files
df_23.to_csv(output_csv_path_23, index=False)
df_22.to_csv(output_csv_path_22, index=False)

print("Sentiment analysis complete. Results saved.")
