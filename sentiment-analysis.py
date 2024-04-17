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
        # Parse the label to extract the number of stars
        if 'star' in sentiment_result['label']:
            # Extract the number before "star"
            score = int(sentiment_result['label'].split()[0])
            sentiment_scores.append(score)
        else:
            print(f"Unexpected label format: {sentiment_result['label']}")

    # Decide final sentiment based on the average score
    # Default to neutral if no scores
    average_score = np.mean(sentiment_scores) if sentiment_scores else 3
    final_sentiment = 'POSITIVE' if average_score > 3 else 'NEGATIVE' if average_score < 3 else 'NEUTRAL'

    return final_sentiment, sentiment_scores


# Directory paths
base_dir = '/Users/aryangarg/Documents/GitHub/VIP/transcripts'
csv_path = '/Users/aryangarg/Documents/GitHub/VIP/dataset.csv'
output_csv_path = '/Users/aryangarg/Documents/GitHub/VIP/dataset_with_sentiments23.csv'

# Read the dataset
df = pd.read_csv(csv_path)

# Initialize new columns
df['Sentiment'] = None
df['Average_Score'] = None

# Dictionary to map from CSV company names to folder names if they differ
company_name_mapping = {
    'JP Morgan': 'JP',
    # Modify as necessary based on your directory structure
    'UnitedHealth Group': 'UnitedHealth'
}

# Process each row in the dataframe
for index, row in df.iterrows():
    # Extracting company name and quarter from the CSV
    company_name = row['Company'].strip()  # Trim any whitespace
    # Map the company name to the folder name using the dictionary if necessary
    folder_name = company_name_mapping.get(company_name, company_name)
    quarter = row['Quarter'].strip()  # Trim any whitespace
    transcript_file = f"{quarter}.txt"
    transcript_path = os.path.join(base_dir, folder_name, transcript_file)

    # Debugging print statements
    print(f"Looking for file at path: {transcript_path}")

    # Check if the file exists
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

# Save the updated dataframe to a new CSV file
df.to_csv(output_csv_path, index=False)

print("Sentiment analysis complete. Results saved.")
