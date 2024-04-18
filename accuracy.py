import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

# Load the dataset
df = pd.read_csv('/Users/aryangarg/Documents/GitHub/VIP/dataset_with_sentiments23.csv')

# Remove the dollar sign from 'SUE' and convert to float
df['SUE'] = df['SUE'].str.replace('$', '').astype(float)

# Convert 'Sentiment' to binary: 1 for 'POSITIVE', 0 for 'NEGATIVE'
df['Sentiment_Binary'] = df['Sentiment'].apply(lambda x: 1 if x == 'POSITIVE' else 0)

# Convert 'SUE' to binary: 1 for positive SUE, 0 for negative SUE
df['SUE_Binary'] = df['SUE'].apply(lambda x: 1 if x > 0 else 0)

# Calculate precision, recall, and F1 score
precision = precision_score(df['SUE_Binary'], df['Sentiment_Binary'])
recall = recall_score(df['SUE_Binary'], df['Sentiment_Binary'])
f1 = f1_score(df['SUE_Binary'], df['Sentiment_Binary'])

# Print the scores
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
