import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv(
    '/Users/aryangarg/Documents/GitHub/VIP/dataset_with_sentiments.csv')

# Assuming 'Average_Score' column contains the sentiment scores
# Filter out rows with no score
df_filtered = df.dropna(subset=['Average_Score'])

# Convert the 'Average_Score' to a numeric type (float)
df_filtered['Average_Score'] = pd.to_numeric(
    df_filtered['Average_Score'], errors='coerce')

# Drop rows where 'Average_Score' could not be converted to float
df_filtered = df_filtered.dropna(subset=['Average_Score'])

# Plot a histogram of the average sentiment scores
plt.hist(df_filtered['Average_Score'], bins=20, edgecolor='black')
plt.title('Distribution of Average Sentiment Scores')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.show()
