import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
# Replace with the actual path
csv_path = '/Users/aryangarg/Documents/GitHub/VIP/dataset_with_sentiments.csv'
df = pd.read_csv(csv_path)

# Filter out rows with no score or no SUE
df_filtered = df.dropna(subset=['Average_Score', 'SUE'])

# Convert the 'Average_Score' and 'SUE' to numeric types (float)
df_filtered['Average_Score'] = pd.to_numeric(
    df_filtered['Average_Score'], errors='coerce')
df_filtered['SUE'] = pd.to_numeric(df_filtered['SUE'], errors='coerce')

# Drop rows where 'Average_Score' or 'SUE' could not be converted to float
df_filtered = df_filtered.dropna(subset=['Average_Score', 'SUE'])

# Plot a scatter plot of the SUE versus the average sentiment scores
plt.figure(figsize=(10, 5))
plt.scatter(df_filtered['SUE'], df_filtered['Average_Score'],
            alpha=0.5, s=50)  # Increased marker size
plt.title('Scatter Plot of SUE versus Average Sentiment Scores')
plt.xlabel('SUE')
plt.ylabel('Average Sentiment Score')

# Adjust the axes to zoom in on the majority of data points if necessary
plt.xlim(df_filtered['SUE'].min(), df_filtered['SUE'].max())
plt.ylim(df_filtered['Average_Score'].min(),
         df_filtered['Average_Score'].max())

plt.grid(True)
plt.show()
