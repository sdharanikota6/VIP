import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv('/Users/aryangarg/Documents/GitHub/VIP/dataset_with_sentiments.csv')

# Preprocess 'Quarter' to extract a common format if necessary (e.g., 'Q1 2023')
# df['Quarter'] = df['Quarter'].apply(lambda x: 'Q' + x.split(' ')[0][-1] + ' ' + x.split(' ')[2])

# Pivot your data so that companies and quarters form the axes and sentiment scores fill the cells
heatmap_data = pd.pivot_table(df, values='Average_Score', index='Quarter', columns='Company', aggfunc='mean')

# Generate the heatmap
sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap='coolwarm')

# Set the title and show the plot
plt.title('Heatmap of Average Sentiment Scores by Company and Quarter')
plt.xticks(rotation=45, ha='right')  # Improve x-axis labels readability
plt.tight_layout()  # Adjust layout to fit the plot and labels
plt.show()
