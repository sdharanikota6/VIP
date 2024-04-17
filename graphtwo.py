import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Load the data
df = pd.read_csv('/Users/aryangarg/Documents/GitHub/VIP/dataset_with_sentiments.csv')

# Convert Date to datetime and sort the dataframe
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values('Date', inplace=True)

# Filter for a single company, e.g., Apple
df_apple = df[df['Company'] == 'Apple']

# Assuming 'Average_Score' column contains the sentiment scores
df_apple['Average_Score'] = pd.to_numeric(df_apple['Average_Score'], errors='coerce')

# Plot a time series of the average sentiment scores
plt.figure(figsize=(10,5))
plt.plot(df_apple['Date'], df_apple['Average_Score'], marker='o')
plt.title('Time Series of Average Sentiment Scores for Apple')
plt.xlabel('Date')
plt.ylabel('Average Sentiment Score')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gcf().autofmt_xdate() # Rotation
plt.show()