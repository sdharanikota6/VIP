import pandas as pd

# Define your metrics
metrics = {
    'Metric': ['Precision', 'Recall', 'F1 Score'],
    'Value': [0.75, 0.48, 0.5853658536585366]
}

# Create a DataFrame
metrics_df = pd.DataFrame(metrics)

# Display the DataFrame
print(metrics_df)

# Save to a CSV file if needed
metrics_df.to_csv('/Users/aryangarg/Documents/GitHub/VIP/metrics_results.csv', index=False)
