# Forecasting Earnings Surprises from Conference Call Transcripts Replication Study

This repository contains our efforts to replicate the findings of the study "Forecasting Earnings Surprises from Conference Call Transcripts" originally conducted by Ross Koval, Nicholas Andrews, and Xifeng Yan.

Contributors: Aryan Garg, Sudeep Dharanikota, Cameron Cooray of Georgia Institute of Technology.

## Abstract
Our project attempts to replicate the methodology used in the original study to forecast earnings surprises by analyzing earnings call transcripts. Despite the challenges of missing original datasets and codebases, we developed our dataset and applied pre-trained BERT models for sentiment analysis, aiming to correlate linguistic features with financial outcomes.

## Data Collection
We sourced earnings call transcripts from major U.S. corporations and manually checked each for correspondence to the correct fiscal quarters and companies.

## Preprocessing
Implemented using Python, we cleaned the text data by removing non-alphabetic characters and numbers, standardized case, and removed stopwords using NLTK.

## Sentiment Analysis
Using the Hugging Face Transformers library, we applied a pre-trained BERT model for sentiment analysis, deriving sentiment scores which were crucial for our predictive modeling.

## Data Visualization
Various visualizations were employed to illustrate sentiment distribution and its relationship with financial performance metrics such as Standardized Unexpected Earnings (SUE).

## Model Training and Evaluation
We trained our models focusing on the relationships between linguistic features and financial outcomes and validated our results using statistical analyses.

Data 
- dataset_2022.csv: Dataset containing earnings call transcripts for the year 2022.
- dataset_2023.csv: Dataset containing earnings call transcripts for the year 2023.
- dataset_with_sentiments22.csv: Dataset containing sentiment scores for earnings call transcripts in 2022.
- dataset_with_sentiments23.csv: Dataset containing sentiment scores for earnings call transcripts in 2023.
- transcripts_22: Folder containing raw transcript text files for the year 2022.
- transcripts_23: Folder containing raw transcript text files for the year 2023.

Code
- preprocess-transcripts.py: Python script for preprocessing the raw transcript text files.
- sentiment_analysis.py: Python script for conducting sentiment analysis on preprocessed transcripts.
- graph.py, graphscatter.py, graphtwo.py, table.py: Data Visulization files


## Limitations and Challenges
We acknowledge the challenges we faced due to the absence of original datasets and codebases, which impacted our model's performance, especially the F1 score.

## Ethics Statement
Our research strictly adhered to the ACL Ethics Policy, ensuring responsible use of our findings and encouraging further research considering the ethical implications.

## Acknowledgements
Special thanks to Agam Akeshkumar Shah for his contribution to data presentation and to Michael Galarnyk for his guidance throughout the research process.

## References
- [Original Study by Ross Koval, Nicholas Andrews, and Xifeng Yan](https://aclanthology.org/2023.findings-acl.520.pdf)

## Contribution
If you'd like to contribute to this project or have any queries, please open an issue or submit a pull request.

---
**Note:** This project is a replication study and is meant for educational and research purposes only. The findings and methods are subject to the limitations stated in our study.
