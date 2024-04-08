import os
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Make sure NLTK's tokenizers and stopwords are available
import nltk
nltk.download('punkt')
nltk.download('stopwords')

# This is the folder where all your company folders are stored.
transcripts_dir = 'transcripts'

# Function to clean a given text.


def clean_text(text):
    # Remove anything that is not a letter or space. Also, convert to lowercase.
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I | re.A).lower()
    # Tokenize the text - split into individual words.
    tokens = word_tokenize(text)
    # Remove stopwords (common words that are usually not informative for analysis).
    filtered_tokens = [
        token for token in tokens if token not in stopwords.words('english')]
    # Join the tokens back into a string.
    cleaned_text = ' '.join(filtered_tokens)
    return cleaned_text

# Function to preprocess all transcripts in the specified directory.


def preprocess_transcripts(directory):
    # This dictionary will store all the cleaned transcripts.
    all_transcripts = {}
    # Loop through each company folder.
    for company in os.listdir(directory):
        company_path = os.path.join(directory, company)
        # Check if the path is indeed a directory/folder.
        if os.path.isdir(company_path):
            all_transcripts[company] = {}
            # Loop through each text file in the company's folder.
            for file in os.listdir(company_path):
                # Make sure to process only text files.
                if file.endswith(".txt"):
                    # Extract quarter from the file name, assuming 'Q1.txt', 'Q2.txt', etc.
                    quarter = file.split('.')[0]
                    with open(os.path.join(company_path, file), 'r', encoding='utf-8') as f:
                        text = f.read()  # Read the file's content.
                        cleaned_text = clean_text(text)  # Clean the text.
                        # Store the cleaned text in the dictionary.
                        all_transcripts[company][quarter] = cleaned_text
    return all_transcripts


# Actually preprocess and load all transcripts using the defined function.
all_transcripts = preprocess_transcripts(transcripts_dir)

# Just to demonstrate, print the first 100 characters of Apple's Q1 transcript.
print(all_transcripts['Apple']['Q1'][:100])
