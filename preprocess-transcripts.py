import os
import re
import ssl
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Ensure NLTK's tokenizers and stopwords are available
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')
nltk.download('stopwords')

# Directory where transcripts are stored
transcripts_dir = 'transcripts'

def clean_text(text):
    # Remove anything that is not a letter or space
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove stopwords
    filtered_tokens = [w for w in tokens if not w in stopwords.words('english')]
    # Rejoin tokens
    cleaned_text = ' '.join(filtered_tokens)
    return cleaned_text

def preprocess_and_save_transcripts(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                cleaned_text = clean_text(text)
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(cleaned_text)

# Run the function to preprocess and save the transcripts
preprocess_and_save_transcripts(transcripts_dir)
