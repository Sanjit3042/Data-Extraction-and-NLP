import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk

nltk.download('punkt')
nltk.download('stopwords')

# Define file paths
input_file_path = r'E:\DATA\Input.xlsx'
output_file_path = r'E:\DATA\Output Data Structure.xlsx'
stopwords_dir = r'E:\DATA\StopWords'
master_dict_dir = r'E:\DATA\MasterDictionary'

# Function to load stopwords from the directory
def load_stopwords(stopwords_dir):
    stop_words = set(stopwords.words('english'))
    for file_name in os.listdir(stopwords_dir):
        with open(os.path.join(stopwords_dir, file_name), 'r', encoding='ISO-8859-1') as file:
            stop_words.update(file.read().splitlines())
    return stop_words

# Function to load master dictionary for positive and negative words
def load_master_dictionary(master_dict_dir):
    positive_words = set()
    negative_words = set()
    for file_name in ['positive-words.txt', 'negative-words.txt']:
        file_path = os.path.join(master_dict_dir, file_name)
        with open(file_path, 'r', encoding='ISO-8859-1') as file:
            if 'positive' in file_name:
                positive_words.update(file.read().splitlines())
            else:
                negative_words.update(file.read().splitlines())
    return positive_words, negative_words

# Function to extract article text
def extract_article_text(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise HTTPError for bad responses
        soup = BeautifulSoup(response.content, 'html.parser')
        title = soup.title.string if soup.title else ''
        article_text = ' '.join(p.get_text() for p in soup.find_all('p'))
        return title + '\n' + article_text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return ''

# Load input data
input_df = pd.read_excel(input_file_path)
output_df = pd.read_excel(output_file_path)

# Load stopwords and master dictionary
stop_words = load_stopwords(stopwords_dir)
positive_words, negative_words = load_master_dictionary(master_dict_dir)

# Function to calculate sentiment scores
def calculate_sentiment_scores(text, positive_words, negative_words):
    tokens = word_tokenize(text)
    positive_score = sum(1 for word in tokens if word in positive_words)
    negative_score = sum(1 for word in tokens if word in negative_words)
    return positive_score, negative_score

# Function to calculate polarity score
def calculate_polarity_score(positive_score, negative_score):
    return (positive_score - negative_score) / ((positive_score + negative_score) + 0.000001)

# Function to calculate subjectivity score
def calculate_subjectivity_score(positive_score, negative_score, total_words):
    return (positive_score + negative_score) / (total_words + 0.000001)

# Function to calculate average sentence length
def calculate_avg_sentence_length(text):
    sentences = sent_tokenize(text)
    if len(sentences) == 0:
        return 0  # Handle zero sentences
    return len(word_tokenize(text)) / len(sentences)

# Function to calculate percentage of complex words
def calculate_percentage_complex_words(text):
    words = word_tokenize(text)
    if len(words) == 0:
        return 0  # Handle zero words
    complex_words = [word for word in words if len(word) > 2]
    return len(complex_words) / len(words)

# Function to calculate fog index
def calculate_fog_index(avg_sentence_length, percentage_complex_words):
    return 0.4 * (avg_sentence_length + percentage_complex_words)

# Function to calculate average words per sentence
def calculate_avg_words_per_sentence(text):
    sentences = sent_tokenize(text)
    if len(sentences) == 0:
        return 0  # Handle zero sentences
    words = word_tokenize(text)
    return len(words) / len(sentences)

# Function to calculate complex word count
def calculate_complex_word_count(text):
    words = word_tokenize(text)
    return len([word for word in words if len(word) > 2])

# Function to calculate word count
def calculate_word_count(text):
    words = word_tokenize(text)
    return len(words)

# Function to calculate syllables per word
def calculate_syllables_per_word(text):
    vowels = "aeiou"
    words = word_tokenize(text)
    if len(words) == 0:
        return 0  # Handle zero words
    syllable_count = sum(sum(1 for char in word if char in vowels) for word in words)
    return syllable_count / len(words)

# Function to calculate personal pronouns
def calculate_personal_pronouns(text):
    pronouns = ['I', 'we', 'my', 'ours', 'us']
    tokens = word_tokenize(text)
    return sum(1 for word in tokens if word in pronouns)

# Function to calculate average word length
def calculate_avg_word_length(text):
    words = word_tokenize(text)
    if len(words) == 0:
        return 0  # Handle zero words
    return sum(len(word) for word in words) / len(words)

# Extract text and perform analysis for each URL
for url_id, url in zip(input_df['URL_ID'], input_df['URL']):
    article_text = extract_article_text(url)
    with open(f"{url_id}.txt", 'w', encoding='utf-8') as f:
        f.write(article_text)
    print(f"Extracted text for URL_ID {url_id}: {article_text[:100]}...")  # Print first 100 characters for verification

for file_name in os.listdir('.'):
    if file_name.endswith('.txt'):
        with open(file_name, 'r', encoding='utf-8') as file:
            text = file.read()

        if not text.strip():  # Skip empty text
            continue

        # Calculating sentiment scores
        positive_score, negative_score = calculate_sentiment_scores(text, positive_words, negative_words)
        print(f"Positive Score for {file_name}: {positive_score}")
        print(f"Negative Score for {file_name}: {negative_score}")

        # Calculating other metrics
        polarity_score = calculate_polarity_score(positive_score, negative_score)
        subjectivity_score = calculate_subjectivity_score(positive_score, negative_score, len(text.split()))
        avg_sentence_length = calculate_avg_sentence_length(text)
        percentage_complex_words = calculate_percentage_complex_words(text)
        fog_index = calculate_fog_index(avg_sentence_length, percentage_complex_words)
        avg_words_per_sentence = calculate_avg_words_per_sentence(text)
        complex_word_count = calculate_complex_word_count(text)
        word_count = calculate_word_count(text)
        syllable_per_word = calculate_syllables_per_word(text)
        personal_pronouns = calculate_personal_pronouns(text)
        avg_word_length = calculate_avg_word_length(text)

        print(f"Calculated metrics for {file_name}")

for i, row in input_df.iterrows():
    url_id = row['URL_ID']
    file_name = f"{url_id}.txt"
    if file_name in os.listdir('.'):
        with open(file_name, 'r', encoding='utf-8') as file:
            text = file.read()

        if not text.strip():  # Skip empty text
            continue

        # Calculating sentiment scores
        positive_score, negative_score = calculate_sentiment_scores(text, positive_words, negative_words)
        polarity_score = calculate_polarity_score(positive_score, negative_score)
        subjectivity_score = calculate_subjectivity_score(positive_score, negative_score, len(text.split()))
        avg_sentence_length = calculate_avg_sentence_length(text)
        percentage_complex_words = calculate_percentage_complex_words(text)
        fog_index = calculate_fog_index(avg_sentence_length, percentage_complex_words)
        avg_words_per_sentence = calculate_avg_words_per_sentence(text)
        complex_word_count = calculate_complex_word_count(text)
        word_count = calculate_word_count(text)
        syllable_per_word = calculate_syllables_per_word(text)
        personal_pronouns = calculate_personal_pronouns(text)
        avg_word_length = calculate_avg_word_length(text)

        output_df.at[i, 'POSITIVE SCORE'] = positive_score
        output_df.at[i, 'NEGATIVE SCORE'] = negative_score
        output_df.at[i, 'POLARITY SCORE'] = polarity_score
        output_df.at[i, 'SUBJECTIVITY SCORE'] = subjectivity_score
        output_df.at[i, 'AVG SENTENCE LENGTH'] = avg_sentence_length
        output_df.at[i, 'PERCENTAGE OF COMPLEX WORDS'] = percentage_complex_words
        output_df.at[i, 'FOG INDEX'] = fog_index
        output_df.at[i, 'AVG NUMBER OF WORDS PER SENTENCE'] = avg_words_per_sentence
        output_df.at[i, 'COMPLEX WORD COUNT'] = complex_word_count
        output_df.at[i, 'WORD COUNT'] = word_count
        output_df.at[i, 'SYLLABLE PER WORD'] = syllable_per_word
        output_df.at[i, 'PERSONAL PRONOUNS'] = personal_pronouns
        output_df.at[i, 'AVG WORD LENGTH'] = avg_word_length

        print(f"Writing metrics for URL_ID {url_id} to output file")

output_df.to_excel(output_file_path, index=False)
print(f"Output written to {output_file_path}")
