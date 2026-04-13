# Import the necessary modules 
import contractions
import re
from gensim.parsing.preprocessing import STOPWORDS
from nltk.tokenize import word_tokenize
from src.configfiles.config import custom_stopwords, tag_map

# method to get the text and change to lower case
def to_lower(text):
  return str(text).lower()

# Expand common contractions with regex before noise removal
def expand_contractions(text):
    return contractions.fix(text)

# Named Entity Recoginization : Replace {product_purchased} with "PRODUCT" with regex substution
def replace_product_purchased(text):
  text = re.sub(
          r'\{product_purchased\}',
          'PRODUCT',
          text,
          flags=re.IGNORECASE
      )
  return text

# Remove Noise 
def remove_noise(text):
    text = re.sub(r'\S+@\S+\.\S+', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'\b\d{5}\b', '', text)
    text = re.sub(r'\b\d+(\.\d+)+\b', '', text)
    text = re.sub(r'\b\d+\b', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Text tokenizations 
def tokenize_text(text):
    return word_tokenize(text)

# remove the Stopwords 
def remove_stopwords_all(tokens):
    return [w for w in tokens if w not in STOPWORDS and w not in custom_stopwords]

# filter the unigram and bigram words 
def filter_short_tokens(tokens, min_len=3):
    return [t for t in tokens if len(t) >= min_len]

# Wordnet Lemmatizer instance
lemmatizer = WordNetLemmatizer()

# Lemmatise the tokens 
def lemmatize_tokens(tokens):
    tagged = pos_tag(tokens)
    return [lemmatizer.lemmatize(token, tag_map.get(tag[0].upper(), 'n'))
            for token, tag in tagged]
