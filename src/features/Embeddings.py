# Import the necessary modules 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# NLP
from sentence_transformers import SentenceTransformer
import gensim.downloader as api

# ML
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Settings
skipgram_model = api.load('word2vec-google-news-300')
sbert = SentenceTransformer('all-MiniLM-L6-v2')

# Step 8 - Vectorization/Embeddings with BOA, TF-IDF and SBERT(sentence)
# bag of words vectorization
bow_vectorizer = CountVectorizer(
    ngram_range=(1, 2),
    max_features=5000,
    max_df=0.85
)
def vectorize_boa(text_list:list):
    bow = bow_vectorizer.fit_transform(text_list)
    return bow

# TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=5000,
    # Ignore rare term
    min_df=3,
    # Ignore common terms
    max_df=0.85
)
def vectorize_tfidf(text_list:list):
    tfidf = tfidf_vectorizer.fit_transform(text_list)
    return tfidf

# Skipgram embeddings with word2vec
def vectorize_skipgram(tokens:list):
    final_embeddings = []
    for token in tokens:
      print(token)
      embeddings = [
          skipgram_model[t]
          for t in token
          if t in skipgram_model
      ]
      final_embeddings.append(np.mean(embeddings, axis=0))
    return np.asarray(final_embeddings)

# SBERT embeddings
def vectorize_sbert(text_list:list):
    embeddings = sbert.encode(
        text_list.tolist(),
        show_progress_bar=True,
        batch_size=64
    )
    return embeddings