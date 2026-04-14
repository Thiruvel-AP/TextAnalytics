# Import the necessary modules 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from gensim import matutils

# Heat map for the topic models 
def plot_topicModel_heatmap(model, num_words=10):
    topics = model.show_topics(formatted=False, num_words=num_words)
    topic_words = []
    for i, topic in topics:
        for word, weight in topic:
            topic_words.append({'Topic': f'Topic {i}', 'Word': word, 'Weight': weight})

    df_topics = pd.DataFrame(topic_words)
    df_pivot = df_topics.pivot(index='Word', columns='Topic', values='Weight').fillna(0)

    plt.figure(figsize=(10, 8))
    sns.heatmap(df_pivot, annot=True, cmap="YlGnBu")
    plt.title("LDA Top Words Weight Heatmap")
    plt.show()

# Histogram of topics 
def plot_topicModel_histogram(model, corpus):
    # Check if the corpus is a scipy sparse matrix and convert if necessary
    if hasattr(corpus, "tocsr"):
        corpus = matutils.Sparse2Corpus(corpus, documents_columns=False)

    dominant_topics = []
    for doc in corpus:
        # doc will now be the (index, value) tuple list Gensim expects
        topics = model.get_document_topics(doc)
        dominant_topic = max(topics, key=lambda x: x[1])[0]
        dominant_topics.append(dominant_topic)

    plt.figure(figsize=(10, 5))
    plt.hist(dominant_topics, bins=np.arange(model.num_topics + 1) - 0.5, 
             rwidth=0.8, color='skyblue', edgecolor='black')
    plt.xticks(range(model.num_topics))
    plt.xlabel('Topic ID')
    plt.ylabel('Number of Documents')
    plt.title('Distribution of Dominant Topics (LDA)')
    plt.show()

