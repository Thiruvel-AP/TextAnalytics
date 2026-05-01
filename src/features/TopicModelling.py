#### TOPPIC MODELLING AS A PRIMARY AXIS (AXIS 1)

# Import the necessary modules 
from gensim.corpora import Dictionary
from gensim.models import LdaModel, HdpModel
from bertopic import BERTopic
from gensim import matutils

def initialize_ldaModel(sparse_matrix, vectorizer, num_topics=20, passes=10):
    # 1. Convert Scikit-Learn vocabulary to Gensim Dictionary format
    # vectorizer.get_feature_names_out() gives the list of words in order
    id2word = dict((i, word) for i, word in enumerate(vectorizer.get_feature_names_out()))
    
    # 2. Convert Sparse Matrix to Gensim Corpus
    corpus = matutils.Sparse2Corpus(sparse_matrix, documents_columns=False)
    
    # 3. Initialize LDA
    return (LdaModel(
        corpus=corpus,
        num_topics=num_topics,
        id2word=id2word,
        passes=passes
    ), id2word)

# BERT topic Model 
def bert_topicModel(text):
    topic_model = BERTopic()
    topics, probs = topic_model.fit_transform(text)

    return topic_model, topics, probs