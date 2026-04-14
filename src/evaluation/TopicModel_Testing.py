from gensim.matutils import Sparse2Corpus
from gensim.models import CoherenceModel
import pandas as pd 
from gensim.corpora import Dictionary
import numpy as np
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
import numpy as np

def sparse_to_gensim_bow(sparse_matrix):
    """Convert a scipy sparse matrix (docs × vocab) to a list of gensim BoW docs."""
    gensim_corpus = []
    cx = sparse_matrix.tocsr()          # ensure CSR format for row slicing
    for i in range(cx.shape[0]):
        row = cx.getrow(i).tocoo()      # COO gives .col (indices) and .data (values)
        gensim_corpus.append(list(zip(row.col.tolist(), row.data.tolist())))
    return gensim_corpus


def evaluate_models(model_list, test_corpus, texts, model_type_name=""):
    results     = []
    token_lists = list(texts)
    gensim_dict = Dictionary(test_tokens) 
    
    print(f"✅ gensim_dict type   : {type(gensim_dict)}")
    print(f"   Vocabulary size    : {len(gensim_dict.token2id)}")
    print(f"   Num docs           : {gensim_dict.num_docs}")
    for model in model_list:
        n_topics = model.num_topics
        coherence_model = CoherenceModel(
            model      = model,
            texts      = token_lists,
            dictionary = gensim_dict,
            coherence  = 'c_v'
        )
        score = coherence_model.get_coherence()
        print(f"[{model_type_name}] {n_topics:>2} topics — C_v coherence: {score:.4f}")
        results.append({
            'model_type'   : model_type_name,
            'n_topics'     : n_topics,
            'coherence_cv' : score
        })
    return results