# Mining Insights from Customer Feedback: NLP & Unsupervised Analytics Pipeline

**Module:** EMATM0067  AI & Text Analytics · MSc Data Science, University of Bristol · Group G43  
**Task:** 4: Mining Insights from Airline Customer Feedback (IATA dataset)

An end-to-end NLP pipeline for unsupervised text analytics on customer support tickets from raw noisy text through a 9-step preprocessing chain, four text representations (BoW · TF-IDF · Word2Vec Skip-gram · SBERT), two clustering algorithms (K-Means · HAC), two topic models (LDA · BERTopic), and two sentiment analysers (VADER · TextBlob)  with quantitative evaluation across Silhouette Score, ARI, and NMI.

> **Stack:** `scikit-learn` · `gensim` · `sentence-transformers` · `BERTopic` · `vaderSentiment` · `TextBlob` · `nltk` · `spaCy` · Python 3.x

---

## 📎 Project Resources

| Resource | Link |
|---|---|
| 📓 **Google Colab Notebook** *(Thiruvel  Individual Contribution)* | [Open in Colab](https://colab.research.google.com/drive/1585pIeWTvfiePhdiELvrqmmkzva1ytNk?usp=sharing) |
| 📋 **Trello Management Board** | [Open Board](https://trello.com/invite/b/69bad5bb22fb691c08db9005/ATTI7ab96990d50f4b6b0d311f69f47c1b6619D4DFD1/ematm0067-g43-managementboard) |
| 📄 **Group Report (Google Docs)** | [Open Report](https://docs.google.com/document/d/1VO_XEv5D4y5fxlWlTrvToi3hPDUlGfPaGcILXEhZSTQ/edit?usp=sharing) |

---

## Pipeline Overview

```
Raw Customer Support Tickets (CSV)
            │
            ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 1  PREPROCESSING  (src/pipeline/Preprocessing.py) │
│                                                         │
│  1. to_lower()               → lowercasing              │
│  2. expand_contractions()    → "don't" → "do not"       │
│  3. replace_product_purchased()                         │
│       regex: {product_purchased} → "PRODUCT"            │
│  4. remove_noise()           → strip:                   │
│       emails, URLs, @mentions, 5-digit postcodes,       │
│       decimal/integer numbers, non-alpha chars,         │
│       excess whitespace                                 │
│  5. tokenize_text()          → NLTK word_tokenize()     │
│  6. remove_stopwords_all()   → Gensim STOPWORDS         │
│                                + 60-term domain list    │
│  7. filter_short_tokens()    → drop tokens < 3 chars    │
│  8. lemmatize_tokens()       → WordNetLemmatizer        │
│                                + POS tagging (pos_tag)  │
│                                J→adj, V→verb,           │
│                                N→noun, R→adverb         │
└──────────────────────────┬──────────────────────────────┘
                           │
            ┌──────────────┴──────────────┐
            ▼                             ▼
┌───────────────────────┐     ┌──────────────────────────┐
│  AXIS 1  TOPIC MODEL  │     │  AXIS 2  SENTIMENT       │
│  (src/features/)      │     │  (notebooks/04_pipeline/)│
│                       │     │                          │
│  LDA (BoW + TF-IDF)   │     │  VADER                   │
│  BERTopic             │     │  TextBlob (PatternAnal.) │
└──────────┬────────────┘     └────────────┬─────────────┘
           │                               │
           └──────────────┬────────────────┘
                          ▼
┌──────────────────────────────────────────────────────────┐
│  TEXT REPRESENTATION  (src/features/Embeddings.py)       │
│                                                          │
│  BoW       CountVectorizer  ngram(1,2)  max_feat=5000    │
│                              max_df=0.85                 │
│  TF-IDF    TfidfVectorizer  ngram(1,2)  max_feat=5000    │
│                              min_df=3   max_df=0.85      │
│  Skip-gram word2vec-google-news-300 (300d) mean-pooled   │
│  SBERT     all-MiniLM-L6-v2  batch_size=64               │
└──────────────────────────┬───────────────────────────────┘
                           │
            ┌──────────────┴──────────────┐
            ▼                             ▼
┌───────────────────────┐     ┌──────────────────────────┐
│  K-MEANS              │     │  HAC                     │
│  (src/clustering/)    │     │  (src/clustering/)       │
│                       │     │                          │
│  n_init=10            │     │  linkage='ward'          │
│  random_state=42      │     │  AgglomerativeClustering │
│  sparse + dense input │     │  sparse + dense input    │
└──────────┬────────────┘     └────────────┬─────────────┘
           └──────────────┬────────────────┘
                          ▼
┌───────────────────────────────────────────────────────────┐
│  EVALUATION  (src/evaluation/)                            │
│                                                           │
│  ClusterEvaluation.py                                     │
│    Silhouette Score  (sample_size=2000, euclidean)        │
│    ARI   Adjusted Rand Index   vs Ticket Type             │
│    NMI   Normalized Mutual Information vs Ticket Type     │
│                                                           │
│  Cluster_Interpretability.py                              │
│    Top-N words per cluster (mean TF-IDF barplots)         │
│    Heatmaps: cluster × Ticket Type (counts + proportions) │
│    Cluster size distribution bar charts                   │
│                                                           │
│  Cluster_Stability.py                                     │
│    Metric vs k line charts (fill_between)                 │
│    Grid: Metrics × Representations                        │
│    Metrics: Silhouette, ARI(Type), NMI(Type),             │
│             ARI(Subject), NMI(Subject)                    │
│                                                           │
│  TopicModel_Visualization.py                              │
│    LDA word-weight heatmap (num_topics × num_words)       │
│    Dominant topic histogram                               │
└───────────────────────────────────────────────────────────┘
```

---

## Key Technical Details

### 1. Preprocessing Chain (`src/pipeline/Preprocessing.py`)

**Contraction expansion**  `contractions.fix()` runs before noise removal so that "don't" is expanded to "do not" before punctuation is stripped  order matters here to preserve negation signal.

**NER-style placeholder replacement**  `{product_purchased}` template tokens in the raw data are replaced with the literal string `"PRODUCT"` via case-insensitive regex substitution before noise removal, preventing the curly-brace artifact from corrupting downstream vocabulary.

**Noise removal**  single `remove_noise()` pass strips in order: email addresses, HTTP/WWW URLs, `@mentions`, 5-digit postcodes, decimal numbers (`\b\d+(\.\d+)+\b`), standalone integers, non-alphabetic characters, and collapses whitespace.

**Domain stopword list** (`src/configfiles/config.py`)  60-term frozen set extending Gensim's default STOPWORDS with domain-specific noise: customer service boilerplate (`please`, `thanks`, `regards`, `kindly`), generic ticket vocabulary (`issue`, `problem`, `request`, `support`, `response`), and high-frequency but uninformative verbs (`work`, `resolve`, `receive`, `provide`, `update`).

**POS-conditioned lemmatisation**  `pos_tag()` output is mapped through `tag_map = {'J': 'a', 'V': 'v', 'N': 'n', 'R': 'r'}` before `WordNetLemmatizer.lemmatize(token, pos)`  without this, the lemmatiser defaults all tokens to noun, producing incorrect forms for verbs and adjectives.

---

### 2. Text Representations (`src/features/Embeddings.py`)

| Method | Implementation | Key Parameters |
|---|---|---|
| **BoW** | `CountVectorizer` | `ngram_range=(1,2)` · `max_features=5000` · `max_df=0.85` |
| **TF-IDF** | `TfidfVectorizer` | `ngram_range=(1,2)` · `max_features=5000` · `min_df=3` · `max_df=0.85` |
| **Skip-gram** | `word2vec-google-news-300` (gensim API) | 300-dimensional · OOV tokens skipped · mean-pooled per document |
| **SBERT** | `all-MiniLM-L6-v2` | `batch_size=64` · 384-dimensional sentence embeddings |

**BoW vs TF-IDF design choice:** `min_df=3` on TF-IDF filters hapax legomena (terms appearing in fewer than 3 documents)  terms too rare to generalise. Both share `max_df=0.85` to suppress near-universal terms not caught by the stopword list.

**Skip-gram pooling:** OOV tokens (not in the Google News vocabulary) are silently skipped; the document vector is the arithmetic mean of in-vocabulary token embeddings  `np.mean(embeddings, axis=0)`. Documents where all tokens are OOV produce a zero vector.

**SBERT advantage over bag-of-words:** `all-MiniLM-L6-v2` encodes full sentence semantics  word order, negation, and contextual meaning are preserved in the 384-dimensional embedding space, unlike BoW and TF-IDF which are purely frequency-based.

---

### 3. Topic Modelling (`src/features/TopicModelling.py`)

**LDA (Latent Dirichlet Allocation):**
- Implemented via `gensim.models.LdaModel`
- Works on both BoW and TF-IDF sparse matrices  converted to Gensim corpus format via `matutils.Sparse2Corpus(sparse_matrix, documents_columns=False)`
- Scikit-learn vocabulary exported to Gensim `id2word` format via `dict(enumerate(vectorizer.get_feature_names_out()))`
- Parameters: `num_topics=20`, `passes=10` (configurable)
- Saved models: `lda_best_bow.pkl`, `lda_best_tfidf.pkl` + associated `id2word` dicts

**BERTopic:**
- `BERTopic().fit_transform(text)`  uses SBERT embeddings internally for semantic clustering before topic extraction
- Returns `(topic_model, topics, probs)` per document
- Saved topic assignments: `bert_train_topics.pkl`, `bert_test_topics.pkl`

**Comparison axis:** LDA is interpretable via word distributions per topic (directly inspectable); BERTopic produces more coherent topics by operating in semantic embedding space rather than bag-of-words space.

---

### 4. Sentiment Analysis

**VADER** (`vadersentiment`)  rule-based, lexicon-driven, optimised for short social-media-style text. Handles capitalisation, punctuation emphasis, and negations without training. Saved as `models/vader_sentiment_analyzer.pkl`.

**TextBlob** (`PatternAnalyzer`)  pattern-matching sentiment with continuous polarity score. Custom thresholds stored in `models/textblob_config.json`:
```json
{
  "thresholds": { "positive": 0.05, "negative": -0.05 },
  "saved_at": "2026-04-20T19:02:42.263899"
}
```
The asymmetric dead zone (−0.05 to +0.05 = neutral) prevents near-zero polarity scores from being misclassified.

---

### 5. Clustering (`src/clustering/`)

**K-Means:** `sklearn.cluster.KMeans(n_clusters=k, random_state=42, n_init=10)`  `n_init=10` runs 10 independent initialisations and selects the best by inertia. Accepts both sparse (CSR) and dense inputs via `.toarray()` guard.

**HAC (Hierarchical Agglomerative Clustering):** `AgglomerativeClustering(n_clusters=k, linkage='ward')`  Ward linkage minimises within-cluster variance at each merge step, producing compact, roughly equal-sized clusters. Dense-only: sparse matrices converted via `.toarray()` before fitting.

---

### 6. Evaluation (`src/evaluation/`)

**Quantitative metrics (`ClusterEvaluation.py`):**

| Metric | Function | Notes |
|---|---|---|
| **Silhouette Score** | `silhouette_score(X, labels, sample_size=2000)` | `sample_size=2000` for tractability on large matrices |
| **ARI** | `adjusted_rand_score(true_labels, cluster_labels)` | Chance-corrected; 0 = random, 1 = perfect |
| **NMI** | `normalized_mutual_info_score(true_labels, cluster_labels)` | Information-theoretic; robust to cluster count differences |

Ground truth labels are drawn from the `Ticket Type` and `Subject` columns  enabling dual evaluation against both categorical dimensions.

**Interpretability (`Cluster_Interpretability.py`):**
- Per-cluster top-N term bar charts: cluster mean TF-IDF weight → `argsort()[-N:][::-1]` → horizontal bar chart with `tab10` colour palette
- Ticket Type cross-tabulation heatmaps: raw counts (`Blues` cmap) + row-normalised proportions (`Oranges` cmap) side-by-side
- Cluster size charts with count and percentage annotations per bar

**Stability (`Cluster_Stability.py`):**
- Grid plot: rows = metrics (Silhouette, ARI, NMI by Type/Subject), columns = representations (BoW, TF-IDF, SBERT, Skip-gram)
- Metric vs k line charts with `fill_between` shading and annotated point values
- Fixed y-axis: Silhouette → [0, 0.4]; ARI/NMI → [−0.05, 0.1]
- Filtered by `algorithm` column  runs separately for KMeans and HAC

**Topic model visualisations (`TopicModel_Visualization.py`):**
- LDA word-weight heatmap: `show_topics(formatted=False)` → pivot to word × topic matrix → `sns.heatmap(cmap="YlGnBu")`
- Dominant topic histogram: `get_document_topics(doc)` → `argmax` per document → histogram over topic IDs

---

### 7. Dimensionality Reduction (`src/evaluation/visualization.py`, notebooks)

- **PCA**  `sklearn.decomposition.PCA(n_components=2)` with explained variance reporting per component and total; used for 2D cluster scatter plots
- **LSA**  Latent Semantic Analysis (`notebooks/02_clustering/LSA_PCA_UMAP.ipynb`)
- **UMAP**  Uniform Manifold Approximation and Projection; non-linear manifold reduction (`notebooks/02_clustering/LSA_PCA_UMAP.ipynb`)

**Representation comparison plot (`compare_representations`):** 4-panel side-by-side scatter (BoW · TF-IDF · SBERT · Skip-gram), each reduced to 2D via PCA, coloured by `label_col`, saved at 300 DPI.

---

## Project Structure

```
EMATM0067_2025_TB-2-g43/
├── Workflow.drawio                    # Architecture / pipeline diagram
├── plan.pdf                           # Project plan
├── requirements.txt
│
├── notebooks/
│   ├── 01_eda/
│   │   ├── EDA-ThiruvelAP.ipynb       # ← Thiruvel's individual EDA
│   │   ├── EDA-Fahmi.ipynb
│   │   ├── EDA-Ukhash.ipynb
│   │   ├── EDA-Xiang.ipynb
│   │   └── EDA-Zhaoqi.ipynb
│   ├── 02_clustering/
│   │   ├── ClusterningAnalysis.ipynb  # K-Means & HAC experiments
│   │   └── LSA_PCA_UMAP.ipynb        # Dimensionality reduction comparison
│   ├── 03_text_representation/
│   │   └── Text-Preprocessing.ipynb  # Preprocessing pipeline walkthrough
│   ├── 04_pipeline/
│   │   ├── EmbeddingsVisualization.ipynb  # BoW/TF-IDF/SBERT/Skipgram plots
│   │   ├── SentimentAnalysis.ipynb        # VADER + TextBlob pipeline
│   │   └── TopicModelling.ipynb           # LDA + BERTopic
│   └── 05_evaluation/
│       └── Evaluation.ipynb               # Full quantitative evaluation
│
├── src/
│   ├── pipeline/
│   │   └── Preprocessing.py          # 9-step text preprocessing chain
│   ├── features/
│   │   ├── Embeddings.py             # BoW, TF-IDF, Skip-gram, SBERT
│   │   └── TopicModelling.py         # LDA (gensim) + BERTopic
│   ├── clustering/
│   │   ├── KMeans.py                 # sklearn KMeans wrapper
│   │   └── Hac.py                    # AgglomerativeClustering (ward)
│   ├── evaluation/
│   │   ├── ClusterEvaluation.py      # Silhouette, ARI, NMI
│   │   ├── Cluster_Interpretability.py  # Top-words, heatmaps, size charts
│   │   ├── Cluster_Stability.py      # Metric vs k grid plots
│   │   ├── TopicModel_Visualization.py  # LDA heatmap + histogram
│   │   └── visualization.py          # PCA reduction + scatter comparison
│   ├── configfiles/
│   │   └── config.py                 # custom_stopwords (60 terms) + tag_map
│   └── utils/
│       └── Utils.py                  # Dataset file management
│
├── models/
│   ├── topic_modelling/
│   │   ├── lda_best_bow.pkl          # Best LDA model on BoW corpus
│   │   ├── lda_best_tfidf.pkl        # Best LDA model on TF-IDF corpus
│   │   ├── bow_vectorizer.pkl        # Fitted CountVectorizer
│   │   ├── tfidf_vectorizer.pkl      # Fitted TfidfVectorizer
│   │   ├── bert_train_topics.pkl     # BERTopic assignments (train)
│   │   ├── bert_test_topics.pkl      # BERTopic assignments (test)
│   │   ├── eval_results_bow.pkl      # LDA evaluation results (BoW)
│   │   ├── eval_results_tfidf.pkl    # LDA evaluation results (TF-IDF)
│   │   └── model_registry.pkl        # Model registry
│   ├── clustering/
│   │   └── clustering_labels.pkl     # Final cluster label assignments
│   ├── vader_sentiment_analyzer.pkl  # Fitted VADER analyser
│   └── textblob_config.json          # TextBlob thresholds + metadata
│
├── data/
│   ├── raw/                          # Original, untouched dataset
│   ├── processed/                    # Cleaned & preprocessed data
│   └── samples/                      # Subsets for development testing
│
└── reports/
    ├── figures/                      # Saved plots (300 DPI PNG)
    └── tables/                       # Evaluation result tables
```

---

## Setup & Quickstart

```bash
git clone https://github.com/fahmi-alshahabi/EMATM0067_2025_TB-2-g43
cd EMATM0067_2025_TB-2-g43
pip install -r requirements.txt

# Download required NLTK resources (auto-handled in Preprocessing.py)
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); \
           nltk.download('averaged_perceptron_tagger_eng')"

# Download spaCy model (if used in EDA)
python -m spacy download en_core_web_sm
```

Then open any notebook in sequence:

```bash
jupyter notebook notebooks/01_eda/EDA-ThiruvelAP.ipynb
```

Or run the full pipeline in Google Colab *(no local setup required)*:
> [Open Colab Notebook](https://colab.research.google.com/drive/1585pIeWTvfiePhdiELvrqmmkzva1ytNk?usp=sharing)

---

## Dependencies

| Package | Role |
|---|---|
| `scikit-learn` | KMeans, HAC, TF-IDF, CountVectorizer, Silhouette/ARI/NMI, PCA |
| `gensim` | LDA, Word2Vec Skip-gram (word2vec-google-news-300), Sparse2Corpus |
| `sentence-transformers` | SBERT  `all-MiniLM-L6-v2` sentence embeddings |
| `bertopic` | BERTopic neural topic modelling |
| `vaderSentiment` | Rule-based sentiment analysis |
| `textblob` | PatternAnalyzer polarity scoring |
| `nltk` | Tokenisation, POS tagging, WordNetLemmatizer |
| `spaCy` | NLP utilities (EDA phase) |
| `contractions` | Contraction expansion before noise removal |
| `transformers` | Transformer backbone for BERTopic |
| `datasets` / `kagglehub` | Dataset loading utilities |
| `matplotlib` / `seaborn` / `plotly` | Visualisation (all saved at 300 DPI) |

---

## Sector Applications

| Sector | Application |
|---|---|
| **Technology / NLP Engineering** | Production NLP pipeline template  preprocessing, multi-representation vectorisation, unsupervised clustering, topic extraction; directly applicable to enterprise feedback analytics, search, and content moderation |
| **Healthcare & Omics Research** | Methodology transferable to patient-reported outcome mining, clinical note clustering, and adverse event signal extraction from free-text EHR fields |
| **Finance & Analytics** | Customer complaint analytics for FCA compliance; unsupervised topic discovery from earnings call transcripts, regulatory filings, or support ticket routing |

---

## Team  Group G43

| Member | Individual Contribution |
|---|---|
| **Thiruvel Andagurunathan Pandian** | EDA (`EDA-ThiruvelAP.ipynb`) · [Individual Colab Notebook](https://colab.research.google.com/drive/1585pIeWTvfiePhdiELvrqmmkzva1ytNk?usp=sharing) |
| Fahmi Al-Shahabi | EDA (`EDA-Fahmi.ipynb`) |
| Ukhash | EDA (`EDA-Ukhash.ipynb`) |
| Xiang | EDA (`EDA-Xiang.ipynb`) |
| Zhaoqi | EDA (`EDA-Zhaoqi.ipynb`) |

---

## Author (Individual Contribution)

**Thiruvel Andagurunathan Pandian**  MSc Data Science, University of Bristol  
Applying NLP and unsupervised ML to extract actionable insights from unstructured text at scale.  
📍 Bristol, UK · **Eligible for Skilled Worker Visa sponsorship** · Open to UK roles

[![LinkedIn](https://img.shields.io/badge/LinkedIn-%230077B5.svg?logo=linkedin&logoColor=white)](https://linkedin.com/in/Thiruvel-AP)
[![GitHub](https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white)](https://github.com/Thiruvel-AP)
