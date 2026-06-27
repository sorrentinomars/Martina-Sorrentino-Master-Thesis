"""
BERTopic modelling.
Workflow
--------
1. Load preprocessed corpus and create and save embeddings.
2. Reduce dimensionality using UMAP.
3. Cluster documents using HDBSCAN.
4. Generate topic representations using BERTopic.
5. Reassign outliers using embedding similarity.
6. Update topic representations.
7. Save final topic model.

Author: Martina Sorrentino
Thesis: "Coping in the Dark: Blackpill Ideology and Social Identity Management in Online Incel Communities"
"""

import re
import joblib
import numpy as np
import pandas as pd
import spacy
import umap
from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance
from hdbscan import HDBSCAN
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer


### LOAD DATA

# ------------------------------------------------------------------
# NOTE:
# The original dataset and trained BERTopic model are not included
# in this repository due to ethical/privacy restrictions.
# Replace the paths below with local copies.
# ------------------------------------------------------------------

df = pd.read_csv(
    "YOUR_DATA_PATH.tsv",
    sep="\t"
)

# Documents were already preprocessed and aggregated at the thread level

df["document"] = df["document"].fillna("").astype(str)
docs = df["document"].tolist()
print(f"Number of documents: {len(docs)}")

# CREATE AND SAVE EMBEDDINGS
embedding_model = SentenceTransformer('all-mpnet-base-v2')
embeddings = embedding_model.encode(
    df["document"].tolist(), 
    show_progress_bar=True 
) 
embeddings = np.array(embeddings)
joblib.dump(embeddings, "embeddings.pkl")
embeddings = embeddings[df.index]


### TOPIC MODEL

# UMAP
umap_model = umap.UMAP(
    n_neighbors=5,
    n_components=20,
    min_dist=0.1,
    metric='cosine',
    random_state=42,
    n_jobs=-1
)

# HDBSCAN 
hdbscan_model = HDBSCAN(
    min_cluster_size=80,
    min_samples=20,
    cluster_selection_method='leaf',
    prediction_data=True
)

# VECTORIZER AND STOPWORDS
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
nlp.max_length = 100_000_000
nltk_stopwords = set(nltk.corpus.stopwords.words('english'))
spacy_stopwords = set(nlp.Defaults.stop_words)
custom_stopwords = nltk_stopwords.union(spacy_stopwords, {
    "just", "", "dont", "like", "im", "u", ",", "///", 
    "youre", "thats", "yes", "no"
}) 

vectorizer_model = CountVectorizer(
    stop_words=list(custom_stopwords),
    min_df=5,
    ngram_range=(1, 2),
    max_features=2000
)

# REPRESENTATION MODEL
representation_model=MaximalMarginalRelevance(diversity=0.5)

topic_model = BERTopic(embedding_model=None,  
                       umap_model=umap_model,  # Usa il modello UMAP per la riduzione
                       hdbscan_model=hdbscan_model, 
                       vectorizer_model = vectorizer_model, 
                       representation_model=representation_model,
                       verbose=True,
                       top_n_words=15,
                       calculate_probabilities=True
)

topics, _ = topic_model.fit_transform(docs, embeddings) #topic model on previewsly computed embeddings and cleaned corpus

num_topics=len(topic_model.get_topic_info())
print({num_topics})
topic_info = topic_model.get_topic_info()
print(topic_info.head(20))

# SAVING THE MODEL BEFORE OUTLIER REDUCTION
joblib.dump(
    topic_model,
    r"Topic_model_outliers.pkl"
)


### OUTLIER REDUCTION

embedding_model = SentenceTransformer("all-mpnet-base-v2") 
topic_model.embedding_model = embedding_model
thresholds = [0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
topics = topic_model.topics_

for t in thresholds:
    new_t = topic_model.reduce_outliers(
        docs,
        topics,
        strategy="embeddings",
        threshold=t,
        embeddings=embeddings
    )
    n_outliers = (np.array(new_t) == -1).sum()
    n_assigned = (np.array(topics) == -1).sum() - n_outliers
    
    print(f"threshold={t:.2f} → outlier={n_outliers:5d} | assegnati={n_assigned:5d}")


# REDUCE OUTLIERS 
class SimpleEmbedder:
    def __init__(self, model):
        self.embedding_model = model
    
    def embed_documents(self, documents, verbose=False):
        return self.embedding_model.encode(documents, show_progress_bar=verbose)
    
    def embed(self, documents, verbose=False):
        return self.embedding_model.encode(documents, show_progress_bar=verbose)

model = SentenceTransformer("all-mpnet-base-v2")

topic_model.embedding_model = SimpleEmbedder(model)

new_topics = topic_model.reduce_outliers(
    docs,
    topics,
    strategy="embeddings",
    threshold=0.4,
    embeddings=embeddings
)

#USE AGAIN VECTORISER TO EXCLUDE STOPWORDS FROM UPDATED MODEL
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
nlp.max_length = 100_000_000
nltk_stopwords = set(nltk.corpus.stopwords.words('english'))
spacy_stopwords = set(nlp.Defaults.stop_words)
custom_stopwords = nltk_stopwords.union(spacy_stopwords, {
    "just", "", "dont", "like", "im", "u", ",", "///", 
    "youre", "thats", "yes", "no"
}) 
vectorizer_model = CountVectorizer(
    stop_words=list(custom_stopwords),
    min_df=5,
    ngram_range=(1, 2),
    max_features=2000
)

# FINAL TOPIC ASSIGNMENTS AFTER OUTLIER REDUCTION
topic_model.update_topics(
    docs,
    topics=new_topics,
    vectorizer_model=vectorizer_model
)

topic_model.topics_ = new_topics

df["topic"] = new_topics

grouped_filtered = (
    df[df["topic"] != -1]
    .reset_index(drop=True)
)

# SAVE UPDATED MODEL
joblib.dump(
    topic_model,
 "Topic_model_no_outliers.pkl"
)

print("Analysis completed.")
