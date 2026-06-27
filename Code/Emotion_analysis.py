"""
Emotion analysis of BERTopic topics.

Workflow
--------
1. Load preprocessed corpus.
2. Load trained BERTopic model.
3. Assign topic labels.
4. Remove outlier and irrelevant topics.
5. Classify document emotions using Hartmann emotion classifier.
6. Aggregate emotions by topic.
7. Export results.

Author: Martina Sorrentino
Thesis: "Coping in the Dark: Blackpill Ideology and Social Identity Management in Online Incel Communities"
"""

import joblib
import pandas as pd
from tqdm import tqdm
from transformers import pipeline
from collections import defaultdict


### LOAD MODEL AND DATA

# ------------------------------------------------------------------
# NOTE:
# The original dataset and trained BERTopic model are not included
# in this repository due to ethical/privacy restrictions.
# Replace the paths below with local copies.
# ------------------------------------------------------------------

topic_model = joblib.load("YOUR_MODEL_PATH.pkl")

df = pd.read_csv(
    "YOUR_DATA_PATH.tsv",
    sep="\t"
)
# Documents were already preprocessed and aggregated at the thread level
df["document"] = df["document"].fillna("").astype(str)

# TOPIC ASSIGNMENTS
df["topic"] = topic_model.topics_
grouped = df[df["topic"] != -1].copy() #REMOVE OUTLIERS

topic_info = topic_model.get_topic_info()

label_column = (
    "CustomName"
    if "CustomName" in topic_info.columns #LOAD CUSTOM TOPIC LABELS IF PRESENT
    else "Name"
)

labels_dict = dict(
    zip(
        topic_info["Topic"],
        topic_info[label_column]
    )
)

grouped["topic_label"] = grouped["topic"].map(labels_dict)


### EMOTION CLASSIFICATION

emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    truncation=True,
    max_length=512
)

def classify_long_text(text, classifier, max_length=512):
    words = text.split()
    chunks = [" ".join(words[i:i+350]) for i in range(0, len(words), 350)]
    
    results = classifier(chunks, truncation=True, max_length=max_length)
    scores = defaultdict(float)
    for r in results:
        scores[r['label']] += r['score']
    
    top_label = max(scores, key=scores.get)
    top_score = scores[top_label] / len(results)  
    return top_label, top_score

def get_emotion_jhartmann(text):
    try:
        return classify_long_text(text, emotion_classifier)
    except:
        return 'neutral', 0.0

tqdm.pandas()

grouped[["emotion_jhartmann", "emotion_jhartmann_score"]] = grouped["document"].progress_apply(
    lambda x: pd.Series(get_emotion_jhartmann(x))
)

# SAVE DOCUMENT LEVEL OUTPUT
grouped[
    [
        "thread_id",
        "document",
        "topic",
        "topic_label",
        "emotion_jhartmann",
        "emotion_jhartmann_score"
    ]
].to_csv(
    "emotion_per_document.csv",
    index=False
)


###  OVERALL EMOTION DISTRIBUTION

overall_emotion = (
    grouped.groupby("emotion_jhartmann")
    .agg(
        count=("emotion_jhartmann", "count"),
        avg_score=("emotion_jhartmann_score", "mean")
    )
    .reset_index()
)

overall_emotion["percentage"] = (
    overall_emotion["count"]
    /
    len(grouped)
    *
    100
).round(2)


overall_emotion.to_csv(
    "overall_emotion_distribution.csv",
    index=False
)


### TOPIC LEVEL EMOTION ANALYSIS

topic_emotion = (
    grouped.groupby(
        [
            "topic",
            "topic_label",
            "emotion_jhartmann"
        ]
    )
    .agg(
        count=("emotion_jhartmann", "count"),
        avg_score=("emotion_jhartmann_score", "mean")
    )
    .reset_index()
)

# Dominant emotion per topic
dominant_emotion = (
    topic_emotion
    .sort_values(
        "count",
        ascending=False
    )
    .groupby("topic")
    .first()
    .reset_index()
.rename(
    columns={
        "emotion_jhartmann":
            "dominant_emotion",
        "avg_score":
            "dominant_emotion_score"
    }
    )
)

# Full emotion distribution per topic
topic_emotion_breakdown = (
    topic_emotion
    .pivot_table(
        index=[
            "topic",
            "topic_label"
        ],
        columns="emotion_jhartmann",
        values="count",
        fill_value=0
    )
    .reset_index()
)

# EXPORT RESULTS
dominant_emotion.to_csv(
    "topic_dominant_emotion.csv",
    index=False
)

topic_emotion_breakdown.to_csv(
    "topic_emotion_breakdown.csv",
    index=False
)

print("Emotion analysis completed.")
