"""
Lexicon-based analysis of BERTopic topics.

Workflow:
1. Load BERTopic model and discussion corpus.
2. Assign topic labels.
3. Count occurrences of lexicon terms.
4. Normalize counts by document length.
5. Compute corpus-level statistics.
6. Compute topic-level statistics.
7. Export results.

Author: Martina Sorrentino
Thesis: "Coping in the Dark: Blackpill Ideology and Social Identity Management in Online Incel Communities"
"""

import re
import joblib
import pandas as pd


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


### LEXICON ANALYSIS

COL_NAMES = {
    "Dehumanisation of Outgroups": "dehumanisation",
    "Acts of Violence": "violence",
    "Weapons References": "weapons",
}

custom_emotion_lexicon = {
    "Dehumanisation of Outgroups": [
        "foids", "chad", "whale", "foid", "stacey", "whore",
        "normies", "soyboy", "bitch", "normalfags", "nigger",
        "normie", "noodlewhores", "curry", "thots", "faggot",
        "tallfags", "cunt", "thot", "slut", "gook", "nigga",
        "subhuman", "beast", "femoids", "tallfag", "fag",
        "bluepillers", "roastie", "soyboys", "roasties",
        "stacy", "wagecuck", "femoid", "chink", "tranny",
        "shitskin", "rice", "alpha", "gorilla", "trannies",
        "chadlite", "normalfag", "creature", "parasite",
        "landwhale", "dumpster", "subhumans", "simps",
        "simp", "numale", "stacies", "hoe", "betabux",
        "negro", "scum", "gigachads", "monster", "beckies",
        "pig", "rat", "kike", "gigachad", "npc", "betabuxx",
        "ricecels", "chadlites", "becky", "tyrones", "stacys"
    ],

    "Acts of Violence": [
        "kill", "rape", "hit", "killed", "attack", "killing",
        "raped", "slay", "destroy", "destroyed", "murder",
        "shooting", "shoot", "kick", "pound", "torture",
        "hitting", "abused", "punch", "kicked", "beaten",
        "slaying", "nuke", "cutting", "smash", "raping",
        "genocide", "attacked", "attacking", "slap",
        "murdered", "uprising", "poison", "punching",
        "slays", "wound", "tortured", "shove", "assault",
        "grab"
    ],

    "Weapons References": [
        "gun", "acid", "weapon", "bullet",
        "knife", "bomb", "poison", "shotgun"
    ]
}

# LEXICON COUNTS
for category, words in custom_emotion_lexicon.items():

    col = COL_NAMES[category]

    pattern = re.compile(
        r"\b(" + "|".join(map(re.escape, set(words))) + r")\b",
        flags=re.IGNORECASE
    )

    grouped[col] = grouped["document"].apply(
        lambda text, p=pattern: len(p.findall(text))
    )


# NORMALIZATION
grouped["doc_word_count"] = (
    grouped["document"]
    .str.split()
    .str.len()
)

for _, col in COL_NAMES.items():

    grouped[f"{col}_normalized"] = (
        grouped[col]
        / grouped["doc_word_count"]
        * 100
    ).round(4)


# OVERALL CORPUS STATISTICS
print("\nCorpus statistics")

total_docs = len(grouped)

for category, col in COL_NAMES.items():

    n_docs = (grouped[col] > 0).sum()
    total_hits = grouped[col].sum()

    print(
        f"{category}: "
        f"{n_docs} docs ({n_docs / total_docs * 100:.2f}%), "
        f"{total_hits} total hits"
    )

neutral_docs = (
    (
        (grouped["dehumanisation"] == 0)
        & (grouped["violence"] == 0)
        & (grouped["weapons"] == 0)
    )
).sum()

print(
    f"Neutral documents: "
    f"{neutral_docs} "
    f"({neutral_docs / total_docs * 100:.2f}%)"
)


### TOPIC-LEVEL AGGREGATION

topic_lexicon = (
    grouped.groupby(["topic", "topic_label"])
    .agg(
        doc_count=("document", "count"),

        total_dehumanisation=("dehumanisation", "sum"),
        total_violence=("violence", "sum"),
        total_weapons=("weapons", "sum"),

        avg_dehumanisation=("dehumanisation", "mean"),
        avg_violence=("violence", "mean"),
        avg_weapons=("weapons", "mean"),

        avg_dehumanisation_normalized=(
            "dehumanisation_normalized",
            "mean"
        ),
        avg_violence_normalized=(
            "violence_normalized",
            "mean"
        ),
        avg_weapons_normalized=(
            "weapons_normalized",
            "mean"
        ),

        pct_docs_dehumanisation=(
            "dehumanisation",
            lambda x: (x > 0).mean() * 100
        ),

        pct_docs_violence=(
            "violence",
            lambda x: (x > 0).mean() * 100
        ),

        pct_docs_weapons=(
            "weapons",
            lambda x: (x > 0).mean() * 100
        ),

        pct_docs_neutral=(
            "dehumanisation",
            lambda x: (
                (
                    (x == 0)
                    & (grouped.loc[x.index, "violence"] == 0)
                    & (grouped.loc[x.index, "weapons"] == 0)
                ).mean()
                * 100
            )
        ),
    )
    .reset_index()
)

# EXPORT RESULTS
grouped[
    [
        "thread_id",
        "document",
        "topic",
        "topic_label",
        "dehumanisation",
        "violence",
        "weapons",
        "doc_word_count",
        "dehumanisation_normalized",
        "violence_normalized",
        "weapons_normalized",
    ]
].to_csv(
    "Incelis_lexicon_per_document.csv",
    index=False
)

topic_lexicon.to_csv(
    "Incelis_lexicon_per_topic.csv",
    index=False
)

print("Analysis completed.")

