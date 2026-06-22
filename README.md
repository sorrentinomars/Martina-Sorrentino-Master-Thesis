# Computational Analysis of Incels.is and r/Braincels archival posts
This repository contains the code and visual outputs developed as part of my Master's thesis project.

The project applies topic modelling, lexicon-based analysis, and emotion analysis to study online discussions in the forum Incels.is and subreddit r/Braincels using BERTopic and related NLP methods.

## Repository Contents

- Python scripts for:
  - BERTopic modelling
  - lexicon-based analysis
  - emotion analysis (j-hartmann/emotion-english-distilroberta-base)

- Generated plots and visualisations from the analysis pipeline. Visual outputs are available in the figures folder.

## Data and Models

The original datasets and trained machine learning models are not included in this repository due to privacy and ethical restrictions related to the source data.

The code documents the methodology and allows reproduction of the analysis pipeline with appropriate access to the required data and trained models.

## Interactive visualisations

The repository includes interactive Plotly/BERTopic visualisations as HTML files.

To view them:
1. Download the `.html` file from the `figures/` folder.
2. Open the file locally in a web browser.

The visualisations are interactive and allow zooming, hovering over topics, and exploring the embedding space.
## Requirements

Install the required Python packages, after installing the requirements, download the spaCy language model:

```bash
pip install -r requirements.txt

```bash
python -m spacy download en_core_web_sm
