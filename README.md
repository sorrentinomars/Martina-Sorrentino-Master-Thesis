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
- [Topic Map r/Braincels](Figures/Braincels_UMAP_projection.html)
- [Topic Map Incels.is](Figures/Incelsis_UMAP_distribution.html)
- [SIT Distribution r/Braincels](Figures/Braincels_SIT_distribution.html)
- [SIT Distribution Incels.is](Figures/Incelsis_SIT_distribution.html)
- [Emotion stacked bars r/Braincels](Figures/Braincels_emotion_.html)
- [Emotion stacked bars Incels.is](Figures/Incelsis_emotion.html)
- [Lexicon total hits per topic r/Braincels](Figures/Braincels_lexicon_totalhits.html)
- [Lexicon total hits per topic Incels.is](Figures/Incelsis_lexicon_totalhits.html)
- [Lexicon normalised scores per topic r/Braincels](Figures/Braincels_lexicon_normalised.html)
- [Lexicon normalised scores per topic Incels.is](Figures/Incelsis_lexicon_normalised.html)

To view them:
1. Download the `.html` file from the `Figures/` folder.
2. Open the file locally in a web browser.

## Lexicon Analysis Dictionary

The lexicon dictionary terms used in this analysis are drawn from the **Incel Violent Extremism Dictionary (IVEC)** (Baele et al., 2023). 

If you use this data or code for subsequent analyses, please cite the original paper:

> Baele, S., Brace, L., & Ging, D. (2023). A Diachronic Cross-Platforms Analysis of Violent Extremist Language in the Incel Online Ecosystem. *Terrorism and Political Violence*, 36(3), 382–405. https://doi.org/10.1080/09546553.2022.2161373

## Requirements

Install the required Python packages from the requirements file and then download the spaCy language model dependencies:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm

