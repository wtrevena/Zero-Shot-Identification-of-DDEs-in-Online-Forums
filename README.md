# Utilizing Open Source Language Models and ChatGPT for Zero-Shot Identification of Drug Discontinuation Events (DDEs) in Online Forums

This project provides tools for classifying Drug Discontinuation Events (DDEs) using various models, including HuggingFace transformer models and GPT models. 

## Prerequisites

The project has been tested with Python 3.9.


## Setting Up

Installation of the necessary dependencies can be achieved with:

```bash
pip install -r requirements.txt
```

## Data

In order to utilize the code in this repository, you first need to download the datasets from the [Dryad repository]{https://doi.org/10.5061/dryad.h44j0zps2}, place them in a folder called `data` in the same repository as the scripts, and then run the `scrape_ground_truth_posts.py` script to generate the `ground_truth.csv.gz` file. In order to construct the `HuggingFace_model_predictions.csv.gz` dataframe which contains all of the classifications from the HuggingFace models, you will need to run the `scrape_all_classified_posts.py` script. 

To ensure successful execution, the `ground_truth.csv.gz` file should be in the `data` folder after running the `scrape_ground_truth_posts.py` script. The `HuggingFace_model_predictions.csv.gz` file should be in the `data` folder after running the `scrape_all_classified_posts.py` script. 

Note that since posts on medhelp.org can be deleted, there may be posts in the `ground_truth.csv.gz` file that are no longer available. Similarly, there may be posts in the `HuggingFace_model_predictions.csv.gz` file that are no longer available. 


## Repo Structure

The `DDE_detection` directory contains scripts for classifying DDEs using HuggingFace transformer models and GPT models. The `DDE_root_cause_classification` directory contains scripts for classifying the root cause(s) of DDEs using HuggingFace transformer models and GPT models.

