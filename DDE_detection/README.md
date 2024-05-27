# Zero-Shot Identification of Drug Discontinuation Events (DDEs) in Online Forums

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


## Script Usage

After executing either of the scripts, an output file is generated in the same directory. The naming convention for this file reflects the chosen model, strategy, and cutoff values. The file naming structure is as follows:

```
<model_name>_cs<classification_strategy>_cutoff<cutoff_value>_medhelp_classified.csv
```

For example, 

```
python DDE_detection/HuggingFace_classifier.py --cutoff 0.9 --strategy 2 distilbert
```

will produce an output file named `distilbert_cs2_cutoff0.9_medhelp_classified.csv` in the `DDE_detection` directory.

The `-h` or `--help` options are also available with each script to display a detailed explanation of the arguments and descriptions on the command line interface.

```bash
python HuggingFace_classifier.py -h
```

### HuggingFace Transformer Models

For `HuggingFace_classifier.py`:

```bash
python HuggingFace_classifier.py <model_name> [--cutoff <cutoff_value>] [--strategy <strategy_number>]
```

- `<model_name>`: Models to choose from include `bart`, `deberta`, `distilbert`, `distilroberta`, and `roberta`.
- `--cutoff`: (Optional) Specifies the prediction threshold for a Drug Discontinuation Event (DDE). Defaults to `0.95`.
- `--strategy`: (Optional) Dictates the classification strategy for the DDE. Choices are `1` and `2`, with `1` being the default.

### GPT Models

Before using `ChatGPT_classifier.py`, modification of the `OPENAI_API_KEY` variable at the top of the script with a personal API key is necessary.

For `ChatGPT_classifier.py`:

```bash
python DDE_detection/ChatGPT_classifier.py <model_name> [--cutoff <cutoff_value>] [--strategy <strategy_number>]
```

- `<model_name>`: The available models are `gpt-4` and `gpt-3.5-turbo`.
- `--cutoff`: (Optional) Sets the prediction threshold for a Drug Discontinuation Event (DDE). Defaults to `0.95`.
- `--strategy`: (Optional) Selection of the classification strategy for the DDE. Options are `1`, `2`, and `3`, with the default set to `1`.

### Calculating and Verifying Metrics

To evaluate the model and classification method metrics — accuracy, false positive rate, false negative rate, and F1 score — run the verify_metrics.py script:

```bash
python DDE_detection/verify_metrics.py
```

Upon completion, a file named `metrics_df_verification.csv` will be generated in the DDE_detection directory, containing these metrics for further analysis. Note that these metrics may change over time as comments that have been deleted from medhelp.org are dropped from the dataset during the scraping process. As a result, there will be a slight misalignment between the metrics in the `metrics_df_verification.csv` file and the metrics calculated and presented in the paper associated with this work. As of 5/27/2024, 11 out of the 1000 original comments in the dataset have been deleted from medhelp.org, and as a result, the metrics in the `metrics_df_verification.csv` file will be slightly different than the metrics presented in the paper.
