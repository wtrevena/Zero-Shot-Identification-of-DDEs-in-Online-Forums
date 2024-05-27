# Zero-Shot Identification of The Root Cause(s) of Drug Discontinuation Events (DDEs) in Online Forums


## Prerequisites

The project has been tested with Python 3.9.


## Setting Up

Installation of the necessary dependencies can be achieved with:

```bash
pip install -r requirements.txt
```

## Data

(Mandatory) In order to utilize the code in this repository, you first need to download the datasets `DDE_root_cause_analysis_ground_truth_model_predictions_URLs_only.csv.gz` and `DDE_root_cause_analysis_ground_truth_URLs_only.csv.gz` from the [Dryad repository]{https://doi.org/10.5061/dryad.h44j0zps2}, place them in a folder called `data` in this folder, and then run the `scrape_ground_truth_posts.py` script to generate the `ground_truth.csv.gz` file. To ensure successful execution, the `ground_truth.csv.gz` file should be in the `data` folder after running the `scrape_ground_truth_posts.py` script. Note that since posts on medhelp.org can be deleted, there may be posts in the `ground_truth.csv.gz` file that are no longer available.

`python DDE_root_cause_classification/scrape_DDE_root_cause_analysis_ground_truth_posts.py`


## Script Usage

After executing either of the scripts, an output file is generated in the same directory. The naming convention for this file reflects the chosen model, strategy, and cutoff values. The file naming structure is as follows:

```
<model_name>_cs<classification_strategy>_cutoff<cutoff_value>_medhelp_classified.csv
```

For example, 

```
python DDE_root_cause_classification/HuggingFace_classifier.py --cutoff 0.9 --strategy 2 bart
```

will produce an output file named `bart_cs2_cutoff0.9_medhelp_classified.csv` in the `DDE_root_cause_classification/outputs` directory.

The `-h` or `--help` options are also available with each script to display a detailed explanation of the arguments and descriptions on the command line interface.

```bash
python DDE_root_cause_classification/HuggingFace_classifier.py -h
```

### HuggingFace Transformer Models

For `HuggingFace_classifier.py`:

```bash
python DDE_root_cause_classification/HuggingFace_classifier.py <model_name> [--cutoff <cutoff_value>] [--strategy <strategy_number>]
```

- `<model_name>`: Models to choose from include `bart` and `deberta`.
- `--cutoff`: (Optional) Specifies the prediction threshold for a Drug Discontinuation Event (DDE). Defaults to `0.95`.
- `--strategy`: (Optional) Dictates the classification strategy for the DDE. Choices are `1` and `2`, with `1` being the default.

### GPT Models

Before using `GPT_classifier.py`, modification of the `OPENAI_API_KEY` variable at the top of the script with a personal API key is necessary.

For `GPT_classifier.py`:

```bash
python DDE_root_cause_classification/GPT_classifier.py <model_name> 
```

- `<model_name>`: The available models are `gpt-4o-2024-05-13` and `gpt-4-0125-preview`.
