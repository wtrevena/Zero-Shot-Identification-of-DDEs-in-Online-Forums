import argparse
from ast import literal_eval
import logging
import time

import torch
import pandas as pd
from tqdm import tqdm
from transformers import pipeline
from transformers import AutoTokenizer

MODELS = {
    "bart": {
        "model_name": "facebook/bart-large-mnli",
        "max_tokens": 1024,
    },
    "deberta": {
        "model_name": "cross-encoder/nli-deberta-base",
        "max_tokens": 512,
    },
    # "distilbert": {
    #     "model_name": "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    #     "max_tokens": 512,
    # }
}

logger = logging.getLogger(__name__)
logging.basicConfig(format="[%(filename)s:%(lineno)s] %(message)s")
logger.setLevel(logging.INFO)


def check_gpu():
    if torch.cuda.is_available():
        return 0
    return -1


def parse_args():
    parser = argparse.ArgumentParser(prog='HuggingFace_classifier',
                                     description="Script to classify Drug Discontinuation Events (DDE) using different transformer models.")

    # Model argument
    parser.add_argument("model", 
                        choices=list(MODELS), 
                        help="Select the transformers model for DDE classification.")

    # Cutoff argument
    parser.add_argument("--cutoff", 
                        type=float, 
                        default=0.95, 
                        help="Set the prediction threshold for a Drug Discontinuation Event (DDE).  Values above this threshold indicate a positive prediction. (default: 0.95)")

    # Classification Strategy argument
    parser.add_argument("--strategy", 
                        type=int, 
                        choices=[1, 2], 
                        default=1, 
                        metavar="Classification Strategy (CS)",
                        help="Choose a classification strategy for the DDE. (default: 1)")

    args = parser.parse_args()
    return args


def run_model():
    args = parse_args()
    model_name = MODELS[args.model]["model_name"]
    max_tokens = MODELS[args.model]["max_tokens"]
    strategy = args.strategy
    logger.info(f"Parameters\n Model Name: {model_name} \n Strategy: {strategy}")

    logger.info("Initializing model")
    classifier = pipeline(
        "zero-shot-classification", model=model_name, device=check_gpu()
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Updated candidate labels and hypothesis templates
    hypothesis_template = {
        "Treatment Success": "The comment indicates that the medication or treatment was stopped after successfully achieving the intended health outcomes.",
        "Treatment Inefficacy": "The comment indicates that the medication or treatment was stopped due to ineffectiveness.",
        "Adverse Reactions": "The comment indicates that the medication or treatment was stopped due to adverse side effects, allergic reactions, or harmful interactions with other medications.",
        "Accessibility Issues": "The comment indicates that the medication or treatment was stopped due to accessibility issues such as a lack of insurance, financial constraints, a lack of market availability, or prescriber decisions.",
        "Personal Choices": "The comment indicates that the medication or treatment was stopped based on personal choices, influenced by personal beliefs, lifestyle changes, or non-adherence to medical advice.",
        "Alternative Medical Reasons": "The comment indicates that the medication or treatment was stopped for specific medical reasons not covered by other categories, such as pregnancy or a new health condition.",
        "Indeterminate": "It is unclear why the medication or treatment was stopped.",
        "Non-Discontinuation": "The comment does not indicate that medication or treatment was stopped."
    }
    candidate_labels = list(hypothesis_template.keys())

    logger.info("Reading input data")
    data = pd.read_csv("./DDE_root_cause_classification/data/ground_truth.csv.gz", compression="gzip")
    # data = data.head(2)
    logger.info(f"Row count: {len(data)}")

    output_columns = ["url", "text", "text_sentences"]
    output_columns.extend(
        [
            f"{args.model}_CS{args.strategy}_single_label_predictions",
            f"{args.model}_CS{args.strategy}_multi_label_predictions"
        ]
    )

    processed_data = pd.DataFrame(index=range(0, data.shape[0]), columns=output_columns)
    processed_data["url"] = data["url"]
    processed_data["text"] = data["text"]
    processed_data["text_sentences"] = data["text_sentences"]

    logger.info(f"Starting classification strategy {args.strategy}")
    for idx, row in tqdm(data.iterrows(), total=data.shape[0]):
        sentences = literal_eval(row["text_sentences"])
        hypotheses = [hypothesis_template[label] for label in candidate_labels]

        # Single-label classification
        single_label_results = classifier(sentences, candidate_labels, hypothesis_template="{}", multi_label=False)
        processed_data.at[idx, f"{args.model}_CS{args.strategy}_single_label_predictions"] = single_label_results
        # [
        #     result["scores"][0] for result in single_label_results
        # ]

        # Multi-label classification
        multi_label_results = classifier(sentences, candidate_labels, hypothesis=hypotheses, multi_label=True)
        processed_data.at[idx, f"{args.model}_CS{args.strategy}_multi_label_predictions"] = multi_label_results
        # [
        #     result["scores"][0] for result in multi_label_results
        # ]

    logger.info("Classification finished. Preparing results")
    output_path = f"DDE_root_cause_classification/outputs/{args.model}_cs{args.strategy}_multi_label_classified.csv"

    logger.info(f"Writing output file to {output_path}")
    processed_data.to_csv(output_path, index=False)


if __name__ == "__main__":
    start_time = time.time()
    run_model()
    logger.info(f"Finished in {round(time.time()-start_time, 2)} seconds")
