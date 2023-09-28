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
    "distilbert": {
        "model_name": "typeform/distilbert-base-uncased-mnli",
        "max_tokens": 512,
    },
    "distilroberta": {
        "model_name": "cross-encoder/nli-distilroberta-base",
        "max_tokens": 514,
    },
    "roberta": {
        "model_name": "roberta-large-mnli",
        "max_tokens": 514,
    },
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

    candidate_labels = ["Person stopped taking medication"]

    logger.info("Reading input data")

    data = pd.read_csv("ground_truth.csv.gz", compression="gzip")
    logger.info(f"Row count: {len(data)}")

    output_columns = ["url", "text", "text_sentences"]
    if args.strategy == 1:
        output_columns.extend(
            [
                f"{args.model}_CS1_predictions",
                f"{args.model}_CS1_max_prediction",
            ]
        )
    elif args.strategy == 2:
        output_columns.extend(
            [
                f"{args.model}_CS2_text_sentences_groups",
                f"{args.model}_CS2_text_sentences_groups_token_counts",
                f"{args.model}_CS2_text_sentences_groups_predictions",
                f"{args.model}_CS2_text_sentences_groups_max_prediction",
            ]
        )
    else:
        raise Exception("Strategy not found")

    processed_data = pd.DataFrame(index=range(0, data.shape[0]), columns=output_columns)
    processed_data["url"] = data["url"]
    processed_data["text"] = data["text"]
    processed_data["text_sentences"] = data["text_sentences"]

    logger.info(f"Starting classification strategy {args.strategy}")
    for idx, row in tqdm(data.iterrows(), total=data.shape[0]):
        sentences = literal_eval(row["text_sentences"])

        if args.strategy == 1:
            results = classifier(sentences, candidate_labels, hypothesis_template="{}")
            maxx = max([result["scores"][0] for result in results])
            processed_data.at[idx, f"{args.model}_CS1_predictions"] = [
                result["scores"][0] for result in results
            ]
            processed_data.at[idx, f"{args.model}_CS1_max_prediction"] = maxx
            processed_data.at[idx, f"{args.model}_CS1_cutoff_{str(args.cutoff)}"] = (
                1 if maxx > args.cutoff else 0
            )

        elif args.strategy == 2:
            counter = 0
            sentence_groups = []
            group_token_count = []
            current_sentence_group = ""
            encoded_inputs = tokenizer(sentences)
            for iii in range(0, len(encoded_inputs.input_ids)):
                if counter + len(encoded_inputs.input_ids[iii]) <= max_tokens:
                    current_sentence_group = (
                        current_sentence_group + " " + sentences[iii]
                    )
                    counter = counter + len(encoded_inputs.input_ids[iii])
                    if iii == (len(encoded_inputs.input_ids) - 1):
                        sentence_groups.append(current_sentence_group)
                        group_token_count.append(counter)
                        current_sentence_group = ""
                        counter = 0
                elif (counter == 0) & (len(encoded_inputs.input_ids[iii]) > max_tokens):
                    sentence_groups.append(sentences[iii])
                    group_token_count.append(len(encoded_inputs.input_ids[iii]))
                else:
                    sentence_groups.append(current_sentence_group)
                    current_sentence_group = ""
                    group_token_count.append(counter)
                    counter = 0
            results = classifier(
                sentence_groups, candidate_labels, hypothesis_template="{}"
            )
            maxx = max([result["scores"][0] for result in results])
            processed_data.at[
                idx, f"{args.model}_CS2_text_sentences_groups_predictions"
            ] = [result["scores"][0] for result in results]
            processed_data.at[
                idx, f"{args.model}_CS2_text_sentences_groups_max_prediction"
            ] = maxx
            processed_data.at[
                idx, f"{args.model}_CS2_text_sentences_groups_cutoff_{str(args.cutoff)}"
            ] = (1 if maxx > args.cutoff else 0)
            processed_data.at[
                idx, f"{args.model}_CS2_text_sentences_groups"
            ] = sentence_groups
            processed_data.at[
                idx, f"{args.model}_CS2_text_sentences_groups_token_counts"
            ] = group_token_count

    logger.info("Classification finished. Preparing results")
    output_path = f"{args.model}_cs{args.strategy}_cutoff{str(args.cutoff)}_medhelp_classified.csv"

    logger.info(f"Writing output file to {output_path}")
    processed_data.to_csv(output_path, index=False)


if __name__ == "__main__":
    start_time = time.time()
    run_model()
    logger.info(f"Finished in {round(time.time()-start_time, 2)} seconds")
