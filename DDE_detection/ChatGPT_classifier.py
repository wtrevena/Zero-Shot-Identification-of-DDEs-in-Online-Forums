import argparse
from ast import literal_eval
import logging
import time
import re

from openai import OpenAI
import pandas as pd
from tqdm import tqdm


OPENAI_API_KEY = "TEST"
client = OpenAI(api_key=OPENAI_API_KEY)

PROMPT_1_2 = """
Your task as an AI model is to determine the probability that a given comment from medhelp.org entails a "Drug Discontinuation Event".

The concept of a "Drug Discontinuation Event" involves a specific individual stopping a recurring medication or treatment. It includes instances where the individual has switched from one medication to another, but not one-time treatments.

To formulate your task, consider each comment as a premise and the statement "A person stopped taking a medication" as the hypothesis. Your goal is to estimate the probability that the hypothesis is true given the premise. Express this probability as a percentage.

For example, if a comment strongly implies a Drug Discontinuation Event, you might respond with "Probability of entailment: 95%". Conversely, if a comment does not suggest a Drug Discontinuation Event, you might respond with "Probability of entailment: 5%".

Examples:

    Comment: "I stopped taking my birth control medication." Response: "Probability of entailment: 100%"
    Comment: "I took the plan B pill yesterday." Response: "Probability of entailment: 5%"
    Comment: "I changed my birth control medication because of side effects." Response: "Probability of entailment: 95%"
    Comment: "I stopped taking my birth control medication because I was feeling worse, but then I started taking it again." Response: "Probability of entailment: 100%"

Next, I will provide a comment, and you will estimate the probability of entailment as instructed. Remember, the output should be in the format of "Probability of entailment: X%" where X is the estimated probability.
"""

PROMPT_3 = """
As an AI model, your task is to classify each comment from medhelp.org into one of two categories based on the content of the comment. The categories are: "Drug Discontinuation Event" (1) and "Non-Drug Discontinuation Event" (0).

A "Drug Discontinuation Event" (1) is any instance where it can be deduced from the comment that a specific individual has stopped a recurring medication or treatment. This includes cases where the individual has switched from one medication to another. It does not include one-time treatments.

A "Non-Drug Discontinuation Event" (0) is any instance where it cannot be inferred from the comment that a specific individual has stopped a recurring medication or treatment.

Examples:
    Comment: "I stopped taking my birth control medication" Response: "1"
    Comment: "I took the plan B pill yesterday" Response: "0"
    Comment: "I changed my birth control medication because of side effects" Response: "1"
    Comment: "I stopped taking my birth control medication because I was feeling worse, but then I started taking it again" Response: "1"
   
Next, I will give you a comment, and you will classify it according to these instructions. Remember to only respond with "1" or "0".
"""

logger = logging.getLogger(__name__)
logging.basicConfig(format="[%(filename)s:%(lineno)s] %(message)s")
logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(prog='ChatGPT_classifier',
                                     description="Script to classify Drug Discontinuation Events (DDE) using different GPT models.")

    # Model argument
    parser.add_argument("model", 
                        choices=["gpt-4", "gpt-3.5-turbo"], 
                        help="Select the GPT model for DDE classification.")

    # Cutoff argument
    parser.add_argument("--cutoff", 
                        type=float, 
                        default=0.95, 
                        help="Set the prediction threshold for a Drug Discontinuation Event (DDE).  Values above this threshold indicate a positive prediction. (default: 0.95)")

    # Classification Strategy argument
    parser.add_argument("--strategy", 
                        type=int, 
                        choices=[1, 2, 3], 
                        default=1, 
                        metavar="Classification Strategy (CS)",
                        help="Choose a classification strategy for the DDE. (default: 1)")

    args = parser.parse_args()
    return args


def run_model():
    args = parse_args()


    model_name = args.model
    strategy = args.strategy
    logger.info(f"Parameters\n Model Name: {model_name} \n Strategy: {strategy}")

    logger.info("Reading input data")
    data = pd.read_csv("./DDE_detection/data/ground_truth.csv.gz", compression="gzip")
    logger.info(f"Row count: {len(data)}")

    output_columns = [
        "url",
        "text",
        "text_sentences",
        f"{args.model}_CS{args.strategy}_predictions",
        f"{args.model}_CS{args.strategy}_max_prediction",
    ]
    if args.strategy in [1, 2]:
        instructions = PROMPT_1_2
    elif args.strategy == 3:
        instructions = PROMPT_3
    else:
        raise Exception("Strategy not found")

    processed_data = pd.DataFrame(index=range(0, data.shape[0]), columns=output_columns)
    processed_data["url"] = data["url"]
    processed_data["text"] = data["text"]
    processed_data["text_sentences"] = data["text_sentences"]
    processed_data[f"{args.model}_CS{args.strategy}_max_prediction"] = None
    processed_data[f"{args.model}_CS{args.strategy}_predictions"] = None

    logger.info(f"Starting classification strategy {args.strategy}")

    for index, row in tqdm(processed_data.iterrows(), total=processed_data.shape[0]):
        if row[f"{args.model}_CS{args.strategy}_predictions"] is None:
            sentence_classifications = []
            if args.strategy == 1:
                text_sentences = literal_eval(row["text_sentences"])
            elif args.strategy in [2, 3]:
                text_sentences = [row["text"]]
            for sentence in text_sentences:
                messages = [
                    {"role": "system", "content": instructions},
                ]
                messages.append({"role": "user", "content": sentence})
                for attempt in range(3):  # Number of attempts
                    try:
                        chat_completion = client.chat.completions.create(
                            model=args.model, messages=messages
                        )
                        answer = chat_completion.choices[0].message.content
                        print(answer)
                        sentence_classifications.append(answer)
                        break
                    except:
                        print(
                            f"Model overloaded, sleeping for {2 ** attempt} seconds...",
                            end="\r",
                        )
                        time.sleep(2)
        else:
            print(
                f"comment: {index} already has a value for chatGPT_classification",
                end="\r",
            )
        processed_data.at[
            index, f"{args.model}_CS{args.strategy}_predictions"
        ] = sentence_classifications

    if args.strategy in [1, 2]:
        none_values = []
        percentage_missing = []
        for i, row in processed_data.iterrows():
            sentences = row[f"{args.model}_CS{args.strategy}_predictions"]
            try:
                numbers = [
                    float(re.search(r"(\d+)", sentence)[0]) / 100
                    for sentence in sentences
                    if sentence is not None
                    and re.search(r"(\d+)", sentence) is not None
                ]
                if numbers:  # Check if the list is not empty
                    processed_data.at[
                        i, f"{args.model}_CS{args.strategy}_max_prediction"
                    ] = max(numbers)
                else:
                    percentage_missing.append(
                        i
                    )  # Save dataset index and row index for rows without '%'
            except:  # Max of empty list
                none_values.append(i)  # Save dataset index and row index

        processed_data[f"chatGPT_classification_cutoff_{args.cutoff}"] = (
            processed_data[f"{args.model}_CS{args.strategy}_max_prediction"]
            >= args.cutoff
        ).astype(int)
    else:
        for i, row in processed_data.iterrows():
            sentences = row[f"{args.model}_CS{args.strategy}_predictions"]
            processed_data.at[
                i, f"{args.model}_CS{args.strategy}_max_prediction"
            ] = int(sentences[0])
        processed_data[f"chatGPT_classification_cutoff_{args.cutoff}"] = processed_data[
            f"{args.model}_CS{args.strategy}_max_prediction"
        ]

    logger.info("Classification finished. Preparing results")
    output_path = f"DDE_detection/outputs/{args.model}_cs{args.strategy}_cutoff{str(args.cutoff)}_medhelp_classified.csv"

    logger.info(f"Writing output file to {output_path}")
    processed_data.to_csv(output_path, index=False)


if __name__ == "__main__":
    start_time = time.time()
    run_model()
    logger.info(f"Finished in {round(time.time()-start_time, 2)} seconds")
