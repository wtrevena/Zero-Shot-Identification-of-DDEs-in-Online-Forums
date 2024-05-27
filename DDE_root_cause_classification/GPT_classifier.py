import argparse
from ast import literal_eval
import logging
import time
import re
import json

from openai import OpenAI
import pandas as pd
from tqdm import tqdm


OPENAI_API_KEY = "TEST"
client = OpenAI(api_key=OPENAI_API_KEY)


system_prompt = "Function as the leading expert on patient narratives from MedHelp.org, with a profound understanding of medication and treatment discontinuation reasons discussed on the forum. Utilize your comprehensive knowledge of health issues, treatments, patient concerns, and the typical language, including acronyms and slang used on MedHelp.org. Your expertise encompasses interpreting comments related to changes in patient behavior regarding medical treatments or medication, including reasons for discontinuation or adherence challenges."

user_prompt_template = """
Analyze the following comment from MedHelp.org and provide responses in structured JSON format:

1. Describe the rationale behind the author's decision to discontinue their medication or treatment as depicted in the comment.
2. Identify all applicable hypotheses from the 'hypotheses_template' that align with the scenario described in the comment. Format your response as: {"all_relevant_hypotheses": [{"hypothesis_key": "[Insert appropriate hypothesis key from the hypotheses_template here]", "explanation": "[Provide an explanation in your word here for why you selected this hypothesis key]"}, ...]}.
3. Determine the most relevant hypothesis from the 'hypotheses_template' based on the comment, and provide a response in the format: {"best_hypothesis_key": "[Insert the most appropriate hypothesis key from the hypotheses_template here]", "best_hypothesis_value": "[Provide an explanation in your word here for why you selected this hypothesis key]"}.

Combine the responses into a single JSON response structured as:
{
  "task_1_response": "Explanation for stopping the treatment or medication",
  "task_2_response": {"all_relevant_hypotheses": [{"hypothesis_key": "[Insert appropriate hypothesis key from the hypotheses_template here]", "explanation": "[Provide an explanation in your word here for why you selected this hypothesis key]"}, ...]},
  "task_3_response": {"best_hypothesis_key": "[Insert the most appropriate hypothesis key from the hypotheses_template here]", "best_hypothesis_value": "[Provide an explanation in your word here for why you selected this hypothesis key]"}
}

Enclosed 'hypotheses_template' JSON:
{
  "Treatment Success": "The comment indicates that the patient discontinued the medication or treatment because their health condition improved significantly or the treatment course was successfully completed, rendering further treatment unnecessary.",
  "Treatment Inefficacy": "The comment states that the medication or treatment was discontinued because it was not effective, compelling cessation of use.",
  "Adverse Reactions": "This comment describes the discontinuation of medication or treatment due to adverse side effects, allergic reactions, or harmful interactions.",
  "Accessibility Issues": "The comment indicates that the medication or treatment was stopped due to accessibility issues such as lack of insurance, financial constraints, market unavailability, or prescriber decisions.",
  "Personal Choices": "The comment indicates that the medication or treatment was stopped based on personal choices, influenced by beliefs, lifestyle changes, or non-adherence.",
  "Alternative Medical Reasons": "The comment indicates that the treatment was stopped for specific medical reasons not covered by other categories, such as pregnancy or a new health condition.",
  "Indeterminate": "It is unclear why the medication or treatment was stopped.",
  "Non-Discontinuation": "The comment does not indicate that any medication or treatment was stopped."
}

**Begin Analysis**
[TEXT GOES HERE]
**End Analysis**

"""


logger = logging.getLogger(__name__)
logging.basicConfig(format="[%(filename)s:%(lineno)s] %(message)s")
logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(prog='ChatGPT_classifier',
                                     description="Script to classify Drug Discontinuation Events (DDE) using different GPT models.")

    # Model argument
    parser.add_argument("model", 
                        choices=["gpt-4o-2024-05-13", "gpt-4-0125-preview"], 
                        help="Select the GPT model for DDE classification.")

    args = parser.parse_args()
    return args

# Function to classify text using OpenAI API with retry behavior
def classify_text(text, max_retries=3, backoff_factor=2, model="gpt-4o-2024-05-13"):
    for attempt in range(max_retries):
        try:
            classification_result = client.chat.completions.create(
                # model="gpt-4-0125-preview",
                model=model,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt_template.replace('[TEXT GOES HERE]', text)}],
                max_tokens=4095,
                temperature=0,  # Ensure reproducibility
                n=1,
                seed=42  # Set a fixed seed for consistent results
            )
            # Attempt to extract and parse the JSON string from the response content
            json_response_string = classification_result.choices[0].message.content.strip('` \n').removeprefix('json\n')
            json_response = json.loads(json_response_string)  # This will throw an error if parsing fails
            return json_response  # Return the response if parsing is successful
        except (json.JSONDecodeError, KeyError, AttributeError) as parse_error:
            print(f"Attempt {attempt + 1} failed due to parsing error: {parse_error}")
        except Exception as e:
            print(f"Attempt {attempt + 1} failed with error: {str(e)}")
        time.sleep(backoff_factor ** attempt)  # Exponential backoff

    # If all attempts fail, return None or raise an exception based on your preference
    print(f"All {max_retries} attempts failed. Skipping classification for this text.")
    return None  # Or you can raise an exception if that fits your workflow better

def run_model():
    args = parse_args()

    model_name = args.model
    strategy = 3
    logger.info(f"Parameters\n Model Name: {model_name} \n Strategy: {strategy}")

    logger.info("Reading input data")
    data = pd.read_csv("./DDE_root_cause_classification/data/ground_truth.csv.gz", compression="gzip")
    logger.info(f"Row count: {len(data)}")

    output_columns = [
        "url",
        "text",
        f"{args.model}_task_1_response",
        f"{args.model}_task_2_response",
        f"{args.model}_task_3_response"
    ]

    processed_data = pd.DataFrame(index=range(0, data.shape[0]), columns=output_columns)
    processed_data["url"] = data["url"]
    processed_data["text"] = data["text"]

    logger.info(f"Starting classification strategy {strategy}")

    for index, row in tqdm(processed_data.iterrows(), total=processed_data.shape[0]):
        print(f"Classifying row {index + 1}...", end="\r")
        classification_result = classify_text(row['text'], model=model_name)

        if classification_result:
            processed_data.at[index, f"{args.model}_task_1_response"] = classification_result['task_1_response']
            processed_data.at[index, f"{args.model}_task_2_response"] = json.dumps(classification_result['task_2_response'])
            processed_data.at[index, f"{args.model}_task_3_response"] = json.dumps(classification_result['task_3_response'])
        else:
            logger.error(f"Failed to classify text for row {index}")

    logger.info("Classification finished. Preparing results")
    output_path = f"DDE_root_cause_classification/outputs/{args.model}_cs3_medhelp_classified.csv"

    logger.info(f"Writing output file to {output_path}")
    processed_data.to_csv(output_path, index=False)

if __name__ == "__main__":
    start_time = time.time()
    run_model()
    logger.info(f"Finished in {round(time.time()-start_time, 2)} seconds")

