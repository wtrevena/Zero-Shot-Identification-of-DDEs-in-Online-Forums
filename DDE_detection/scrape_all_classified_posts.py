import argparse
import requests
import json
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import spacy
# First, make sure to install the spacy model:
# python -m spacy download en_core_web_sm

nlp = spacy.load('en_core_web_sm')

headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.54 Safari/537.36',
}

data = pd.read_csv("DDE_detection/data/HuggingFace_model_predictions_URLs_only.csv.gz", compression="gzip")

data['text'] = np.nan
data['text_sentences'] = np.nan

for i, row in data.iterrows():
    print(round(i / len(data) * 100, 2), "%", end="\r" )
    if pd.isna(row['text']):
        try:
            url = row['url']
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            # Find all the 'script' tags with the 'type' attribute set to 'application/ld+json'
            script_tags = soup.find('script', attrs={'type': 'application/ld+json'})
            json_object = json.loads(script_tags.contents[0])
            # If the URL is a question, the 'text' is in json_object["mainEntity"]["text"]
            # If the URL is an answer, the URL will contain '#post' 
            if '#post' in row['url']:
                # search the answers in json_object["mainEntity"]["suggestedAnswer"] for the one with the same url as the row
                for answer in json_object["mainEntity"]["suggestedAnswer"]:
                    # check if the 'url' of the 'answer' is the same as the 'url' of the row
                    # split string on "/" character, take the last element of the list 
                    # (this is necessary since the URL structure of the website changed at some point)
                    if str(answer["url"].split("/")[-1]) == str(row['url'].split("/")[-1]):
                        # If so, save the text from the answer to the relevant row in the 'text' column
                        text = json.dumps(answer["text"])
                        data.loc[i, 'text'] = text
                        spacy_sentencizer_output = nlp(text)
                        text_sentences = [s.text for s in spacy_sentencizer_output.sents]
                        data.loc[i, 'text_sentences'] = json.dumps(text_sentences)
                        break
            else:
                text = json.dumps(json_object["mainEntity"]["text"])
                data.loc[i, 'text'] = text
                spacy_sentencizer_output = nlp(text)
                text_sentences = [s.text for s in spacy_sentencizer_output.sents]
                data.loc[i, 'text_sentences'] = json.dumps(text_sentences)

        except Exception as e:
            # If there is an error, it is likely because the question / answer has been deleted
            continue

# Take a subset of the data which have a "text" which is not NA
data2 = data[~pd.isna(data['text'])]
# Save the dataset with the scraped text to a new CSV file
data2.to_csv("DDE_detection/data/HuggingFace_model_predictions.csv.gz", compression="gzip", index=False)
