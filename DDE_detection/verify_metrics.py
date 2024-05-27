import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

ground_truth_dataset = pd.read_csv(
    r"DDE_detection/data/ground_truth_model_predictions.csv.gz", compression="gzip"
)


def calculate_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    false_positive_rate = fp / (fp + tn)
    false_negative_rate = fn / (fn + tp)
    f1_score = 2 * tp / (2 * tp + fp + fn)
    return accuracy, false_positive_rate, false_negative_rate, f1_score


GPT35_CS3_metrics = calculate_metrics(
    ground_truth_dataset["ground_truth_label"],
    ground_truth_dataset["GPT3.5_CS3_prediction"],
)
print("Metrics for GPT3.5 with Classification Strategy 3")
print("Accuracy: ", GPT35_CS3_metrics[0])
print("False positive rate: ", GPT35_CS3_metrics[1])
print("False negative rate :", GPT35_CS3_metrics[2])
print("F1 score: ", GPT35_CS3_metrics[3])
print()

GPT4_CS3_metrics = calculate_metrics(
    ground_truth_dataset["ground_truth_label"],
    ground_truth_dataset["GPT4_CS3_prediction"],
)
print("Metrics for GPT4 with Classification Strategy 3")
print("Accuracy: ", GPT4_CS3_metrics[0])
print("False positive rate: ", GPT4_CS3_metrics[1])
print("False negative rate :", GPT4_CS3_metrics[2])
print("F1 score: ", GPT4_CS3_metrics[3])
print()

model_names = [
    "DistilRoBERTa",
    "DistilBERT",
    "DeBERTa",
    "RoBERTa",
    "BART",
    "GPT3.5",
    "GPT4",
]

classification_strategies = [
    "CS1",
    "CS2",
]

metrics_df = pd.DataFrame(
    columns=[
        "Cutoff",
        "RoBERTa_CS1_accuracy",
        "DistilRoBERTa_CS1_accuracy",
        "DeBERTa_CS1_accuracy",
        "DistilBERT_CS1_accuracy",
        "BART_CS1_accuracy",
        "GPT3.5_CS1_accuracy",
        "GPT4_CS1_accuracy",
        "RoBERTa_CS1_false_positive_rate",
        "DistilRoBERTa_CS1_false_positive_rate",
        "DeBERTa_CS1_false_positive_rate",
        "DistilBERT_CS1_false_positive_rate",
        "BART_CS1_false_positive_rate",
        "GPT3.5_CS1_false_positive_rate",
        "GPT4_CS1_false_positive_rate",
        "RoBERTa_CS1_false_negative_rate",
        "DistilRoBERTa_CS1_false_negative_rate",
        "DeBERTa_CS1_false_negative_rate",
        "DistilBERT_CS1_false_negative_rate",
        "BART_CS1_false_negative_rate",
        "GPT3.5_CS1_false_negative_rate",
        "GPT4_CS1_false_negative_rate",
        "RoBERTa_CS1_f1_score",
        "DistilRoBERTa_CS1_f1_score",
        "DeBERTa_CS1_f1_score",
        "DistilBERT_CS1_f1_score",
        "BART_CS1_f1_score",
        "GPT3.5_CS1_f1_score",
        "GPT4_CS1_f1_score",
        "RoBERTa_CS2_accuracy",
        "DistilRoBERTa_CS2_accuracy",
        "DeBERTa_CS2_accuracy",
        "DistilBERT_CS2_accuracy",
        "BART_CS2_accuracy",
        "GPT3.5_CS2_accuracy",
        "GPT4_CS2_accuracy",
        "RoBERTa_CS2_false_positive_rate",
        "DistilRoBERTa_CS2_false_positive_rate",
        "DeBERTa_CS2_false_positive_rate",
        "DistilBERT_CS2_false_positive_rate",
        "BART_CS2_false_positive_rate",
        "GPT3.5_CS2_false_positive_rate",
        "GPT4_CS2_false_positive_rate",
        "RoBERTa_CS2_false_negative_rate",
        "DistilRoBERTa_CS2_false_negative_rate",
        "DeBERTa_CS2_false_negative_rate",
        "DistilBERT_CS2_false_negative_rate",
        "BART_CS2_false_negative_rate",
        "GPT3.5_CS2_false_negative_rate",
        "GPT4_CS2_false_negative_rate",
        "RoBERTa_CS2_f1_score",
        "DistilRoBERTa_CS2_f1_score",
        "DeBERTa_CS2_f1_score",
        "DistilBERT_CS2_f1_score",
        "BART_CS2_f1_score",
        "GPT3.5_CS2_f1_score",
        "GPT4_CS2_f1_score",
    ]
)

y_true = ground_truth_dataset["ground_truth_label"]

# Verifying accuracy metrics for cutoffs for all the models
cutoffs = np.arange(0.05, 1, 0.05).round(2)

# Looping over each cutoff
for cutoff in cutoffs:
    row = {"Cutoff": cutoff}
    for model in model_names:
        for classification_strategy in classification_strategies:
            if model == "GPT3.5":
                if classification_strategy == "CS1":
                    columns_with_predictions = [
                        "GPT3.5_CS1_run_1_max_prediction",
                        "GPT3.5_CS1_run_2_max_prediction",
                        "GPT3.5_CS1_run_3_max_prediction",
                    ]
                else:
                    columns_with_predictions = [
                        "GPT3.5_CS2_run_1_text_sentences_groups_max_prediction",
                        "GPT3.5_CS2_run_2_text_sentences_groups_max_prediction",
                        "GPT3.5_CS2_run_3_text_sentences_groups_max_prediction",
                    ]
                # Comparing all three columns to the cutoff, and take the majority vote to be the prediction
                y_preds = [
                    ground_truth_dataset[columns_with_predictions[0]].apply(
                        lambda x: 1 if x > cutoff else 0
                    ),
                    ground_truth_dataset[columns_with_predictions[1]].apply(
                        lambda x: 1 if x > cutoff else 0
                    ),
                    ground_truth_dataset[columns_with_predictions[2]].apply(
                        lambda x: 1 if x > cutoff else 0
                    ),
                ]
                y_pred = pd.DataFrame(y_preds).mode().iloc[0]
            else:
                if classification_strategy == "CS1":
                    column_with_prediction = (
                        f"{model}_{classification_strategy}_max_prediction"
                    )
                else:
                    column_with_prediction = f"{model}_{classification_strategy}_text_sentences_groups_max_prediction"
                # Comparing the column with the prediction to the cutoff, and if the prediction is greater than or equal to the cutoff, then setting the prediction to 1, otherwise setting it to 0
                y_pred = ground_truth_dataset[column_with_prediction].apply(
                    lambda x: 1 if x > cutoff else 0
                )

            (
                accuracy,
                false_positive_rate,
                false_negative_rate,
                f1_score,
            ) = calculate_metrics(y_true, y_pred)
            row[f"{model}_{classification_strategy}_accuracy"] = accuracy
            row[
                f"{model}_{classification_strategy}_false_positive_rate"
            ] = false_positive_rate
            row[
                f"{model}_{classification_strategy}_false_negative_rate"
            ] = false_negative_rate
            row[f"{model}_{classification_strategy}_f1_score"] = f1_score

    print(
        "For cutoff ",
        cutoff,
        "- Best F1 score: ",
        *max(
            {k[:-9]: v for k, v in row.items() if "_f1_score" in k}.items(),
            key=lambda x: x[1],
        ),
    )
    metrics_df = metrics_df._append(row, ignore_index=True)

print()
print("Writing dataframe to metrics_df_verification.csv")
metrics_df.to_csv(r"DDE_detection/outputs/metrics_df_verification.csv", index=False)
