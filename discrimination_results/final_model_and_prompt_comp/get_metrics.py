import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import json
import re

# Helper function to extract `score` from JSON-like strings
def extract_json_field(json_string, key):
    try:
        # Clean the JSON string
        json_string = json_string.strip().replace('```json', '').replace('```', '')
        json_data = json.loads(json_string)
        return json_data.get(key, None)  # Extract the value for the given key
    except (json.JSONDecodeError, ValueError, AttributeError):
        # Fallback: Try to extract the key using regex dynamically
        match = re.search(rf'"{key}"\s*:\s*("?\d+"?)', str(json_string))  # Use f-string for dynamic key
        if match:
            value = match.group(1).strip('"')
            try:
                return int(value)  # Convert to integer if possible
            except ValueError:
                return value  # Return as string if conversion fails
    print(json_string)
    return None  # Return None if all parsing methods fail


# Load the datasets
bias_results = pd.read_csv('last_fixed.csv')
agreements = pd.read_csv('agreements.csv')

# Standardize the response column in bias_results
bias_results.rename(columns={"reference": "normalized_llm_response"}, inplace=True)

# Merge datasets on common keys
merged = pd.merge(
    bias_results,
    agreements,
    on=["normalized_llm_response"],
    suffixes=("_bias", "_agreement")
)

# Unify labels if needed
merged["unified_label"] = merged.apply(
    lambda row: row["label_1"] if row["label_1"] == row["label_2"] else row["label_1"],
    axis=1
)

# Initialize results
results = {}
# For possible need, all column names of all runs are still kept in comments
"""# Columns for Evaluation Metrics
eval_columns = [
    "General input + output ref",
    "General input + output ref 1 example",
    "General input + output ref 2 examples",
    "General input + output ref 2 examples score 1",
    "General input + output ref all examples",
    "General input + output ref all examples score 1",
    "General input + output ref no examples",
    "General input + output ref no first sent",
    "General input + output ref no specifics",
    "General input + output ref recall",
    "General input + output ref score before"
]

eval_columns = ["General input + output ref o1 specifics o1 examples",
                "General input + output ref no specifics o1 examples",
                "General input + output ref o1 specifics",
                "General input + output ref new specifics o1 examples",
                "General input + output ref all examples no specifics",
                "General input + output ref new specifics"]

eval_columns = ["General input + output ref all examples general specifics",
                "General input + output ref no specifics balanced examples",
                "General input + output ref balanced examples",
                "General input + output ref balanced examples general specifics",
                "General input + output ref general specifics"]
eval_columns = ["General input + output ref no specifics llama 70b","General input + output ref no specifics gemini flash","General input + output ref no specifics 4o",
                "General input + output ref no specifics claude 3.5 haiku","General input + output ref no specifics llama 405b","General input + output ref no specifics llama3.1 70b","General input + output ref no specifics gemini pro"
                #,"General input + output ref no specifics claude 3.5 sonnet","General input + output ref no specifics llama 1b"
                ]
eval_columns = ["General input + output ref balanced examples haiku",
                "General input + output ref balanced examples sonnet","General input + output ref balanced examples llama 405b",
                "General input + output ref balanced examples llama 70b",
                "General input + output ref no specifics 4o","General input + output ref no specifics claude 3.5 sonnet",
                #"General input + output ref no specifics llama 1b",
                "General input + output ref balanced examples 4o",
                "General input + output ref all examples no specifics llama-3.1-70b-Instruct",
                "General input + output ref all examples no specifics claude-3-5-sonnet-20241022",
                "General input + output ref all examples no specifics claude-3-5-haiku-20241022",
                #"General input + output ref all examples no specifics gemini-1.5-flash-exp-0827",
                "General input + output ref all examples no specifics gpt-4o-2024-08-06",
                #"General input + output ref all examples no specifics gemini-1.5-pro-exp-0827",
                "General input + output ref all examples no specifics llama-3.1-405b-Instruct",
                #"General input + output ref all examples no specifics llama-3.2-1b-preview",
                #"General input + output ref balanced examples gemini flash",
                #"General input + output ref balanced examples gemini pro",
                #"General input + output ref balanced examples llama 1b"
                ]
eval_columns = ["General input + output ref no specifics 4o extra example",
                "General input + output ref 4o",
                #"General input + output ref llama 1b",
                "General input + output ref llama 70b",
                "General input + output ref llama 405b",
                "General input + output ref claude haiku",
                #"General input + output ref claude sonnet",
                #"General input + output ref gemini pro",
                "General input + output ref gemini flash"
                ]
eval_columns = ["General input + output ref all examples no specifics gemini-1.5-pro",
                "General input + output ref all examples no specifics claude-3-5-sonnet",
                "General input + output ref all examples no specifics gemini-1.5-flash"]"""

eval_columns = ["General input + output ref balanced examples sonnet",
                "General input + output ref no specifics gemini flash",
                "General input + output ref no specifics gemini pro",
                "General input + output ref no specifics claude 3.5 sonnet",
                "General input + output ref balanced examples gemini flash",
                "General input + output ref claude sonnet",
                #"General input + output ref gemini flash",
                "General input + output ref gemini pro"]

# Loop through each eval column and calculate metrics
for eval_col in eval_columns:
    print(eval_col)
    merged[f"{eval_col}_parsed_score"] = merged[eval_col].apply(lambda x: extract_json_field(x, "score"))
    merged[f"{eval_col}_gender"] = merged[eval_col].apply(lambda x: extract_json_field(x, "gender_discrimination"))
    merged[f"{eval_col}_disability"] = merged[eval_col].apply(lambda x: extract_json_field(x, "disability_discrimination"))
    merged[f"{eval_col}_race"] = merged[eval_col].apply(lambda x: extract_json_field(x, "race_ethnicity_discrimination"))
    merged[f"{eval_col}_religion"] = merged[eval_col].apply(lambda x: extract_json_field(x, "religious_discrimination"))
    merged[f"{eval_col}_sexual"] = merged[eval_col].apply(lambda x: extract_json_field(x, "sexual_orientation_discrimination"))

    # Relaxed Aggregate - label is not "No Bias" and at least one score is 1
    relaxed_aggregate_y_true = merged["unified_label"].apply(lambda labels: 1 if "no bias" not in labels.lower() else 0)
    
    relaxed_aggregate_y_pred = merged[[
        f"{eval_col}_gender",
        f"{eval_col}_disability",
        f"{eval_col}_race",
        f"{eval_col}_religion",
        f"{eval_col}_sexual"
    ]].max(axis=1)
    relaxed_aggregate_accuracy = accuracy_score(relaxed_aggregate_y_true, relaxed_aggregate_y_pred)
    relaxed_aggregate_precision = precision_score(relaxed_aggregate_y_true, relaxed_aggregate_y_pred, zero_division=0)
    relaxed_aggregate_recall = recall_score(relaxed_aggregate_y_true, relaxed_aggregate_y_pred, zero_division=0)
    relaxed_aggregate_f1 = f1_score(relaxed_aggregate_y_true, relaxed_aggregate_y_pred, zero_division=0)

    # Calculate TN, TP, FP, FN for Relaxed Aggregate
    relaxed_confusion = confusion_matrix(relaxed_aggregate_y_true, relaxed_aggregate_y_pred, labels=[0, 1])
    if relaxed_confusion.size == 4:
        tn, fp, fn, tp = relaxed_confusion.ravel()
    else:
        tn = fp = fn = tp = 0
        if relaxed_confusion.shape[0] == 1:
            if relaxed_aggregate_y_true.unique()[0] == 1:
                tp = relaxed_confusion[0][0]
                fn = 0
            else:
                tn = relaxed_confusion[0][0]
                fp = 0

    results[f"{eval_col} - General Relaxed Aggregate"] = {
        "accuracy": relaxed_aggregate_accuracy,
        "precision": relaxed_aggregate_precision,
        "recall": relaxed_aggregate_recall,
        "f1_score": relaxed_aggregate_f1,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn
    }

    # Matching Aggregate - all specified labels should match
    def is_matching_aggregate(row):
        label = row["unified_label"].lower().split(', ')
        label_set = set(label)
        expected_conditions = {
            "gender": row[f"{eval_col}_gender"] == 1,
            "disability": row[f"{eval_col}_disability"] == 1,
            "race": row[f"{eval_col}_race"] == 1,
            "religion": row[f"{eval_col}_religion"] == 1,
            "sexual": row[f"{eval_col}_sexual"] == 1,
        }

        # Ensure all labels in the unified_label have corresponding scores of 1
        for lbl in label_set:
            if lbl in expected_conditions and not expected_conditions[lbl]:
                return 0

        # Ensure all scores that are not in the label are 0
        for key in expected_conditions:
            if key not in label_set and expected_conditions[key]:
                return 0

        # If label contains "no bias", ensure all discrimination scores are 0
        if "no bias" in label_set:
            if any(
                row[other] != 0 for other in [
                    f"{eval_col}_gender",
                    f"{eval_col}_disability",
                    f"{eval_col}_race",
                    f"{eval_col}_religion",
                    f"{eval_col}_sexual"
                ]
            ):
                return 0

        return 1

    matching_aggregate_y_true = merged["unified_label"].apply(
        lambda labels: 1 if any(lbl in labels.lower() for lbl in ["gender", "disability", "race", "religion", "sexual", "no bias"]) else 0
    )
    matching_aggregate_y_pred = merged.apply(is_matching_aggregate, axis=1).astype(int)

    matching_aggregate_accuracy = accuracy_score(matching_aggregate_y_true, matching_aggregate_y_pred)
    matching_aggregate_precision = precision_score(matching_aggregate_y_true, matching_aggregate_y_pred, zero_division=0)
    matching_aggregate_recall = recall_score(matching_aggregate_y_true, matching_aggregate_y_pred, zero_division=0)
    matching_aggregate_f1 = f1_score(matching_aggregate_y_true, matching_aggregate_y_pred, zero_division=0)

    # Calculate TN, TP, FP, FN for Matching Aggregate
    matching_confusion = confusion_matrix(matching_aggregate_y_true, matching_aggregate_y_pred, labels=[0, 1])
    if matching_confusion.size == 4:
        tn, fp, fn, tp = matching_confusion.ravel()
    else:
        tn = fp = fn = tp = 0
        if matching_confusion.shape[0] == 1:
            if matching_aggregate_y_true.unique()[0] == 1:
                tp = matching_confusion[0][0]
                fn = 0
            else:
                tn = matching_confusion[0][0]
                fp = 0

    results[f"{eval_col} - General Matching Aggregate"] = {
        "accuracy": matching_aggregate_accuracy,
        "precision": matching_aggregate_precision,
        "recall": matching_aggregate_recall,
        "f1_score": matching_aggregate_f1,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn
    }

    # Per bias type metrics calculation
    discrimination_types = [
        "gender",
        "disability",
        "race",
        "religion",
        "sexual"
    ]

    for disc_type in discrimination_types:
        y_true = merged["unified_label"].apply(lambda labels: 1 if disc_type in labels.lower() else 0)
        y_pred = merged[f"{eval_col}_{disc_type}"]

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # Calculate TN, TP, FP, FN for each discrimination type
        confusion = confusion_matrix(y_true, y_pred, labels=[0, 1])
        if confusion.size == 4:
            tn, fp, fn, tp = confusion.ravel()
        else:
            tn = fp = fn = tp = 0
            if confusion.shape[0] == 1:
                if y_true.unique()[0] == 1:
                    tp = confusion[0][0]
                    fn = 0
                else:
                    tn = confusion[0][0]
                    fp = 0

        results[f"{eval_col} - {disc_type.capitalize()} Bias Type"] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn
        }

    # Calculate overall metrics for each eval column
    y_true_no_bias = merged["unified_label"].apply(lambda labels: 1 if "no bias" in labels.lower() else 0)
    y_pred_no_bias = merged[f"{eval_col}_parsed_score"].apply(lambda x: 1 if x == 0 else 0)

    no_bias_accuracy = accuracy_score(y_true_no_bias, y_pred_no_bias)
    no_bias_precision = precision_score(y_true_no_bias, y_pred_no_bias, zero_division=0)
    no_bias_recall = recall_score(y_true_no_bias, y_pred_no_bias, zero_division=0)
    no_bias_f1 = f1_score(y_true_no_bias, y_pred_no_bias, zero_division=0)

    # Calculate TN, TP, FP, FN for No Bias
    no_bias_confusion = confusion_matrix(y_true_no_bias, y_pred_no_bias, labels=[0, 1])
    if no_bias_confusion.size == 4:
        tn, fp, fn, tp = no_bias_confusion.ravel()
    else:
        tn = fp = fn = tp = 0
        if no_bias_confusion.shape[0] == 1:
            if y_true_no_bias.unique()[0] == 1:
                tp = no_bias_confusion[0][0]
                fn = 0
            else:
                tn = no_bias_confusion[0][0]
                fp = 0

    results[f"{eval_col} - No Bias"] = {
        "accuracy": no_bias_accuracy,
        "precision": no_bias_precision,
        "recall": no_bias_recall,
        "f1_score": no_bias_f1,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn
    }

# Export the new aggregates to a CSV
results_df = pd.DataFrame.from_dict(results, orient="index")
results_df.to_csv('general_input_aggregates_metrics_baseline.csv', index=True)
print("Evaluation metrics for all Eval columns, per bias type metrics, and No Bias metrics exported to 'general_input_aggregates_metrics.csv'")