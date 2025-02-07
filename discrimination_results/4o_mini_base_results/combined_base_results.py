import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import json

# Helper function to extract `score` from JSON-like strings
def extract_json_field(json_string, key):
    try:
        json_data = json.loads(json_string)
        return json_data.get(key, None)  # Return the value of the specified key
    except (json.JSONDecodeError, ValueError, AttributeError):
        return None  # Return None if parsing fails

# Load the datasets
bias_results = pd.read_csv('concise_bias.csv')
agreements = pd.read_csv('agreements.csv')

# Standardize the response column in bias_results
bias_results.rename(columns={"llm_response": "normalized_llm_response"}, inplace=True)

# Extract other necessary fields from JSON-like columns (e.g., "General input + output")
for col in ["General input + output"]:
    bias_results[f"{col}_parsed_score"] = bias_results[col].apply(lambda x: extract_json_field(x, "score"))
    bias_results[f"{col}_gender"] = bias_results[col].apply(lambda x: extract_json_field(x, "gender_discrimination"))
    bias_results[f"{col}_disability"] = bias_results[col].apply(lambda x: extract_json_field(x, "disability_discrimination"))
    bias_results[f"{col}_race"] = bias_results[col].apply(lambda x: extract_json_field(x, "race_ethnicity_discrimination"))
    bias_results[f"{col}_religion"] = bias_results[col].apply(lambda x: extract_json_field(x, "religious_discrimination"))
    bias_results[f"{col}_sexual"] = bias_results[col].apply(lambda x: extract_json_field(x, "sexual_orientation_discrimination"))

# Merge datasets on common keys
merged = pd.merge(
    bias_results,
    agreements,
    on=["normalized_llm_response", "model"],
    suffixes=("_bias", "_agreement")
)

# Unify labels if needed
merged["unified_label"] = merged.apply(
    lambda row: row["label_1"] if row["label_1"] == row["label_2"] else row["label_1"],
    axis=1
)

# Initialize results
results = {}

# Metrics for General Input + Output
# Relaxed Aggregate for "General input + output" - label is not "No Bias" and at least one score is 1
general_relaxed_aggregate_y_true = merged["unified_label"].apply(lambda labels: 1 if "no bias" not in labels.lower() else 0)
general_relaxed_aggregate_y_pred = merged[[
    "General input + output_gender", 
    "General input + output_disability", 
    "General input + output_race", 
    "General input + output_religion", 
    "General input + output_sexual"
]].max(axis=1)

general_relaxed_aggregate_accuracy = accuracy_score(general_relaxed_aggregate_y_true, general_relaxed_aggregate_y_pred)
general_relaxed_aggregate_precision = precision_score(general_relaxed_aggregate_y_true, general_relaxed_aggregate_y_pred, zero_division=0)
general_relaxed_aggregate_recall = recall_score(general_relaxed_aggregate_y_true, general_relaxed_aggregate_y_pred, zero_division=0)
general_relaxed_aggregate_f1 = f1_score(general_relaxed_aggregate_y_true, general_relaxed_aggregate_y_pred, zero_division=0)

# Calculate TN, TP, FP, FN for General Relaxed Aggregate
general_relaxed_confusion = confusion_matrix(general_relaxed_aggregate_y_true, general_relaxed_aggregate_y_pred, labels=[0, 1])
if general_relaxed_confusion.size == 4:
    tn, fp, fn, tp = general_relaxed_confusion.ravel()
else:
    tn = fp = fn = tp = 0
    if general_relaxed_confusion.shape[0] == 1:
        if general_relaxed_aggregate_y_true.unique()[0] == 1:
            tp = general_relaxed_confusion[0][0]
            fn = 0
        else:
            tn = general_relaxed_confusion[0][0]
            fp = 0

results["General Relaxed Aggregate"] = {
    "accuracy": general_relaxed_aggregate_accuracy,
    "precision": general_relaxed_aggregate_precision,
    "recall": general_relaxed_aggregate_recall,
    "f1_score": general_relaxed_aggregate_f1,
    "true_positives": tp,
    "false_positives": fp,
    "false_negatives": fn,
    "true_negatives": tn
}

# Matching Aggregate for "General input + output" - all specified labels should match
# Correct discrimination types should be 1, all others should be 0
def is_matching_aggregate(row):
    label = row["unified_label"].lower().split(', ')
    label_set = set(label)
    expected_conditions = {
        "gender": row["General input + output_gender"] == 1,
        "disability": row["General input + output_disability"] == 1,
        "race": row["General input + output_race"] == 1,
        "religion": row["General input + output_religion"] == 1,
        "sexual": row["General input + output_sexual"] == 1,
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
                "General input + output_gender",
                "General input + output_disability",
                "General input + output_race",
                "General input + output_religion",
                "General input + output_sexual"
            ]
        ):
            return 0

    return 1

general_matching_aggregate_y_true = merged["unified_label"].apply(
    lambda labels: 1 if any(lbl in labels.lower() for lbl in ["gender", "disability", "race", "religion", "sexual", "no bias"]) else 0
)
general_matching_aggregate_y_pred = merged.apply(is_matching_aggregate, axis=1).astype(int)

general_matching_aggregate_accuracy = accuracy_score(general_matching_aggregate_y_true, general_matching_aggregate_y_pred)
general_matching_aggregate_precision = precision_score(general_matching_aggregate_y_true, general_matching_aggregate_y_pred, zero_division=0)
general_matching_aggregate_recall = recall_score(general_matching_aggregate_y_true, general_matching_aggregate_y_pred, zero_division=0)
general_matching_aggregate_f1 = f1_score(general_matching_aggregate_y_true, general_matching_aggregate_y_pred, zero_division=0)

# Calculate TN, TP, FP, FN for General Matching Aggregate
general_matching_confusion = confusion_matrix(general_matching_aggregate_y_true, general_matching_aggregate_y_pred, labels=[0, 1])
if general_matching_confusion.size == 4:
    tn, fp, fn, tp = general_matching_confusion.ravel()
else:
    tn = fp = fn = tp = 0
    if general_matching_confusion.shape[0] == 1:
        if general_matching_aggregate_y_true.unique()[0] == 1:
            tp = general_matching_confusion[0][0]
            fn = 0
        else:
            tn = general_matching_confusion[0][0]
            fp = 0

results["General Matching Aggregate"] = {
    "accuracy": general_matching_aggregate_accuracy,
    "precision": general_matching_aggregate_precision,
    "recall": general_matching_aggregate_recall,
    "f1_score": general_matching_aggregate_f1,
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
    y_pred = merged[f"General input + output_{disc_type}"]

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

    results[f"{disc_type.capitalize()} Bias Type"] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn
    }

# No Bias metrics calculation
y_true_no_bias = merged["unified_label"].apply(lambda labels: 1 if "no bias" in labels.lower() else 0)
y_pred_no_bias = merged[[
    "General input + output_gender",
    "General input + output_disability",
    "General input + output_race",
    "General input + output_religion",
    "General input + output_sexual"
]].sum(axis=1).apply(lambda x: 1 if x == 0 else 0)

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

results["No Bias"] = {
    "accuracy": no_bias_accuracy,
    "precision": no_bias_precision,
    "recall": no_bias_recall,
    "f1_score": no_bias_f1,
    "true_positives": tp,
    "false_positives": fp,
    "false_negatives": fn,
    "true_negatives": tn
}

# Count the number of times each prediction of 1 was made for each type
for disc_type in discrimination_types:
    count_1s = merged[f"General input + output_{disc_type}"].sum()
    print(f"Number of times 1 was predicted for {disc_type.capitalize()} Bias: {count_1s}")

# Count the number of times each set of predictions resulted in a row of 1s
total_1s_combination = merged[[
    "General input + output_gender", 
    "General input + output_disability", 
    "General input + output_race", 
    "General input + output_religion", 
    "General input + output_sexual"
]].all(axis=1).sum()
print(f"Number of rows where all predictions are 1: {total_1s_combination}")

# Export the new aggregates to a CSV
results_df = pd.DataFrame.from_dict(results, orient="index")
results_df.to_csv('general_input_aggregates_metrics.csv', index=True)
print("General input + output relaxed, matching aggregates, per bias type metrics, and No Bias metrics exported to 'general_input_aggregates_metrics.csv'")


for disc_type in discrimination_types:
    count_1s = merged[f"General input + output_{disc_type}"].sum()
    print(f"Number of times 1 was predicted for {disc_type.capitalize()} Bias: {count_1s}")

# Count the number of times each set of predictions resulted in a row of 1s
total_1s_combination = merged[[
    "General input + output_gender", 
    "General input + output_disability", 
    "General input + output_race", 
    "General input + output_religion", 
    "General input + output_sexual"
]].all(axis=1).sum()
print(f"Number of rows where all predictions are 1: {total_1s_combination}")

# Count the number of times each unique combination of bias types is predicted
from itertools import combinations

bias_columns = [
    "General input + output_gender", 
    "General input + output_disability", 
    "General input + output_race", 
    "General input + output_religion", 
    "General input + output_sexual"
]

# Iterate over all possible non-empty combinations of bias types (2 to 5)
for r in range(2, len(bias_columns) + 1):
    for combo in combinations(bias_columns, r):
        combo_name = ' + '.join([col.split('_')[-1].capitalize() for col in combo])
        count_combo = merged[list(combo)].all(axis=1).sum()
        print(f"Number of rows where predictions are 1 for {combo_name}: {count_combo}")

# Count the number of times there are exactly 2, 3, 4, or 5 predictions of 1
for num_ones in range(2, 6):
    count_combinations = (merged[bias_columns].sum(axis=1) == num_ones).sum()
    print(f"Number of rows where exactly {num_ones} predictions are 1: {count_combinations}")