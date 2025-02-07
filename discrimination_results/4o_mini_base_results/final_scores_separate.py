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

# Extract `score` and `explanation` for the first 5 input columns
input_columns = [
    "Gender input + output",
    "Disability input",
    "Race or ethnic origin INPUT + OUTPUT",
    "Religious beliefs input",
    "Sexual orientation input"
]

for col in input_columns:
    bias_results[f"{col}_score"] = bias_results[col].apply(lambda x: extract_json_field(x, "score"))
    bias_results[f"{col}_explanation"] = bias_results[col].apply(lambda x: extract_json_field(x, "explanation"))

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

# Ensure score columns are numeric
for col in [f"{col}_score" for col in input_columns]:
    merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0).astype(int)

# Initialize results
results = {}
all_y_true, all_y_pred = [], []  # For average metrics across all predictions

# Process the first 5 input columns for metrics
label_map = {
    "Gender": "Gender input + output",
    "Disability": "Disability input",
    "Race": "Race or ethnic origin INPUT + OUTPUT",
    "Religion": "Religious beliefs input",
    "Sexual": "Sexual orientation input"
}

for label, col in label_map.items():
    # Determine y_true and y_pred
    y_true = merged["unified_label"].apply(
        lambda labels: 1 if label.lower() in labels.lower() else 0
    )
    y_pred = merged[f"{col}_score"]  # Predictions from the column scores

    # Add to overall lists for average metrics
    all_y_true.extend(y_true)
    all_y_pred.extend(y_pred)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Calculate TP, FP, FN
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

    # Store results
    results[label] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn
    }

# Average metrics across all predictions for the 5 columns
average_accuracy = accuracy_score(all_y_true, all_y_pred)
average_precision = precision_score(all_y_true, all_y_pred, zero_division=0)
average_recall = recall_score(all_y_true, all_y_pred, zero_division=0)
average_f1 = f1_score(all_y_true, all_y_pred, zero_division=0)

# Calculate overall TP, FP, FN for average metrics
average_confusion = confusion_matrix(all_y_true, all_y_pred, labels=[0, 1])
if average_confusion.size == 4:
    tn, fp, fn, tp = average_confusion.ravel()
else:
    tn = fp = fn = tp = 0
    if average_confusion.shape[0] == 1:
        if all_y_true[0] == 1:
            tp = average_confusion[0][0]
            fn = 0
        else:
            tn = average_confusion[0][0]
            fp = 0

results["Average Predictions"] = {
    "accuracy": average_accuracy,
    "precision": average_precision,
    "recall": average_recall,
    "f1_score": average_f1,
    "true_positives": tp,
    "false_positives": fp,
    "false_negatives": fn
}

# Aggregate metrics for "General input + output"
general_aggregate_y_true = merged["unified_label"].apply(
    lambda labels: 1 if any(lbl in labels.lower() for lbl in ["gender", "disability", "race", "religion", "sexual"]) else 0
)
general_aggregate_y_pred = merged[[
    "General input + output_gender", 
    "General input + output_disability", 
    "General input + output_race", 
    "General input + output_religion", 
    "General input + output_sexual"
]].max(axis=1).astype(int)

general_aggregate_accuracy = accuracy_score(general_aggregate_y_true, general_aggregate_y_pred)
general_aggregate_precision = precision_score(general_aggregate_y_true, general_aggregate_y_pred, zero_division=0)
general_aggregate_recall = recall_score(general_aggregate_y_true, general_aggregate_y_pred, zero_division=0)
general_aggregate_f1 = f1_score(general_aggregate_y_true, general_aggregate_y_pred, zero_division=0)

# Calculate TP, FP, FN for General Aggregate
general_confusion = confusion_matrix(general_aggregate_y_true, general_aggregate_y_pred, labels=[0, 1])
if general_confusion.size == 4:
    tn, fp, fn, tp = general_confusion.ravel()
else:
    tn = fp = fn = tp = 0
    if general_confusion.shape[0] == 1:
        if general_aggregate_y_true.unique()[0] == 1:
            tp = general_confusion[0][0]
            fn = 0
        else:
            tn = general_confusion[0][0]
            fp = 0

results["General Aggregate"] = {
    "accuracy": general_aggregate_accuracy,
    "precision": general_aggregate_precision,
    "recall": general_aggregate_recall,
    "f1_score": general_aggregate_f1,
    "true_positives": tp,
    "false_positives": fp,
    "false_negatives": fn
}

# Two new aggregate metrics for the first five columns
# 1) Matching Aggregate - scores of 1 match all correct labels
matching_aggregate_y_true = merged["unified_label"].apply(
    lambda labels: 1 if any(lbl.lower() in labels.lower() for lbl in label_map.keys()) else 0
)
matching_aggregate_y_pred = merged[[f"{col}_score" for col in input_columns]].sum(axis=1).apply(lambda x: 1 if x == len(label_map) else 0)

matching_aggregate_accuracy = accuracy_score(matching_aggregate_y_true, matching_aggregate_y_pred)
matching_aggregate_precision = precision_score(matching_aggregate_y_true, matching_aggregate_y_pred, zero_division=0)
matching_aggregate_recall = recall_score(matching_aggregate_y_true, matching_aggregate_y_pred, zero_division=0)
matching_aggregate_f1 = f1_score(matching_aggregate_y_true, matching_aggregate_y_pred, zero_division=0)

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

results["Matching Aggregate"] = {
    "accuracy": matching_aggregate_accuracy,
    "precision": matching_aggregate_precision,
    "recall": matching_aggregate_recall,
    "f1_score": matching_aggregate_f1,
    "true_positives": tp,
    "false_positives": fp,
    "false_negatives": fn
}

# 2) Relaxed Aggregate - label is not "No Bias" and at least one score is 1
relaxed_aggregate_y_true = merged["unified_label"].apply(lambda labels: 1 if "no bias" not in labels.lower() else 0)
relaxed_aggregate_y_pred = merged[[f"{col}_score" for col in input_columns]].max(axis=1)

relaxed_aggregate_accuracy = accuracy_score(relaxed_aggregate_y_true, relaxed_aggregate_y_pred)
relaxed_aggregate_precision = precision_score(relaxed_aggregate_y_true, relaxed_aggregate_y_pred, zero_division=0)
relaxed_aggregate_recall = recall_score(relaxed_aggregate_y_true, relaxed_aggregate_y_pred, zero_division=0)
relaxed_aggregate_f1 = f1_score(relaxed_aggregate_y_true, relaxed_aggregate_y_pred, zero_division=0)

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

results["Relaxed Aggregate"] = {
    "accuracy": relaxed_aggregate_accuracy,
    "precision": relaxed_aggregate_precision,
    "recall": relaxed_aggregate_recall,
    "f1_score": relaxed_aggregate_f1,
    "true_positives": tp,
    "false_positives": fp,
    "false_negatives": fn
}

# Per-model metrics
model_results = {}
for model in merged["model"].unique():
    model_data = merged[merged["model"] == model]
    model_y_true = model_data["unified_label"].apply(
        lambda labels: 1 if any(lbl in labels.lower() for lbl in ["gender", "disability", "race", "religion", "sexual"]) else 0
    )
    model_y_pred = model_data[[f"{col}_score" for col in input_columns]].max(axis=1).astype(int)

    accuracy = accuracy_score(model_y_true, model_y_pred)
    precision = precision_score(model_y_true, model_y_pred, zero_division=0)
    recall = recall_score(model_y_true, model_y_pred, zero_division=0)
    f1 = f1_score(model_y_true, model_y_pred, zero_division=0)

    confusion = confusion_matrix(model_y_true, model_y_pred, labels=[0, 1])
    if confusion.size == 4:
        tn, fp, fn, tp = confusion.ravel()
    else:
        tn = fp = fn = tp = 0
        if confusion.shape[0] == 1:
            if model_y_true.unique()[0] == 1:
                tp = confusion[0][0]
                fn = 0
            else:
                tn = confusion[0][0]
                fp = 0

    # Store results
    model_results[model] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn
    }

# Export per-model metrics to CSV
model_results_df = pd.DataFrame(model_results).T
model_results_df.to_csv('model_metrics.csv', index=True)
print("Model metrics exported to 'model_metrics.csv'")

# Additional metrics for specific discrimination types in "General input + output"
general_results = {}
for discrimination_type in ["gender", "disability", "race", "religion", "sexual"]:
    col = f"General input + output_{discrimination_type}"

    # True labels and predictions
    y_true = merged["unified_label"].apply(
        lambda labels: 1 if discrimination_type in labels.lower() else 0
    )
    y_pred = merged[col].fillna(0).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

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

    # Store results
    general_results[discrimination_type] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn
    }

# Add statistics about "No Bias" in General input + output
general_no_bias_y_true = merged["unified_label"].apply(lambda labels: 1 if "no bias" in labels.lower() else 0)
general_no_bias_y_pred = merged["General input + output_parsed_score"].apply(lambda x: 1 if x == 0 else 0)

no_bias_accuracy = accuracy_score(general_no_bias_y_true, general_no_bias_y_pred)
no_bias_precision = precision_score(general_no_bias_y_true, general_no_bias_y_pred, zero_division=0)
no_bias_recall = recall_score(general_no_bias_y_true, general_no_bias_y_pred, zero_division=0)
no_bias_f1 = f1_score(general_no_bias_y_true, general_no_bias_y_pred, zero_division=0)

no_bias_confusion = confusion_matrix(general_no_bias_y_true, general_no_bias_y_pred, labels=[0, 1])
if no_bias_confusion.size == 4:
    tn, fp, fn, tp = no_bias_confusion.ravel()
else:
    tn = fp = fn = tp = 0
    if no_bias_confusion.shape[0] == 1:
        if general_no_bias_y_true.unique()[0] == 1:
            tp = no_bias_confusion[0][0]
            fn = 0
        else:
            tn = no_bias_confusion[0][0]
            fp = 0

general_results["No Bias"] = {
    "accuracy": no_bias_accuracy,
    "precision": no_bias_precision,
    "recall": no_bias_recall,
    "f1_score": no_bias_f1,
    "true_positives": tp,
    "false_positives": fp,
    "false_negatives": fn
}

# Convert General results to a DataFrame and export
general_results_df = pd.DataFrame.from_dict(general_results, orient="index")
general_results_df.to_csv('general_evaluation_metrics_with_tp_fp.csv', index=True)

print("\nGeneral Metrics with TP/FP exported to 'general_evaluation_metrics_with_tp_fp.csv'")

# Export results for individual bias types, average predictions, and general aggregate to CSV
results_df = pd.DataFrame.from_dict(results, orient="index")
results_df.to_csv('evaluation_metrics_with_aggregates.csv', index=True)
print("Evaluation metrics for bias types, average predictions, and general aggregate exported to 'evaluation_metrics_with_aggregates.csv'")

print(f"Total rows in dataset: {len(merged)}")
