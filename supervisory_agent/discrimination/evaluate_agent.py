import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# Load the evaluation and agreements datasets
evaluation_results = pd.read_csv("evaluation_results2.csv")
agreements = pd.read_csv("agreements.csv")

# Merge datasets on `normalized_llm_response`
merged = pd.merge(
    evaluation_results,
    agreements,
    on="normalized_llm_response",
    suffixes=("_eval", "_agreement")
)

# Define conditions for evaluation
# Loop number 1 should have label "No Bias", others should not
merged["expected_label"] = merged["loop_number"].apply(
    lambda x: "No Bias" if x == 1 else "Bias"
)

print(merged["expected_label"].head())
print(merged["label_1"].head())

# Calculate metrics
y_pred = merged["expected_label"].apply(lambda x: 1 if x == "Bias" else 0).astype(int)
y_true = merged["label_1"].apply(lambda x: 0 if "no bias" in x.lower() else 1).astype(int)

# Metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

# Confusion matrix for TP, FP, FN, TN
tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

# Output results
results = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "true_positives": tp,
    "false_positives": fp,
    "false_negatives": fn,
    "true_negatives": tn
}

# Export merged results and metrics
merged.to_csv("merged_evaluation_results.csv", index=False)
results_df = pd.DataFrame([results])
results_df.to_csv("evaluation_metrics.csv", index=False)

# Print the results
print("Evaluation Metrics:")
print(results)