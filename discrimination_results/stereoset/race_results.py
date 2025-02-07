import pandas as pd
import json
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Define helper functions
def extract_score(json_string):
    """
    Extract the "score" from a JSON-like string in the Race columns.
    """
    try:
        # Clean the JSON string
        json_string = json_string.strip().replace('```json', '').replace('```', '')
        json_data = json.loads(json_string)
        return json_data.get('score', None)  # Extract the score
    except (json.JSONDecodeError, ValueError, AttributeError):
        # Fallback: Try to extract score using regex
        match = re.search(r'"score"\s*:\s*("?\d+"?)', str(json_string))
        if match:
            score = match.group(1).strip('"')
            return int(score)
    return None  # Return None if all parsing methods fail

# Load the export.csv file (containing LLM responses)
export_file = 'race_results2.csv'  # Replace with the correct path to your export.csv file
export_df = pd.read_csv(export_file)

# Extract scores from all Race columns
race_columns = [col for col in export_df.columns if "Race" in col]
for col in race_columns:
    export_df[f'score_{col}'] = export_df[col].apply(extract_score)

# Compute average score for race columns that don't contain "ref"
for col in race_columns:
    if "ref" not in col:
        export_df[f'avg_score_{col}'] = export_df[col].apply(extract_score).mean()

# Segment the dataset into first, second, and last 52 rows before merging
segments = {
    "first_52": export_df.iloc[:52],
    "second_52": export_df.iloc[52:109],
    "last_52": export_df.iloc[-52:]
}

# Load the processed bias CSV files
bias_file = 'bias_data/race_bias.csv'
bias_df = pd.read_csv(bias_file)

bias_df = bias_df[bias_df['bias_type'] == 'race']

# Initialize results storage for segmented DataFrames
segmented_results = {}

i = 0

for segment_name, segment_df in segments.items():
    # Perform merging for the current segment
    if i > 0:
        break

    else:
        segment_df = pd.merge(
            segment_df,
            bias_df,
            left_on="reference",
            right_on="sentence",
            how="inner",
            suffixes=("", "_bias")
        )

    # Add a column to indicate whether the label is a stereotype
    segment_df['is_stereotype'] = segment_df['label'] == 'stereotype'

    # Initialize thresholds and results
    thresholds = [2, 5, 6, 7, 8, 9]
    threshold_results = {}
    ref_column = "Race or ethnic origin INPUT + OUTPUT ref recall"

    race_columns = [
    col for col in race_columns
    if "ref" in col and f'score_{col}' in segment_df.columns and segment_df[f'score_{col}'].dropna().isin([0, 1]).all()
    ]
    print(segment_df)

    for threshold in thresholds:
        # Create binary predictions for the current threshold
        segment_df[f'binary_score_{threshold}+'] = segment_df[f'score_{ref_column}'].apply(
            lambda x: 1 if isinstance(x, (int, float)) and x >= threshold else 0
        )

        # Predictions for the current threshold
        y_pred = segment_df[f'binary_score_{threshold}+']

        # Ground truth for comparison
        y_true = segment_df['is_stereotype'].fillna(False).astype(int)

        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # Store results for the current threshold
        threshold_results[threshold] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

    # Store threshold results
    threshold_results_df = pd.DataFrame.from_dict(threshold_results, orient='index')

    # Calculate race column metrics
    race_results = {}

    for col in race_columns:
        # Extract predicted scores and ground truth for the current segment
        y_pred = segment_df[f'score_{col}'].fillna(0).astype(int)  # Predictions
        y_true = segment_df['is_stereotype'].fillna(False).astype(int)

        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # Store results
        race_results[col] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

    # Convert race results to DataFrame
    race_results_df = pd.DataFrame.from_dict(race_results, orient='index')

    # Save results for this segment
    segmented_results[segment_name] = {
        "threshold_results": threshold_results_df,
        "race_results": race_results_df
    }
    i += 1


pd.set_option('display.width', None)  # No limit on the display width
pd.set_option('display.max_colwidth', None)  # No limit on column width
# Print results for each segment
for segment_name, results in segmented_results.items():
    print(f"\nSegment: {segment_name}")
    print("Threshold Results:")
    print(results["threshold_results"])
    print("Race Column Results:")
    print(results["race_results"])