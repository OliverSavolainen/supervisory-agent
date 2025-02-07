import pandas as pd
import json

# Function to safely extract the "score" field from a JSON-like string
def extract_score(json_string):
    """
    Extract the 'score' field from a JSON-like string in the input columns.
    """
    try:
        # Clean and parse the JSON-like string
        json_string = json_string.strip().replace('```json', '').replace('```', '')
        json_data = json.loads(json_string)
        return json_data.get('score', None)  # Return the score
    except (json.JSONDecodeError, ValueError, AttributeError):
        return None  # Return None if parsing fails

# Function to extract the "content" field from the "template" column
def extract_content(template_string):
    """
    Extract the 'content' field from the JSON-like string in the 'template' column.
    """
    try:
        json_data = json.loads(template_string)  # Parse the JSON
        if isinstance(json_data, list) and "content" in json_data[0]:
            return json_data[0]["content"]  # Return the 'content' field
    except (json.JSONDecodeError, ValueError, IndexError, KeyError):
        return None  # Return None if parsing fails or field is missing

# Load the CSV file
file_path = 'concise_bias.csv'  # Replace with your actual file path
df = pd.read_csv(file_path)

# Automatically identify all columns containing "input" (case-insensitive)
input_columns = [col for col in df.columns if "input" in col.lower()]

# Extract the prompt (content) from the "template" column
df["prompt"] = df["template"].apply(extract_content)

# Add scores for all "input" columns paired with the model
for col in input_columns:
    for model in df["model"].unique():  # Iterate over unique models
        pair_col_name = f"{col.lower().replace(' ', '_')}_{model.lower().replace(' ', '_')}_score"
        df[pair_col_name] = df.apply(
            lambda row: extract_score(row[col]) if row["model"] == model else None, axis=1
        )

# Ensure all newly created columns are numeric
pairing_columns = [
    col for col in df.columns if any(input_col.lower().replace(' ', '_') in col for input_col in input_columns)
]
for col in pairing_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

# Add a separate column for each model's llm_response
for model in df["model"].unique():
    response_col_name = f"llm_response_{model.lower().replace(' ', '_')}"
    df[response_col_name] = df.apply(
        lambda row: row["llm_response"] if row["model"] == model else None, axis=1
    )

# Remove unnecessary columns, keep only the relevant ones
columns_to_keep = ["prompt", "model"] + pairing_columns + [f"llm_response_{model.lower().replace(' ', '_')}" for model in df["model"].unique()]
df = df[columns_to_keep]

# Normalize dataset to have rows for each model and prompt
normalized_dataset = df.melt(
    id_vars=["prompt", "model"],
    value_vars=[f"llm_response_{model.lower().replace(' ', '_')}" for model in df["model"].unique()],
    var_name="llm_response_model",
    value_name="normalized_llm_response",  # Use a unique name for the melted column
).dropna()

# Export the aggregated scores DataFrame to a CSV file
aggregated_output_file = 'aggregated_pairing_scores.csv'
df.to_csv(aggregated_output_file, index=False)
print(f"\nAggregated Scores DataFrame exported to {aggregated_output_file}")

# Export the normalized dataset to a separate CSV file
normalized_output_file = 'normalized_prompt_responses.csv'
#normalized_dataset.to_csv(normalized_output_file, index=False)
print(f"\nNormalized Prompt-Response Dataset exported to {normalized_output_file}")

column_scores = df[pairing_columns].sum().sort_values(ascending=False)
print("\nTotal Scores from Each Column:")
print(column_scores)

# Total scores from each model (number of rows where at least one value is 1)
model_scores = {}
for model in df["model"].unique():
    model_columns = [col for col in pairing_columns if model.lower().replace(' ', '_') in col]
    model_mask = (df[model_columns] == 1).any(axis=1)
    model_scores[model] = model_mask.sum()

print("\nTotal Scores (Row Counts) from Each Model:")
for model, score in model_scores.items():
    print(f"{model}: {score}")

# Total scores from each input type
input_type_scores = {}
for input_type in input_columns:
    input_columns_subset = [col for col in pairing_columns if input_type.lower().replace(' ', '_') in col]
    input_type_scores[input_type] = df[input_columns_subset].sum().sum()

print("\nTotal Scores from Each Input Type:")
for input_type, score in input_type_scores.items():
    print(f"{input_type}: {score}")