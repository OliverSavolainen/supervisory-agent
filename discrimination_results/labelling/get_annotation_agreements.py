import pandas as pd

# Load the labeled datasets
df1 = pd.read_csv('labeled_responses.csv')
df2 = pd.read_csv('labeled_responses2.csv')

# Ensure the datasets are sorted the same way (optional, based on unique identifiers)
df1 = df1.sort_values(by=['prompt', 'model']).reset_index(drop=True)
df2 = df2.sort_values(by=['prompt', 'model']).reset_index(drop=True)

# Merge the two datasets on shared columns (e.g., prompt and model)
merged = pd.merge(df1, df2, on=['prompt', 'model', 'normalized_llm_response'], suffixes=('_1', '_2'))

# Identify rows where labels agree and disagree
merged['agreement'] = merged['label_1'] == merged['label_2']

# Dataset of agreements
agreements = merged[merged['agreement']].copy()

# Dataset of disagreements
disagreements = merged[~merged['agreement']].copy()

# Save agreements and disagreements to CSV files
agreements.to_csv('agreements.csv', index=False)
disagreements.to_csv('disagreements.csv', index=False)

# Calculate agreement scores per label
label_agreements = []
for label in set(merged['label_1'].unique()).union(set(merged['label_2'].unique())):
    total_label_rows = merged[
        (merged['label_1'] == label) | (merged['label_2'] == label)
    ].shape[0]
    agreed_label_rows = merged[
        (merged['label_1'] == label) & (merged['label_2'] == label)
    ].shape[0]
    agreement_score = agreed_label_rows / total_label_rows if total_label_rows > 0 else 0
    label_agreements.append({'label': label, 'agreement_score': agreement_score})

label_agreements_df = pd.DataFrame(label_agreements)

# Calculate agreement scores per model
model_agreements = []
for model in merged['model'].unique():
    total_model_rows = merged[merged['model'] == model].shape[0]
    agreed_model_rows = merged[(merged['model'] == model) & (merged['agreement'])].shape[0]
    agreement_score = agreed_model_rows / total_model_rows if total_model_rows > 0 else 0
    model_agreements.append({'model': model, 'agreement_score': agreement_score})

model_agreements_df = pd.DataFrame(model_agreements)

# Save agreement scores to CSV files
#label_agreements_df.to_csv('label_agreement_scores.csv', index=False)
#model_agreements_df.to_csv('model_agreement_scores.csv', index=False)

# Print agreement scores
print("\nAgreement Scores per Label:")
print(label_agreements_df)

print("\nAgreement Scores per Model:")
print(model_agreements_df)