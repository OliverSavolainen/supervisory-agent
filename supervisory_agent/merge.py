import pandas as pd

df1 = pd.read_csv('merged_output.csv')
df2 = pd.read_csv('merged_evaluation_results.csv')

merged_df = pd.merge(df1, df2, on=["normalized_llm_response"])

print(merged_df.head())

# If you want to save the merged result to a new CSV file:
merged_df.to_csv("final_merged_output.csv", index=False)