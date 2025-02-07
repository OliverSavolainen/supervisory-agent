import pandas as pd

# Corrected detailed data for all metrics grouped by variations
data_corrected = {
    "Variation": [
        "recall", "score before", "2 examples score", "no first sent", 
        "no examples", "no specifics", "basic", "2 examples", "1 example"
    ],
    "Race_or_Ethnic_Origin_Accuracy": [0.68, 0.64, 0.68, 0.66, 0.68, 0.66, 0.68, 0.72, 0.68],
    "Race_or_Ethnic_Origin_Precision": [0.642857, 0.600000, 0.642857, 0.640000, 0.681818, 0.652174, 0.653846, 0.727273, 0.653846],
    "Race_or_Ethnic_Origin_Recall": [0.750000, 0.750000, 0.750000, 0.666667, 0.625000, 0.625000, 0.708333, 0.666667, 0.708333],
    "Race_or_Ethnic_Origin_F1": [0.692308, 0.666667, 0.692308, 0.653061, 0.652174, 0.638298, 0.680000, 0.695652, 0.680000],
    "Gender_Accuracy": [0.56, 0.48, 0.58, 0.52, 0.48, 0.58, 0.58, 0.56, 0.56],
    "Gender_Precision": [0.500000, 0.416667, 0.521739, 0.454545, 0.428571, 0.533333, 0.520000, 0.500000, 0.500000],
    "Gender_Recall": [0.590909, 0.454545, 0.545455, 0.454545, 0.545455, 0.363636, 0.590909, 0.545455, 0.454545],
    "Gender_F1": [0.541667, 0.434783, 0.533333, 0.454545, 0.480000, 0.432432, 0.553191, 0.521739, 0.476190],
    "Religious_Beliefs_Accuracy": [0.66, 0.70, 0.72, 0.74, 0.70, 0.76, 0.70, 0.70, 0.72],
    "Religious_Beliefs_Precision": [0.680000, 0.720000, 0.772727, 0.782609, 0.739130, 0.850000, 0.720000, 0.720000, 0.800000],
    "Religious_Beliefs_Recall": [0.653846, 0.692308, 0.653846, 0.692308, 0.653846, 0.653846, 0.692308, 0.692308, 0.615385],
    "Religious_Beliefs_F1": [0.666667, 0.705882, 0.708333, 0.734694, 0.693878, 0.739130, 0.705882, 0.705882, 0.695652],
}

# Convert to DataFrame
df_corrected = pd.DataFrame(data_corrected)

# Calculate average scores across all categories for each metric and variation
df_corrected["Accuracy_Avg"] = df_corrected[
    ["Race_or_Ethnic_Origin_Accuracy", "Gender_Accuracy", "Religious_Beliefs_Accuracy"]
].mean(axis=1)
df_corrected["Precision_Avg"] = df_corrected[
    ["Race_or_Ethnic_Origin_Precision", "Gender_Precision", "Religious_Beliefs_Precision"]
].mean(axis=1)
df_corrected["Recall_Avg"] = df_corrected[
    ["Race_or_Ethnic_Origin_Recall", "Gender_Recall", "Religious_Beliefs_Recall"]
].mean(axis=1)
df_corrected["F1_Avg"] = df_corrected[
    ["Race_or_Ethnic_Origin_F1", "Gender_F1", "Religious_Beliefs_F1"]
].mean(axis=1)

# Select relevant columns for display
averages_df = df_corrected[["Variation", "Accuracy_Avg", "Precision_Avg", "Recall_Avg", "F1_Avg"]]

averages_df = averages_df.sort_values(by="F1_Avg", ascending=False)


print(averages_df)
