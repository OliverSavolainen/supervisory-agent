## Scripts and data for EU AI Act Discrimination experiments

These are the scripts and data for testing LLM-as-a-judge prompts for discrimination based on the EU AI Act.

gender_results, religion_results, and race_results.py are for StereoSet. all of them need correct column names, and might have some unnecessary stuff. 9 variations of single prompts were tested on these datasets.

Most other files are for the manually labeled dataset. Labelling was done with a GUI from data_gui.py

agreements.csv has the 72 responses where we agreed on the label.

final_scores.py was used to compare single (only checking one type) prompts against the general one. General one was better.

final_general_scores.col can be used to get results for the last experiments, paths and column names need to be fixed for it to work correctly.

Those results are in general_input_aggregates... .csv files. Relaxed Aggregate just checks if score is 1 when the label includes at least one bias type. Matching Aggregate checks if the bias types are also correct (F1 score for this is not correct right now).

## Main script to run

Main command is: python final_general_scores_col.py

To get the results of the best variations, no_specifics.csv should be the input file (bias_results variable). Make sure to specify the eval_columns that you want to evaluate. Otherwise, any exported csv from orq should work unless some responses don't use the JSON format at all.

Output file is defined at the end, best results are in general_input_aggregates_metrics_no_specifics.csv currently