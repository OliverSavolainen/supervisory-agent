## Scripts and data for EU AI Act Discrimination experiments

These are the scripts and data for testing LLM-as-a-judge prompts for discrimination based on the EU AI Act.

stereoset folder is scripts and data for StereoSet. All scripts need correct column names, and might have some unnecessary stuff. 9 variations of single prompts were tested on these datasets.

Folder labelling has all labels from the both of us coming from the scripts in the folder. Labelling was done with a GUI from data_gui.py

final_dataset.csv has the 72 responses where we agreed on the label.

4o_mini_base_results has the first stage of experiments with GPT 4o-mini model. final_scores_separate.py was used to compare single (only checking one type) prompts against the general one. General one was better.

final_general_scores.col can be used to get results for the last experiments between models and best prompts variations, paths and column names need to be fixed for it to work correctly.

Those results are in the all_scores.csv file. Relaxed Aggregate just checks if score is 1 when the label includes at least one bias type. Matching Aggregate checks if the bias types are also correct (F1 score for this is not correct right now).

## Main script to run

Main command is: python get_metrics.py

To get the results of the best variations, output_files/no_specifics.csv should be the input file (bias_results variable). Make sure to specify the eval_columns that you want to evaluate. Otherwise, any exported csv coming from orq.ai platform should work unless some responses don't use the JSON format at all.

Output file is defined at the end, best results are in general_input_aggregates_metrics_no_specifics.csv currently
