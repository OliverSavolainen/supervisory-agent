## Evaluation Agent

Evaluation Agent is initialized to evaluate any kind of response based on the task description. It has its own evaluation task in config/general_eval_task.yaml, which has a variable {last_answer} as the last task output from the crew. Based on this and instructions from the task description, it will always respond with a JSON with keys "approved" and "feedback". If the approved value is True, the Flow will end. If the value is False, the crew will look to improve on the response {last_answer} based on {feedback}. The crew should always have an instruction to reflect on how it's using the feedback.

Evaluation/Supervisory Agent includes the Evaluator Selector, which chooses between available LLM-as-judge prompts from config/evaluators.yaml (Only Discrimination is there right now, Translation and Tone of Voice are in a previous commit). Then this agent will also have a task to create valid input texts for the chosen evaluators.

## Main script to run

Main commands: python discrimination/bias_flow.py for the discrimination dataset results. Towards the end the input dataset (currently agreements.csv) and output file can be defined. All the resulting files and other evaluation scripts are also in the discrimination folder.

python translation/main_translation.py for the translation flow. Inside translation_flow.py source_language, target_language and source_text should be specified.