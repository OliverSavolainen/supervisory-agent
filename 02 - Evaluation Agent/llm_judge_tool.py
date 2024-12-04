import os
import openai
from crewai_tools import tool
import json

openai.api_key = os.getenv("OPENAI_API_KEY")

client = openai.OpenAI()

@tool("LLM-as-a-Judge")
def llm_as_a_judge_tool(text: str = "", output_text: str = "", judge_name: str = "", prompt_template: str = "") -> dict:
    """
    Useful for assessing text quality, tone, translation accuracy, etc., using specific evaluators (judges).
    Judges return a JSON object with an explanation and a score.

    Arguments:
    - text: The input text or original content to evaluate.
    - output_text: The generated output text to be evaluated by the judge.
    - judge_name: The name of the specific evaluator (e.g., "Tone of Voice", "Translation Accuracy").
    - prompt_template: The prompt template for the LLM, with placeholders for `log_input` and `log_output`.

    Example:
    llm_as_judge_tool(
        text="The original text.",
        output_text="The translated text.",
        judge_name="Translation Accuracy",
        prompt_template="Evaluate the following text..."
    )
    """
    print(prompt_template)
    # Prepare the prompt by inserting variables
    prompt = prompt_template.format(
        log_input=text,
        log_output=output_text
    )
    print(prompt)
    # Call the LLM API with the prompt
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # replace with actual model as required
        messages=[{"role": "user", "content": prompt}],
        response_format={
        "type": "json_schema",
        "json_schema": {
        "name": "eval",
        "schema": {
            "type": "object",
            "properties": {
                "explanation": {
                    "description": "The explanation for the evaluation result.",
                    "type": "string"
                },
                "score": {
                    "description": "The score associated with the evaluation.",
                    "type": "integer"
                }
            },
            "additionalProperties": False
        }
        }
    }
        max_tokens=500,
        temperature=0.0
    ).choices[0].message.content()
    print(response)
    exit()

    response_data = {"explanation": "", "score": ""}

    # Assuming the model returns a valid JSON format as output
    try:
        # Parse JSON directly if the model output is a JSON string
        response_data = json.loads(response_text)  # Safe JSON parsing
    except json.JSONDecodeError as e:
        # Fallback if JSON parsing fails, add error message as explanation
        response_data["explanation"] = f"Error parsing response: {e}. Raw output: {response_text}"
        response_data["score"] = "N/A"

    return response_data