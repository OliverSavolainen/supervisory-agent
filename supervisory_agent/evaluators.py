import openai
from typing import Optional, Dict
from pydantic import BaseModel

from pydantic import BaseModel

class Evaluator(BaseModel):
    name: str
    description: str
    prompt_template: str

    def generate_prompt(self, input_text: str, output_text: Optional[str] = None) -> str:
        """
        Generates a prompt by substituting placeholders with the given input and output texts.

        Parameters:
        - input_text: The original or reference text that provides context for evaluation.
        - output_text: The generated output text to be evaluated (if applicable).

        Returns:
        - A formatted prompt string ready to be used in the LLM tool.
        """
        prompt = self.prompt_template.replace("{{log.input}}", input_text)
        if output_text:
            prompt = prompt.replace("{{log.output}}", output_text)
        return prompt