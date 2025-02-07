from typing import List, Dict
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
#from llm_judge_tool import llm_as_a_judge_tool
from llm_judge import llm_as_a_judge
from crewai import LLM
from pydantic import BaseModel
import openai
import os
import yaml

# Set up OpenAI key and model
openai.api_key = os.getenv("OPENAI_API_KEY")
GPT_MODEL = LLM(model="gpt-4o-mini", temperature=0)

# Define output format
class EvaluationOutput(BaseModel):
    feedback: str
    approved: bool

class SelectionOutput(BaseModel):
    selected_evaluators: List[str]
    inputs: List[str]

class BestAnswerOutput(BaseModel):
    rationale: str
    final_decision: str

@CrewBase
class EvaluationCrew:
    agents_config = 'config/eval_agents.yaml'
    tasks_config = 'config/general_eval_task.yaml'
    evaluators_config = 'config/evaluators.yaml'

    def __init__(self):
        #self.judge_tool = llm_as_a_judge_tool
        self.available_evaluators = self.load_evaluators_from_yaml()
        self.selected_evaluators: List[Dict] = []
        self.evaluator_feedback: List[str] = []

    def load_evaluators_from_yaml(self) -> Dict[str, Dict]:
        """Load evaluators from the YAML configuration file."""
        with open(self.evaluators_config, 'r', encoding='utf-8') as file:
            evaluator_configs = yaml.safe_load(file)
        
        evaluators = {}
        for key, config in evaluator_configs.items():
            evaluators[key] = config
        return evaluators

    @agent
    def evaluation_selection_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['evaluator_selection_agent'],
            verbose=True,
            memory=False
        )

    @task
    def select_evaluators_for_task(self) -> Task:
        task_config = self.tasks_config['select_evaluators_for_task']

        evaluator_options = "\n".join(
            [f"- {evaluator['name']}: {evaluator['description']}" for evaluator in self.available_evaluators.values()]
        )
        task_config["description"] = task_config["description"].replace("{evaluator_options}", evaluator_options)

        return Task(
            config=task_config,
            agent=self.evaluation_selection_agent()
        )

    @agent
    def evaluation_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['evaluation_agent'],
            verbose=True,
            memory=False
        )

    @task
    def evaluate_task_response(self) -> Task:

        return Task(
            config=self.tasks_config['evaluate_task_response'],
            agent=self.evaluation_agent(),
            output_json=EvaluationOutput
        )
    @agent
    def best_answer_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["best_answer_agent"],
            verbose=True,
            memory=False
        )

    @task
    def select_best_answer_task(self) -> Task:
        """
        Task to select the best answer after evaluating all iterations.
        """
        return Task(
            config=self.tasks_config["select_best_answer_task"],
            agent=self.best_answer_agent()
        )

    @crew
    def select_best_answer_crew(self) -> Crew:
        """
        Crew to run the best answer selection task.
        """
        return Crew(
            agents=[self.best_answer_agent()],
            tasks=[self.select_best_answer_task()],
            process=Process.sequential,
            memory=True,
            output_json=BestAnswerOutput
        )

    @task
    def create_inputs_for_evaluators(self) -> Task:
        task_config = self.tasks_config['create_inputs_for_evaluators']

        # Build the evaluator options and prompt templates section
        selected_evaluators_section = "\n".join([evaluator['name'] for evaluator in self.selected_evaluators])
        prompt_templates_section = "\n".join(
            [f"- {evaluator['name']}: {evaluator['prompt_template']}" for evaluator in self.selected_evaluators]
        )

        # Replace placeholders in the task description
        task_config["description"] = task_config["description"].replace("{selected_evaluators}", selected_evaluators_section)
        task_config["description"] = task_config["description"].replace("{prompt_templates}", prompt_templates_section)

        return Task(
            config=task_config,
            agent=self.evaluation_selection_agent(),
            output_json=SelectionOutput
            
        )

    @crew
    def evaluator_selection_crew(self) -> Crew:
        return Crew(
            agents=[self.evaluation_selection_agent()],
            tasks=[self.select_evaluators_for_task(), self.create_inputs_for_evaluators()],
            process=Process.sequential,
            memory=True,
            output_json=SelectionOutput
        )

    @crew
    def task_evaluation_crew(self) -> Crew:
        return Crew(
            agents=[self.evaluation_agent()],
            tasks=[self.evaluate_task_response()],
            process=Process.sequential,
            memory=True,
            output_json=EvaluationOutput
        )

    async def gather_evaluator_feedback(self, input_text: List[str], output_text: str, selected_evaluators: list):
        """
        Gather feedback from selected evaluators using the LLM-as-a-Judge tool.

        Parameters:
        - input_text: List of inputs derived from the evaluator selection process.
        - output_text: The generated output text.
        - selected_evaluators: List of selected evaluator names.
        """
        self.evaluator_feedback = []

        for evaluator_name, evaluator_input in zip(selected_evaluators, input_text):
            if evaluator_name not in [evaluator['name'] for evaluator in self.available_evaluators.values()]:
                print(f"Warning: Evaluator {evaluator_name} not found in available evaluators.")
                continue

            evaluator = next(
                            (evaluator for evaluator in self.available_evaluators.values() if evaluator['name'] == evaluator_name),
                            None
                        )

            response = llm_as_a_judge(
                text=evaluator_input,
                output_text=output_text,
                judge_name=evaluator_name,
                prompt_template=evaluator["prompt_template"]
            )
            print(f"Response from {evaluator_name}:", response)

            # Format feedback and add to evaluator feedback list
            formatted_feedback = f"Evaluator: {evaluator_name}\nExplanation: {response.get('explanation', 'No explanation')}\nScore: {response.get('score', 'N/A')}"
            self.evaluator_feedback.append(formatted_feedback)