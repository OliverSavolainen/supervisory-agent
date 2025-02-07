import os
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai import LLM
import openai
from openai import OpenAI
import os
from pydantic import BaseModel


openai.api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(
    api_key = openai.api_key,
)
GPT_MODEL = LLM(model="gpt-4o-mini", temperature=0.0)

class BiasOutput(BaseModel):
    feedback_consideration: bool
    final_answer: str


@CrewBase
class BiasCrew:
    agents_config = 'config/bias_agents.yaml'
    tasks_config = 'config/bias_tasks.yaml'
    ## AGENTS
    @agent
    def bias_agent(self) -> Agent:
        return Agent(
            role="Query Answerer",
            config=self.agents_config['answer_agent'],
            verbose=True,
            llm=GPT_MODEL
        )

    @task
    def bias_task(self) -> Task:
        return Task(
            config=self.tasks_config['answer_task'],
            output_json=BiasOutput
        )

    @crew
    def bias_crew(self) -> Crew:
        return Crew(
            agents=[self.bias_agent()],
            tasks=[self.bias_task()],
            process=Process.sequential,
            output_json=BiasOutput
        )