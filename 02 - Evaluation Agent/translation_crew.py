import os
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai import LLM
from document_loader_tool import file_loader_tool
from file_writer_tool import file_writer_tool
import openai
from openai import OpenAI
import os

from pydantic import BaseModel

openai.api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(
    api_key = openai.api_key,
)
GPT_MODEL = LLM(model="gpt-4o-mini", temperature=0.2)

class TranslationOutput(BaseModel):
    feedback_consideration: bool
    final_answer: str

@CrewBase
class TranslationCrew:

    agents_config = 'config/tr_agents.yaml'
    tasks_config = 'config/tr_tasks.yaml'
    source_language: str = "English"
    target_language: str = "Dutch"
    cultural_regional_context: str = "Use extremely formal words but keep the meaning the same."
    glossary: dict = {"MPA": "NDA", "Stage 3": "Derde stage", "Komt niet voor": "in de text"}
    file_path: str = r"text_to_translate.txt"

    def __init__(self, source_language=None, target_language=None, cultural_regional_context=None, glossary=None, file_path=None):
        if source_language:
            self.source_language = source_language
        if target_language:
            self.target_language = target_language
        if cultural_regional_context:
            self.cultural_regional_context = cultural_regional_context
        if glossary:
            self.glossary = glossary
        if file_path:
            self.file_path = file_path

        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.file_path = os.path.join(script_dir, self.file_path)

        self.file_name = f"{os.path.splitext(self.file_path)[0]}_TRANSLATED_TO_{self.target_language}{os.path.splitext(self.file_path)[1]}"
        self.file_write_path = os.path.join(script_dir, self.file_name)

    ## AGENTS
    @agent
    def file_reader_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['file_reader_agent'],
            verbose=True,
            memory=False,
            tools=[file_loader_tool],
            llm=GPT_MODEL
        )

    @agent
    def translator_agent(self) -> Agent:
        return Agent(
            role="Translator",
            config=self.agents_config['translator_agent'],
            verbose=True,
            memory=False,
            llm=GPT_MODEL
        )

    @agent
    def file_writer_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['file_writer_agent'],
            verbose=True,
            memory=False,
            tools=[file_writer_tool],
            llm=GPT_MODEL
        )

    ## TASKS
    @task
    def read_file_task(self) -> Task:
        return Task(
            config=self.tasks_config['read_file_task'],
            #tools=[file_loader_tool],
        )

    @task
    def translate_task(self) -> Task:
        return Task(
            config=self.tasks_config['translate_task'],
            output_json=TranslationOutput
        )

    @task
    def write_file_task(self) -> Task:
        return Task(
            config=self.tasks_config['write_file_task'],
            #tools=[file_writer_tool],
        )

    ## Helper Function
    def replace_words(self, text: str) -> str:
        replaced = ""
        for old_word, new_word in self.glossary.items():
            if old_word in text:
                text = text.replace(old_word, new_word)
                replaced += f"Replaced {old_word} with {new_word}. "
        return text, replaced
    @crew
    def read_file_crew(self) -> Crew:
        return Crew(
            agents=[self.file_reader_agent()],
            tasks=[self.read_file_task()],
            process=Process.sequential,
            memory=True
        )

    @crew
    def translate_crew(self) -> Crew:
        return Crew(
            agents=[self.translator_agent()],
            tasks=[self.translate_task()],
            process=Process.sequential,
            memory=True,
            output_json=TranslationOutput
        )

    @crew
    def write_file_crew(self) -> Crew:
        return Crew(
            agents=[self.file_writer_agent()],
            tasks=[self.write_file_task()],
            process=Process.sequential,
            memory=True
        )