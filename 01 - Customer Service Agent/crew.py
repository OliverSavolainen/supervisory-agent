from crewai import Agent, Crew, Process, Task, Flow
from crewai.project import CrewBase, agent, crew

import openai
from openai import OpenAI
from langchain_openai import ChatOpenAI

import os
openai.api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(
    api_key = openai.api_key,
)
GPT_MODEL = "gpt-4o-mini"



# manager_llm
manager_llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, openai_api_key=openai.api_key)

class EvaluationCrew:
    @agent
    def evaluation_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['evaluation_agent'],
            verbose=True
        )

    @crew
    def crew(self) -> Crew:
        """Creates the CustomerServiceCrew"""
        return Crew(
            agents=self.agents,  
            process=Process.hierarchical,
            manager_llm = manager_llm,
            verbose=True, 
        )
    

@CrewBase
class CustomerServiceCrew:
    """Customer Service crew"""
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def accounting_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['accounting_agent'],
            allow_delegation=False,
            verbose=True,
            llm=GPT_MODEL
        )
    
    @agent
    def technical_support_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['technical_support_agent'],
            allow_delegation=False,
            verbose=True,
            llm=GPT_MODEL
        )
    
    @agent
    def marketing_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['marketing_agent'],
            allow_delegation=False,
            verbose=True,
            llm=GPT_MODEL
        )
    
    @agent
    def human_resource_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['human_resource_agent'],
            allow_delegation=False,
            verbose=True,
            llm=GPT_MODEL
        )
    @agent
    def evaluation_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['evaluation_agent'],
            allow_delegation=False,
            verbose=True,
            llm=GPT_MODEL
        )
    
    # manager_agent
    manager = Agent(
            role ="Project Manager",
            goal=" Efficiently manage the crew and ensure high-quality task completion",
            backstory=" You're an experienced project manager, skilled in overseeing complex projects and guiding teams to success. Your role is to coordinate the efforts of the crew members, ensuring that each task is completed on time and to the highest standard.",
            allow_delegation=True,
            verbose=True
        )

    @crew
    def crew(self) -> Crew:
        """Creates the CustomerServiceCrew"""
        return Crew(
            agents=self.agents,  
            process=Process.hierarchical,
            # manager_agent=self.manager,
            manager_llm = manager_llm,
            verbose=True, 
        )