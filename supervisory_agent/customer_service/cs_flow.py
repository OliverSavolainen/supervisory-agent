from typing import List
from crewai.flow.flow import Flow, listen, start,or_, router
from pydantic import BaseModel
from evaluation_crew import EvaluationCrew
from crew import CustomerServiceCrew
import json


desc = """Address customer queries related to using the generative AI platform. Respond clearly and concisely, solving the issue or providing next steps. If the question is unclear, ask follow-up questions. 

  CUSTOMER QUERY TYPES
  ---------------------
  Common issues include account access, technical performance, billing, data privacy, and output quality. If the query isn’t relevant to platform use (e.g., code requests unrelated to AI settings), politely clarify the limitations of support.

  FAQ REFERENCE
  -------------
  - **Account Access**: Suggest password reset or account recovery.
  - **Output Quality**: Recommend prompt adjustments or best practices.
  - **Technical Issues**: Advise clearing cache, checking internet, or restarting.
  - **Subscription/Billing**: Guide to account settings or billing support.
  - **Data Privacy**: Reassure with data security practices and link to the privacy policy.

  Use a professional, empathetic tone, and integrate any available feedback to improve response quality. This is the query: 
"""


class TaskState(BaseModel):
    approved: bool = False
    last_feedback:str = ""
    current_output: str = ""
    customer_query: str = "I’m considering upgrading my subscription, but I want to understand if there are options for customizing the AI model's behavior beyond what I see in the standard settings. Could you provide details?"

class CustomerQueryFlow(Flow[TaskState]):
    initial_state = TaskState

    @start()
    def start_flow(self):
        print("Starting Task")
    
    @listen(or_(start_flow,"improve_based_on_feedback"))
    async def handle_task(self):
        customer_service_crew = CustomerServiceCrew().crew()
        self.state.current_output = str(await customer_service_crew.kickoff_async(inputs={
            "customer_query" : self.state.customer_query,
            "feedback": self.state.last_feedback,
            "last_answer": self.state.current_output
        }))

    @router(handle_task)
    async def evaluate_task(self):
        if len(str(self.state.current_output)) > 0:
            
            evaluation = await EvaluationCrew().crew().kickoff_async(inputs={"task_description": desc + self.state.customer_query,"last_feedback": self.state.last_feedback,"last_answer":self.state.current_output})
            evaluation = evaluation.json_dict

            self.state.approved = evaluation.get("approved", False)
            self.state.last_feedback = evaluation.get("feedback", "")
            print("Evaluation Completed")
        else:
            print("No customer queries to evaluate.")

        if self.state.approved:
            exit()
        else:
            return "improve_based_on_feedback"


        print("Task Completed")