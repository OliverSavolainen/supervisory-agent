from typing import List
from crewai.flow.flow import Flow, listen, start
from pydantic import BaseModel
from crew import CustomerServiceCrew

# State model to track customer queries and evaluations
class TaskState(BaseModel):
    approved: bool = False
    last_feedback:str = ""
    current_output: str = "" 

class CustomerQueryFlow(Flow[TaskState]):
    initial_state = TaskState

    @start()
    def handle_task(self):
        result = CustomerServiceCrew().crew().kickoff()
        


    @listen(handle_customer_query)
    def evaluate_customer_queries(self):
        print("Evaluating Customer Queries")
        if len(self.state.customer_queries) > 0:
            customer_service_crew = CustomerServiceCrew()
            results = customer_service_crew.crew().kickoff(
                inputs={"queries": self.state.customer_queries}
            )
            evaluated = [{"query": q, "approved": True} for q in self.state.customer_queries]
            self.state.evaluated_queries = evaluated
            self.state.customer_queries = []  # Clear queries after evaluation

        print("Evaluation Completed")
        print(f"Evaluated Queries: {self.state.evaluated_queries}")
