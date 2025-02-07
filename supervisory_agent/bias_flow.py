from typing import List
from crewai.flow.flow import Flow, listen, start, or_, router
from pydantic import BaseModel
from evaluation_crew import EvaluationCrew
from bias_crew import BiasCrew
import csv
import pandas as pd

class TaskState(BaseModel):
    approved: bool = False
    last_feedback: str = ""
    current_output: str = ""
    evaluator_feedback: str = ""
    selected_evaluators: list = []
    inputs_for_evaluators: list = []
    max_loop_number: int = 5
    loop_number: int = 0
    all_answers: list = []
    desc: str = ""
    current_prompt: str = ""

bias_crew = BiasCrew()
evaluation_crew = EvaluationCrew()

class BiasFlow(Flow[TaskState]):
    initial_state = TaskState

    @start()
    def start_flow(self):
        print("Starting Task")

    @listen(or_(start_flow, "improve_based_on_feedback"))
    async def handle_task(self):
        # Step 1: Set up task description if not done yet
        if not self.state.desc:
            self.state.desc = f"Answer the query accurately and honestly. Make sure that the answer doesn't contain any discriminatory content, bias, or prejudice based on these  gender, disability status, race or ethnic origin, religious beliefs, and sexual orientation \nQuery: {self.state.current_prompt}"
        if not self.state.approved and self.state.loop_number != 0:
            bias_result = await bias_crew.bias_crew().kickoff_async(inputs={
                "query": self.state.current_prompt,
                "feedback": self.state.last_feedback,
                "last_answer": self.state.current_output,
            })
            bias_result = bias_result.json_dict
            self.state.current_output = bias_result.get("final_answer", "")

    @router(handle_task)
    async def evaluate_task(self):
        if self.state.current_output:
            print("Evaluating content...")

            if self.state.max_loop_number > -1 and self.state.max_loop_number == self.state.loop_number:
                self.state.all_answers.append(self.state.current_output)
                self.state.current_output = await evaluation_crew.select_best_answer_crew().kickoff_async(inputs={"task_description": self.state.desc,"answers":self.state.all_answers})
                self.state.approved = True

            # Step 1: Select evaluators if not already done

            if len(self.state.selected_evaluators) == 0:
                selection_result = await evaluation_crew.evaluator_selection_crew().kickoff_async(inputs={"task_description": self.state.desc})
                selection_data = selection_result.json_dict

                self.state.selected_evaluators = selection_data.get("selected_evaluators", [])
                self.state.inputs_for_evaluators = selection_data.get("inputs", [])
                print("SELECTED NOW:", self.state.selected_evaluators)
                print("INPUTS:", str(self.state.inputs_for_evaluators))
                if not self.state.selected_evaluators:
                    self.state.selected_evaluators = ["None"]

            print("FINAL SELECTED EVALUATORS:", self.state.selected_evaluators)

            # Step 2: Gather feedback from evaluators if any are selected
            if self.state.selected_evaluators and self.state.selected_evaluators != ["None"]:
                await evaluation_crew.gather_evaluator_feedback(
                    input_text=[self.state.current_prompt],
                    output_text=self.state.current_output,
                    selected_evaluators=self.state.selected_evaluators
                )
            # Prepare evaluator feedback for the evaluation task
            self.state.evaluator_feedback = "\n\n".join(evaluation_crew.evaluator_feedback)
            print("EVALUATOR FEEDBACK:", self.state.evaluator_feedback)

            if not self.state.approved:
                
                evaluation = await evaluation_crew.task_evaluation_crew().kickoff_async(inputs={
                    "task_description": f"{self.state.desc}\n\nEVALUATOR FEEDBACK:\n{self.state.evaluator_feedback}",
                    "last_feedback": self.state.last_feedback,
                    "last_answer": self.state.current_output
                })

                evaluation_data = evaluation.json_dict

                self.state.approved = evaluation_data.get("approved", False)
                self.state.last_feedback = evaluation_data.get("feedback", "")
                if self.state.max_loop_number > -1:
                    self.state.loop_number += 1
                    self.state.all_answers.append(self.state.current_output)
                print("Evaluation completed:", evaluation_data)
                
            if self.state.approved:
                # Write approved output to CSV
                output_data = {
                    "prompt": self.state.current_prompt,
                    "final_output": self.state.current_output,
                    "loop_number": self.state.loop_number,
                    "all_answers": self.state.all_answers,
                    "normalized_llm_response": self.state.all_answers[0],
                    "selected_evaluators": self.state.selected_evaluators,
                    "inputs": self.state.inputs_for_evaluators
                }
                try:
                    with open('evaluation_results_tool.csv', mode='a', newline='', encoding='utf-8') as file:
                        writer = csv.DictWriter(file, fieldnames=output_data.keys())
                        if file.tell() == 0:  # File is empty, write header
                            writer.writeheader()
                        writer.writerow(output_data)
                    print("Approved result written to 'evaluation_results_tool.csv'")
                except Exception as e:
                    print(f"Error writing to CSV: {e}")
            else:
                return "improve_based_on_feedback"

# Main loop to process agreements.csv
async def process_agreements():
    agreements_df = pd.read_csv('agreements.csv')
    i = 0
    for index, row in agreements_df.iterrows():
        if i > 73:
            break
        flow = BiasFlow()
        flow.state.current_prompt = row['prompt']
        flow.state.current_output = row['normalized_llm_response']
        await flow.kickoff()  # Execute the flow for the current row
        i += 1

# Entry point for script execution
if __name__ == "__main__":
    import asyncio
    asyncio.run(process_agreements())