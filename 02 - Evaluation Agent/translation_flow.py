from typing import List
from crewai.flow.flow import Flow, listen, start, or_, router
from pydantic import BaseModel
from evaluation_crew import EvaluationCrew
from translation_crew import TranslationCrew
import csv

class TaskState(BaseModel):
    approved: bool = False
    last_feedback: str = ""
    current_output: str = ""
    source_language: str = "English"
    target_language: str = "Estonian"
    cultural_regional_context: str = ""
    reference: str = 'Is everything alright? You have been acting a little suspiciously lately'
    source_text: str = "With Orq, teams can ideate and test Generative AI solutions, run them safely to production, and monitor performance for continuous improvement."
    glossary_replacements: str = ""
    first_translation: str = ""
    desc: str = ""
    evaluator_feedback: str = ""
    selected_evaluators: list = []
    inputs_for_evaluators: list = []
    max_loop_number: int = 5
    loop_number: int = 0
    all_answers: list = []


translation_crew = TranslationCrew()
evaluation_crew = EvaluationCrew()

class TranslationFlow(Flow[TaskState]):
    initial_state = TaskState


    @start()
    def start_flow(self):
        print("Starting Task")

    @listen(or_(start_flow, "improve_based_on_feedback"))
    async def handle_task(self):
        
        # Step 1: Set up task description if not done yet
        if not self.state.desc:
            print("Setting up translation description...")
            read_result = await translation_crew.read_file_crew().kickoff_async(inputs={"file_path": translation_crew.file_path})
            print("File content loaded:", read_result)
            
            # Replace words based on glossary
            self.state.source_text, self.state.glossary_replacements = translation_crew.replace_words(self.state.source_text)
            
            # Set up description for the translation task
            self.state.desc = f"Translate the text from {self.state.source_language} to {self.state.target_language} based on specified guidelines. Ensure formality and cultural relevance.\nText: {self.state.source_text}"
            
            if self.state.glossary_replacements:
                self.state.desc += f"\nGlossary replacements: {self.state.glossary_replacements}"
            
            if self.state.cultural_regional_context:
                self.state.desc += f"\nContext/guidelines: {self.state.cultural_regional_context}"

            print("Description set up completed.")

        # Step 2: Perform translation if not approved yet
        if not self.state.approved:
            print("Translating content...")
            translate_result = await translation_crew.translate_crew().kickoff_async(inputs={
                "text": self.state.source_text,
                "source_language": self.state.source_language,
                "target_language": self.state.target_language,
                "feedback": self.state.last_feedback,
                "last_answer": self.state.current_output,
                "cultural_regional_context": self.state.cultural_regional_context,
                "glossary_replacements": self.state.glossary_replacements
            })
            translate_result = translate_result.json_dict
            
            # Save first translation if it’s the first time translating
            if not self.state.current_output:
                self.state.first_translation = translate_result.get("final_answer", "")
            
            self.state.current_output = translate_result.get("final_answer", "")
            print("Translation completed:", self.state.current_output)

        # Step 3: Write to file and exit if translation is approved
        if self.state.approved:
            exit()
            print("Writing translated content to file...")
            await translation_crew.write_file_crew().kickoff_async(inputs={"content": self.state.current_output, "file_name": translation_crew.file_path})
            
            # Append translations to CSV file
            with open("translation_output.csv", mode="a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                
                # Write headers if the file is empty
                if file.tell() == 0:
                    writer.writerow(["Source Text", "First Translation", "Final Translation", "Reference", "Source Language", "Target Language"])
                
                # Write row with translations
                writer.writerow([self.state.source_text, self.state.first_translation, self.state.current_output, self.state.reference, self.state.source_language, self.state.target_language])
            
            print("File writing completed.")
            exit()

    @router(handle_task)
    async def evaluate_task(self):
        if self.state.current_output:
            print("Evaluating translated content...")
            

            if self.state.max_loop_number > -1 and self.state.max_loop_number == self.state.loop_number:
                self.state.all_answers.append(self.state.current_output)
                self.state.current_output = await evaluation_crew.select_best_answer_crew().kickoff_async(inputs={"task_description": self.state.desc,"answers":self.state.all_answers})
                self.state.approved = True
                return "improve_based_on_feedback"

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
            if self.state.selected_evaluators != ["None"] and self.state.selected_evaluators:
                await evaluation_crew.gather_evaluator_feedback(
                    input_text=self.state.inputs_for_evaluators,
                    output_text=self.state.current_output,
                    selected_evaluators=self.state.selected_evaluators
                )
            # Prepare evaluator feedback for the evaluation task
            self.state.evaluator_feedback = "\n\n".join(evaluation_crew.evaluator_feedback)
            print("EVALUATOR FEEDBACK:", self.state.evaluator_feedback)

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
        return "improve_based_on_feedback"