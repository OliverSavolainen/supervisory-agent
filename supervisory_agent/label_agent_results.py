import pandas as pd
import tkinter as tk
from tkinter import messagebox
from tkinter.ttk import Checkbutton

# Load the dataset
data = pd.read_csv("evaluation_results_tool.csv")

data = data[data["loop_number"] > 1]

# Prepare to store labeled data
labeled_data = []

# Labels
bias_types = ["Race", "Gender", "Religion", "Disability", "Sexual", "No Bias"]
improvement_labels = ["Improved", "Not Improved"]

# GUI logic
class LabelingApp:
    def __init__(self, master, dataset):
        self.master = master
        self.master.title("Label Evaluation Results")
        self.dataset = dataset
        self.index = 0  # Current row index
        self.labels = {bias: tk.IntVar() for bias in bias_types}  # Checkbox states
        self.improved_label = tk.StringVar()  # Improved/Not Improved label

        # Layout
        self.prompt_label = tk.Label(master, text="Prompt:")
        self.prompt_label.pack()
        self.prompt_text = tk.Text(master, height=5, wrap=tk.WORD)
        self.prompt_text.pack()

        self.response_label = tk.Label(master, text="Response (First Output):")
        self.response_label.pack()
        self.response_text = tk.Text(master, height=5, wrap=tk.WORD)
        self.response_text.pack()

        self.final_output_label = tk.Label(master, text="Final Output:")
        self.final_output_label.pack()
        self.final_output_text = tk.Text(master, height=5, wrap=tk.WORD)
        self.final_output_text.pack()

        self.checkboxes = []
        for bias in bias_types:
            checkbox = Checkbutton(master, text=bias, variable=self.labels[bias])
            checkbox.pack(anchor="w")
            self.checkboxes.append(checkbox)

        # Buttons for Improvement
        self.improved_button = tk.Button(master, text="Improved", command=self.set_improved, state="normal")
        self.improved_button.pack(side=tk.LEFT)

        self.not_improved_button = tk.Button(master, text="Not Improved", command=self.set_not_improved, state="normal")
        self.not_improved_button.pack(side=tk.LEFT)

        # Navigation Buttons
        self.next_button = tk.Button(master, text="Next", command=self.save_and_next)
        self.next_button.pack()

        # Load the first row
        self.load_row()

    def load_row(self):
        if self.index >= len(self.dataset):
            messagebox.showinfo("Done", "You have labeled all the rows!")
            self.master.quit()
            return

        row = self.dataset.iloc[self.index]
        self.prompt_text.delete("1.0", tk.END)
        self.prompt_text.insert(tk.END, row["prompt"])

        self.response_text.delete("1.0", tk.END)
        self.response_text.insert(tk.END, row["normalized_llm_response"])

        self.final_output_text.delete("1.0", tk.END)
        self.final_output_text.insert(tk.END, row["final_output"])

        # Reset checkboxes and improvement label
        for bias in bias_types:
            self.labels[bias].set(0)
        self.improved_label.set("")

    def save_and_next(self):
        # Save the current labels
        row = self.dataset.iloc[self.index]
        current_labels = {bias: self.labels[bias].get() for bias in bias_types}
        labeled_data.append({
            "prompt": row["prompt"],
            "normalized_llm_response": row["normalized_llm_response"],
            "final_output": row["final_output"],
            "labels": [bias for bias, value in current_labels.items() if value == 1],
            "improvement": self.improved_label.get()
        })

        # Move to the next row
        self.index += 1
        self.load_row()

    def set_improved(self):
        self.improved_label.set("Improved")

    def set_not_improved(self):
        self.improved_label.set("Not Improved")

# Run the GUI
root = tk.Tk()
app = LabelingApp(root, data)
root.mainloop()

# Save labeled data to CSV
output_df = pd.DataFrame(labeled_data)
output_df.to_csv("labeled_evaluation_results.csv", index=False)

print("Labeled data saved to 'labeled_evaluation_results.csv'")
