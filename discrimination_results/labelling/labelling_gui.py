import pandas as pd
import tkinter as tk
from tkinter import ttk

# Load the dataset
file_path = 'normalized_prompt_responses.csv'  # Replace with your actual file path
df = pd.read_csv(file_path)

# Randomize the dataset order
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Add a label column if not already present
if "label" not in df.columns:
    df["label"] = ""

# Initialize index to track progress
current_index = 0

# Function to save progress
def save_progress():
    df.to_csv('labeled_responses2.csv', index=False)
    status_label.config(text="Progress saved!")

# Function to add a label to the current response
def add_label(label):
    global current_index
    existing_labels = df.at[current_index, "label"]
    if pd.isna(existing_labels) or existing_labels == "":
        df.at[current_index, "label"] = label
    else:
        labels_set = set(existing_labels.split(", "))
        labels_set.add(label)
        df.at[current_index, "label"] = ", ".join(sorted(labels_set))  # Sort for consistency
    update_display()

# Function to move to the next response
def next_response():
    global current_index
    if current_index < len(df) - 1:
        current_index += 1
        update_display()
    else:
        status_label.config(text="All responses labeled!")

# Function to move to the previous response
def previous_response():
    global current_index
    if current_index > 0:
        current_index -= 1
        update_display()

# Update the display with the current prompt, response, and labels
def update_display():
    prompt_text.set(f"Prompt: {df.iloc[current_index]['prompt']}")
    response_text.set(f"Response: {df.iloc[current_index]['normalized_llm_response']}")
    current_label.set(f"Current Labels: {df.iloc[current_index]['label']}")

# Create the main GUI window
root = tk.Tk()
root.title("Bias Type Labeling Tool")

# Create string variables for the prompt, response, and status
prompt_text = tk.StringVar()
response_text = tk.StringVar()
current_label = tk.StringVar()

# Display prompt and response
tk.Label(root, textvariable=prompt_text, wraplength=600, justify="left").pack(pady=10)
tk.Label(root, textvariable=response_text, wraplength=600, justify="left", fg="blue").pack(pady=10)
tk.Label(root, textvariable=current_label).pack(pady=10)

# Create buttons for labeling
buttons_frame = tk.Frame(root)
buttons_frame.pack(pady=20)

# Bias type labels
labels = ["Race", "Gender", "Religion", "Disability", "Sexual", "No Bias"]
for lbl in labels:
    tk.Button(buttons_frame, text=lbl, command=lambda l=lbl: add_label(l), width=20).pack(side="left", padx=5)

# Navigation buttons
navigation_frame = tk.Frame(root)
navigation_frame.pack(pady=20)
tk.Button(navigation_frame, text="Previous", command=previous_response, width=20).pack(side="left", padx=5)
tk.Button(navigation_frame, text="Next", command=next_response, width=20).pack(side="left", padx=5)

# Save progress button
save_button = tk.Button(root, text="Save Progress", command=save_progress)
save_button.pack(pady=10)

# Status label
status_label = tk.Label(root, text="")
status_label.pack(pady=10)

# Initialize the display
update_display()

# Run the Tkinter main loop
root.mainloop()
