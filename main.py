import os
import subprocess
import shutil
import pandas as pd
from datetime import datetime
import glob

source_folder = 'Data/'
file_name = 'anki_deck.apkg'
script_dir = os.path.dirname(os.path.abspath(__file__))

source_file = os.path.join(source_folder, file_name)
destination_file = os.path.join(script_dir, file_name)
cards_copy_folder = os.path.join(script_dir, 'cards_for_merging')

if os.path.exists(source_file):
    shutil.copyfile(source_file, destination_file)
    print(f"File {file_name} has been copied from {source_folder} to {script_dir} and overwritten.")
else:
    print(f"Source file {source_file} does not exist.")

def find_pdfs_in_script_folder():
    pdf_files = glob.glob(os.path.join(script_dir, '*.pdf'))
    if len(pdf_files) == 0:
        raise FileNotFoundError(f"No PDF files found in {script_dir}.")
    return pdf_files


pdf_files = find_pdfs_in_script_folder()

for pdf_file in pdf_files:
    pdf_name = os.path.splitext(os.path.basename(pdf_file))[0]

    script_file_pairs = [
        ("Scripts/make_learning_objectives.py", pdf_file),
        ("Scripts/select_cards.py", "Data/anki_embeddings.csv", f"{pdf_name}_learning_objectives.csv"),
    ]

    for pair in script_file_pairs:
        script = pair[0]
        files = pair[1:]

        # Ensure the necessary input files exist before running the script
        for file in files:
            if not os.path.exists(file):
                print(f"Required file {file} not found, skipping {script}.")
                continue

        command = ["python3", "-u", script] + list(files)

        print(f"Running {script} with {files}")
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        with process.stdout:
            for line in iter(process.stdout.readline, ''):
                print(line, end='', flush=True)
        with process.stderr:
            for line in iter(process.stderr.readline, ''):
                print(line, end='', flush=True)
        process.wait()
        print(f"{script} finished with exit code {process.returncode}\n", flush=True)

    cards_csv = f"{pdf_name}_cards.csv"
    if os.path.exists(cards_csv):
        shutil.copy(cards_csv, os.path.join(cards_copy_folder, cards_csv))
        print(f"Copied {cards_csv} to {cards_copy_folder} for merging.")


    # Move files only after all scripts are successfully run
    def move_files_to_new_folder(files_to_move, subfolder_path):
        new_folder_name = f"{pdf_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        new_folder_path = os.path.join(subfolder_path, new_folder_name)
        os.makedirs(new_folder_path, exist_ok=True)
        for file_path in files_to_move:
            if os.path.isfile(file_path):
                shutil.move(file_path, new_folder_path)
            else:
                print(f"File not found: {file_path}")

        print(f"Files moved to {new_folder_path}")


    files_to_move = [
        f"{pdf_name}_cards.csv",
        f"{pdf_name}_learning_objectives.csv",
        f"{pdf_name}_progress.csv",
        #'anki_deck.apkg',
        pdf_file
    ]
    subfolder_path = 'Archive'

    move_files_to_new_folder(files_to_move, subfolder_path)

print(f"All PDFs have been processed.")

# List to hold DataFrames for each CSV file
dfs = []

# Loop through each file in the cards_copy_folder
for file_name in os.listdir(cards_copy_folder):
    if file_name.endswith('.csv'):  # Only process files with the .csv extension
        file_path = os.path.join(cards_copy_folder, file_name)

        # Read the current CSV file into a DataFrame
        df = pd.read_csv(file_path)

        # Append the DataFrame to the list
        dfs.append(df)

# Concatenate all DataFrames into a single DataFrame
merged_df = pd.concat(dfs, ignore_index=True)

# Save the merged DataFrame to a new CSV file
output_file_path = os.path.join(script_dir, 'Merged.csv')
merged_df.to_csv(output_file_path, index=False)

print(f'Merged CSV has been created at: {output_file_path}')

# Run tag_deck.py script on the merged CSV and anki_deck.apkg
print(f"Running Scripts/tag_deck.py with Merged.csv and anki_deck.apkg")

tag_deck_command = ["python3", "Scripts/tag_deck.py", "Merged.csv", "anki_deck.apkg"]
process = subprocess.Popen(tag_deck_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
with process.stdout:
    for line in iter(process.stdout.readline, ''):
        print(line, end='', flush=True)
with process.stderr:
    for line in iter(process.stderr.readline, ''):
        print(line, end='', flush=True)
process.wait()

if process.returncode == 0:
    print(f"Scripts/tag_deck.py finished successfully.\n")
else:
    print(f"Scripts/tag_deck.py encountered an error (exit code: {process.returncode}).\n")

# Cleanup: Delete all {pdf_name}_cards.csv files in cards_copy_folder after merging and tagging
for file_name in os.listdir(cards_copy_folder):
    file_path = os.path.join(cards_copy_folder, file_name)
    if file_name.endswith('_cards.csv'):
        os.remove(file_path)
        print(f"Deleted {file_path}")

print("All temporary cards CSV files have been deleted. Process complete.")
