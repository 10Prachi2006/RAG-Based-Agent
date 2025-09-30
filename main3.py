import os
import pandas as pd
from prompts import instruction_str
from note_engine import note_engine



from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core import Settings
import os
from dotenv import load_dotenv

load_dotenv()
google_key = os.getenv("GOOGLE_API_KEY")
if not google_key:
    raise ValueError("Missing GOOGLE_API_KEY in environment variables!")
Settings.llm = GoogleGenAI(api_key=google_key, model="gemini-2.5-flash")






csv_options = {
    "hr": "data/hr_candidates.csv",
    "patients": "data/patients.csv",
    "projects": "data/projects.csv"
}
print("Choose a CSV to analyze (type only: hr, patients, projects):")
for key, path in csv_options.items():
    print(f"{key}: {path}")
chosen = input("Enter choice: ").strip().lower()
csvfile_path = csv_options.get(chosen, "data/hr_candidates.csv")
csvfile_df = pd.read_csv(csvfile_path)
print("Data preview:\n", csvfile_df.head())




from llama_index.experimental.query_engine import PandasQueryEngine
csv_engine = PandasQueryEngine(
    df=csvfile_df,
    instruction_str=instruction_str,
    verbose=True
)

def main():
    print("Direct CSV Decision Support Ready!")
    while True:
        prompt = input("Ask your data question (q to quit, 'note:' to save note): ").strip()
        if prompt.lower() == "q":
            print("Exiting...")
            break
        if prompt.lower().startswith("note:"):
            note = prompt[5:].strip()
            result = note_engine(note)
            print("Note Saved:", result)
        else:
            result = csv_engine.query(prompt)
            print("Answer:", result)

if __name__ == "__main__":
    main()
