import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core import Settings
from llama_index.experimental.query_engine import PandasQueryEngine
from prompts import instruction_str
from note_engine import note_engine

# Load LLM key and initialize Gemini
load_dotenv()
google_key = os.getenv("GOOGLE_API_KEY")
Settings.llm = GoogleGenAI(api_key=google_key, model="gemini-2.5-flash")





# Sidebar: CSV selection
csv_options = {
    "HR Candidates": "data/hr_candidates.csv",
    "Patient Records": "data/patients.csv",
    "Project Status": "data/projects.csv"
}
st.sidebar.title("Decision Support CSVs")
csv_choice = st.sidebar.selectbox("Choose a CSV data file", list(csv_options.keys()))
csvfile_path = csv_options[csv_choice]
df = pd.read_csv(csvfile_path)






# Show a preview and stats
st.title("SmartRAG: The Ultimate AI Decision Support Agent for HR, Healthcare, and Beyond")
st.subheader("Showing: " + csv_choice)
st.dataframe(df.head(20))  # show first 20 rows
st.markdown(f"**{df.shape[0]} rows, {df.shape[1]} columns**")




# Set up PandasQueryEngine 
csv_engine = PandasQueryEngine(df=df, instruction_str=instruction_str, verbose=True)




# User enters a NL question
question = st.text_input("Ask a question about the data (e.g. best candidate, most risky project, urgent patient):")
if st.button("Get Answer") or question:
    if not question.strip():
        st.info("Please enter a question.")
    else:
        answer = csv_engine.query(question)
        st.success(f"Answer:\n{answer}")

        # Option to save insight as note
        if st.button("Save this answer as a note"):
            note_engine(str(answer))
            st.toast("Note saved!")

            
# block for custom notes 
        st.subheader("Save a Custom Note")
        custom_note = st.text_area("Write your own note below:", "", height=100)
        if st.button("Save Custom Note"):
            if custom_note.strip():
                note_engine(custom_note.strip())
                st.success("Custom note saved!")
            else:
                st.info("Type a note before saving.")


# (Optional) Show all previously saved notes
st.subheader("Decision Notes")
if os.path.exists("data/notes.txt"):
    with open("data/notes.txt", "r") as f:
        notes = f.read().strip()
        if notes: st.text_area("Your saved notes:", notes, height=200)
        else: st.write("No notes saved yet.")
else:
    st.write("No notes found.")

st.caption("Built with LlamaIndex, Gemini LLM, and Streamlit—see how ANY business data becomes real, actionable decisions. 🚀")


