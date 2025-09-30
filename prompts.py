from llama_index.core import PromptTemplate

# instruction_str = """\
# 1. Convert the query to executable Python code using Pandas.
# 2. The final line of code should be a Python expression that can be called with the `eval()` function.
# 3. The code should represent a solution to the query.
# 4. PRINT ONLY THE EXPRESSION.
# 5. Do not quote the expression.
# """
# THIS WAS THE EXISTING ONE DON'T DELETE

instruction_str = """
You are a decision support expert working with a pandas dataframe named df.
- Only generate Python code that operates directly on 'df', which is already loaded and contains all relevant data.
- Never import pandas, never redefine df, never recreate the data structure.
- Do not use print statements.
- Only output Pandas code that references columns in df, like:
  df.loc[df['PerformanceScore'].idxmin()]
or
  df.sort_values('PerformanceScore').iloc[0]

Here are some examples
Q: “Who is the worst candidate?”
df.loc[df['PerformanceScore'].idxmin()]

Q: “Who has the lowest leadership score?”
df.loc[df['LeadershipScore'].idxmin()]
"""







new_prompt = PromptTemplate(
    """\
You are working with a pandas dataframe in Python.
The name of the dataframe is `df`.
This is the result of `print(df.head())`:
{df_str}

Follow these instructions:
{instruction_str}
Query: {query_str}

Expression: """
)


# context = """Purpose: The primary role of this agnet is to assist users by providing accurate
#             information about world population statistics and details about a battle."""