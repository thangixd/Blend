def get_generate_context_prompt(table: str, question: str, num_of_rows: int):
    return f"""Given a dataset, consisting of {num_of_rows} rows, with the following columns:
/*
{table}
*/
and this question:
/*
{question}
*/
Assume you are the creator of the table and have all the necessary information to respond to the question. Generate a concise answer to the question based on the table, satisfying the following criteria:
1. Completeness: The answer must definitively and comprehensively address all parts of the question.
2. Relevance: The answer must directly provide the information requested in the question without any extraneous details."""


def get_generate_bx1_prompt(context: str):
    return f"""Context: "{context}"

The above context describes a table. Please create a question that requests a table based on this description. For example, given a context "This table was created by X", the question would be "Provide a table that was created by X."

Respond in the following format:
Question: ..."""


def get_generate_bx2_prompt(question: str):
    return f"""Original Question:"{question}"

Rephrase the above question with different wordings. Respond in the following format:
Rephrased Question: ..."""


def get_bx2_extra_rephrase_prompt(question: str):
    return f"""Original Question:"{question}"

Rephrase the above question with completely different words (do not share any keywords). Respond in the following format:
Rephrased Question: ..."""


def get_labeling_prompt(context: str, question: str):
    return f"""Context:"{context}"

Question:"{question}"

The context describes a specific table that we have access to. Does this table answer the question? Begin your response with yes/no."""
