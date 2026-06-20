from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate

MODEL_NAME = "llama3"
CODE_MODEL_NAME = "codellama"
_MODEL = None
_CHAT_MODEL = None

_CODE_LLM = None

f_log = None

def set_logger(logger):
    global f_log
    f_log = logger

def code_llm():
    global _CODE_LLM
    if _CODE_LLM is None:
        _CODE_LLM = Ollama(model=CODE_MODEL_NAME)
    return _CODE_LLM

def write_log(log_msg, commit=False):
    f_log.write(log_msg + '\n')
    if commit:
        f_log.flush()

def query_llm(query):
    global _MODEL
    if _MODEL is None:
        _MODEL = Ollama(model=MODEL_NAME)
    if f_log is not None:
        write_log(query)

    response = _MODEL.invoke(query)
    
    if f_log is not None:
        write_log(response)
        write_log('-'*100, commit=True)

    return response

def chat_llm(messages):
    global _CHAT_MODEL
    if _CHAT_MODEL is None:
        _CHAT_MODEL = ChatOllama(model=CODE_MODEL_NAME)
    return _CHAT_MODEL.invoke(messages)

if __name__ == '__main__':
    examples_template = PromptTemplate(
        input_variables=["question", "answer"],
        template="Please answer this question: {question}\n{answer}"
    )

    examples = [
        {
            "question": "What is 2+2?",
            "answer": "The answer is 4.",
        },
        {
            "question": "What is 3 - 5?",
            "answer": "The answer is -2.",
        },
    ]

    prompt_template = FewShotPromptTemplate(
        examples=examples,
        example_prompt=examples_template,
        suffix="Please answer this question: {question}",
        input_variables=["question"],
    )

    prompt = prompt_template.format(question="What is 6 * 7?")
    print(prompt)
    print(query_llm(prompt))

    # If you want a straightforward prompt without examples:
    prompt_template = PromptTemplate(
        input_variables=["question"],
        template="Please answer this question: {question}",
    )
    prompt = prompt_template.format(question="How many kidney beans are there?")
    print(prompt)
    print(query_llm(prompt))
