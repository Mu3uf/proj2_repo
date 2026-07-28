


import os
from dotenv import load_dotenv
load_dotenv()  # يقرأ المتغيرات من ملف .env

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

# ---------------------------------------------------------------------------
# Tools available to the agents
# ---------------------------------------------------------------------------
@tool
def read_file(file_path: str) -> str:
    """Reads and returns the full text content of a file given its path."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


@tool
def count_word_in_text(text: str, word: str) -> str:
    """Counts how many times a specific word appears inside a block of text (case-insensitive)."""
    words = text.lower().replace(",", " ").replace(".", " ").split()
    count = words.count(word.lower())
    return f"The word '{word}' appears {count} times."


# ---------------------------------------------------------------------------
# Agent 1: Summarizer Agent
# ---------------------------------------------------------------------------
summarizer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that reads text files and produces short, clear summaries."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

summarizer_agent = create_tool_calling_agent(llm, [read_file], summarizer_prompt)
summarizer_executor = AgentExecutor(agent=summarizer_agent, tools=[read_file], verbose=True)


def summarize_file_agent(file_path: str) -> str:
    """Agent Task 1: summarize a given text file."""
    result = summarizer_executor.invoke(
        {"input": f"Read the file at path '{file_path}' and summarize its content in 3-4 sentences."}
    )
    return result["output"]


# ---------------------------------------------------------------------------
# Agent 2: Word Counter Agent
# ---------------------------------------------------------------------------
counter_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an assistant that reads text files and counts how many times a specific word appears."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

counter_agent = create_tool_calling_agent(llm, [read_file, count_word_in_text], counter_prompt)
counter_executor = AgentExecutor(agent=counter_agent, tools=[read_file, count_word_in_text], verbose=True)


def count_word_in_file_agent(file_path: str, word: str) -> str:
    """Agent Task 2: count a specific word inside a given text file."""
    result = counter_executor.invoke(
        {"input": f"Read the file at path '{file_path}', then count how many times the word '{word}' appears in it."}
    )
    return result["output"]


# ---------------------------------------------------------------------------
# Manual test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample_path = os.path.join(os.path.dirname(__file__), "sample_tasks.txt")

    print("=== Agent 1: Summarizer ===")
    print(summarize_file_agent(sample_path))

    print("\n=== Agent 2: Word Counter ===")
    print(count_word_in_file_agent(sample_path, "task"))