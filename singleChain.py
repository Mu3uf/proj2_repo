
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7)
parser = StrOutputParser()

summary_prompt = PromptTemplate(
    input_variables=["tasks"],
    template="You are an AI assistant. Please summarize the following daily tasks into a clean, brief summary:\n\n{tasks}"
)
summary_chain = summary_prompt | llm | parser

chain_1 = summary_chain

word_count_prompt = PromptTemplate(
    input_variables=["summary"],
    template="Count the number of words in the following text and return only the number:\n\n{summary}"
)
word_count_chain = {"summary": chain_1} | word_count_prompt | llm | parser

classification_prompt = PromptTemplate(
    input_variables=["summary"],
    template="Classify the topic of the following text into one of these categories (Education, Business, Health, Personal, Productivity):\n\n{summary}"
)
classification_chain = {"summary": chain_1} | classification_prompt | llm | parser

sequential_pipeline = RunnableParallel(
    summary=chain_1,
    word_count=word_count_chain,
    category=classification_chain
)

sample_tasks = """
- Finish report for project
- Prepare slides for team meeting
- Buy groceries
- Schedule doctor appointment
- Reply to urgent emails
"""

print("--- Testing Sequential Multi-step Workflow ---\n")
final_output = sequential_pipeline.invoke({"tasks": sample_tasks})

print("1. Summary Output:\n", final_output["summary"])
print("\n2. Word Count:\n", final_output["word_count"])
print("\n3. Category:\n", final_output["category"])