


import os
from dotenv import load_dotenv
load_dotenv() 

from typing import TypedDict, List, Dict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)


class PlannerState(TypedDict):
    task_input: str
    subtasks: List[str]
    classification: Dict[str, str]
    priority: Dict[str, str]
    smart_plan: str


# ---------------------------------------------------------------------------
# Node 1: Summarize the raw input into a clean list of subtasks
# ---------------------------------------------------------------------------
def summarize_node(state: PlannerState) -> PlannerState:
    prompt = ChatPromptTemplate.from_template(
        "Break the following daily activities into a clean list of separate subtasks. "
        "Return ONLY the subtasks, one per line, with no numbering or bullet symbols.\n\n"
        "Tasks:\n{task_input}"
    )
    chain = prompt | llm
    response = chain.invoke({"task_input": state["task_input"]})
    lines = [line.strip("-•0123456789. ").strip() for line in response.content.split("\n") if line.strip()]
    state["subtasks"] = lines
    return state


# ---------------------------------------------------------------------------
# Node 2: Classify each subtask into Work / Study / Personal
# ---------------------------------------------------------------------------
def classify_node(state: PlannerState) -> PlannerState:
    prompt = ChatPromptTemplate.from_template(
        "Classify each of the following subtasks into exactly one category: Work, Study, or Personal.\n"
        "Return the result strictly in the format 'subtask -> category', one per line, nothing else.\n\n"
        "Subtasks:\n{subtasks}"
    )
    chain = prompt | llm
    response = chain.invoke({"subtasks": "\n".join(state["subtasks"])})

    classification = {}
    for line in response.content.split("\n"):
        if "->" in line:
            task, category = line.split("->", 1)
            classification[task.strip()] = category.strip()
    state["classification"] = classification
    return state


# ---------------------------------------------------------------------------
# Node 3: Agent-style prioritization node
# ---------------------------------------------------------------------------
def prioritize_node(state: PlannerState) -> PlannerState:
    prompt = ChatPromptTemplate.from_template(
        "You are a prioritization agent. Given these classified subtasks, decide a priority "
        "(High, Medium, or Low) for each one, based on the urgency and importance implied by its wording.\n"
        "Return strictly as 'subtask -> priority', one per line, nothing else.\n\n"
        "Subtasks with categories:\n{classified}"
    )
    classified_text = "\n".join(f"{t} ({c})" for t, c in state["classification"].items())
    chain = prompt | llm
    response = chain.invoke({"classified": classified_text})

    priority = {}
    for line in response.content.split("\n"):
        if "->" in line:
            task, prio = line.split("->", 1)
            priority[task.strip()] = prio.strip()
    state["priority"] = priority
    return state


# ---------------------------------------------------------------------------
# Node 4: Generate the final Smart Plan
# ---------------------------------------------------------------------------
def smart_plan_node(state: PlannerState) -> PlannerState:
    prompt = ChatPromptTemplate.from_template(
        "Given these subtasks with their category and priority, write a short 'Smart Plan' "
        "suggesting the best order to do them today. Give a one-line reason for the order.\n\n"
        "Data:\n{data}"
    )
    data_lines = []
    for task in state["subtasks"]:
        cat = state["classification"].get(task, "Unknown")
        prio = state["priority"].get(task, "Unknown")
        data_lines.append(f"{task} | category={cat} | priority={prio}")

    chain = prompt | llm
    response = chain.invoke({"data": "\n".join(data_lines)})
    state["smart_plan"] = response.content
    return state


# ---------------------------------------------------------------------------
# Build and compile the graph
# ---------------------------------------------------------------------------
def build_planner_graph():
    graph = StateGraph(PlannerState)

    graph.add_node("summarize", summarize_node)
    graph.add_node("classify", classify_node)
    graph.add_node("prioritize", prioritize_node)
    graph.add_node("smart_plan", smart_plan_node)

    graph.set_entry_point("summarize")
    graph.add_edge("summarize", "classify")
    graph.add_edge("classify", "prioritize")
    graph.add_edge("prioritize", "smart_plan")
    graph.add_edge("smart_plan", END)

    return graph.compile()


planner_graph = build_planner_graph()


def run_planner(task_input: str) -> PlannerState:
    initial_state: PlannerState = {
        "task_input": task_input,
        "subtasks": [],
        "classification": {},
        "priority": {},
        "smart_plan": "",
    }
    return planner_graph.invoke(initial_state)


if __name__ == "__main__":
    sample = (
        "Finish the math homework, prepare the client presentation, "
        "buy groceries, call mom, review the pull request"
    )
    result = run_planner(sample)
    print(result)