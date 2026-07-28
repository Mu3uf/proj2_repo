# proj2_repo
## Setup

1. Clone the repository:
```bash
   git clone git@github.com:Mu3uf/proj2_repo.git
   cd proj2_repo
```

2. Create and activate a virtual environment:
```bash
   python3 -m venv venv
   source venv/bin/activate
```

3. Install dependencies:
```bash
   pip install -r requirements.txt
```

4. Set up your environment variables:
```bash
   cp .env.example .env
```
   Then open `.env` and add your own OpenAI API key:

   ## How to Run

### Run the Agents standalone
```bash
python agents.py
```
Runs the Summarizer Agent and Word Counter Agent against `sample_tasks.txt` and prints the results to the terminal.

### Run the LangGraph workflow standalone
```bash
python graph.py
```
Runs the full planner graph on a sample task list and prints the final state.

### Run the full web app
```bash
python app.py
```
Then open **http://127.0.0.1:5000** in your browser.

## How It Works

### Agents (`agents.py`)
Two LangChain agents that decide on their own when to call tools:
- **Summarizer Agent** — reads a text file and produces a short summary.
- **Word Counter Agent** — reads a text file and counts how many times a given word appears.

### LangGraph Workflow (`graph.py`)
A 4-node graph pipeline:
| Node | Purpose |
|---|---|
| `summarize` | Breaks raw task input into individual subtasks |
| `classify` | Labels each subtask as Work / Study / Personal |
| `prioritize` | Assigns a priority (High / Medium / Low) based on urgency and importance |
| `smart_plan` | Produces a final suggested order for completing the tasks |

### Flask App (`app.py`)
Serves a simple form where a user enters their daily tasks and receives:
- Original task input
- Generated subtasks
- Classification & priority table
- Final Smart Plan

## Example Output

**Input:**
**Output:**

| Subtask | Category | Priority |
|---|---|---|
| Finish the math homework | Study | High |
| Prepare the client presentation | Work | High |
| Buy groceries | Personal | Medium |
| Call mom | Personal | Low |
| Review the pull request | Work | Medium |

**Smart Plan:** A short suggested order with reasoning for tackling the tasks.

## Deployment Notes (for the Cybersecurity Team)

- The app is a standard Flask app (`app.py`), runs on port `5000`.
- Requires an `OPENAI_API_KEY` set via `.env` (not committed to the repo).
- No database is used — all state is in-memory per request.
- Recommended: run behind a WSGI server (e.g. Gunicorn) and add rate limiting / input validation before public deployment.

## Learning Outcomes

- Built and executed LangChain tool-calling Agents
- Designed a multi-node LangGraph workflow with shared state
- Connected an AI backend to a Flask front-end
- Managed secrets safely using `.env` and `.gitignore`