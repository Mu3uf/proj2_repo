

from flask import Flask, render_template, request
from graph import run_planner

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    task_input = ""
    if request.method == "POST":
        task_input = request.form.get("task_input", "")
        if task_input.strip():
            result = run_planner(task_input)
    return render_template("index.html", result=result, task_input=task_input)


if __name__ == "__main__":
    app.run(debug=True, port=5000)