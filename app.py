from flask import Flask, render_template, request, session
from graph import app as rag_app


app = Flask(__name__)

app.secret_key = "orbitdesk_secret_key"  


def run_agent(question: str) -> dict:
    """
    Invokes the LangGraph app and packages the result into the shape
    described by output_schema.json. Falls back to safe defaults for
    any field the graph didn't set for a given path.
    """
    result = rag_app.invoke({"question": question, "revision_count": 0})

    return {
        "question": question,
        "classification": result.get("classification", "unknown"),
        "answer": result.get("answer", "No answer found"),
        "sources": result.get("sources", []),
        "confidence": result.get("confidence", 0.0),
        "requires_human": result.get("requires_human", False),
        "reason": result.get("reason", ""),
        "clarification_question": result.get("clarification_question"),
        "warnings": result.get("warnings", []),
    }


@app.route("/", methods=["GET", "POST"])
def home():

    if "history" not in session:
        session["history"] = []

    current = None

    if request.method == "POST":

        question = request.form.get("question")

        if question:

            current = run_agent(question)

            history = session["history"]
            history.append(current)
            session["history"] = history

    return render_template(
        "index.html",
        history=session["history"],
        current=current,
    )


@app.route("/reset", methods=["POST"])
def reset():
    session["history"] = []
    return render_template("index.html", history=[], current=None)


if __name__ == "__main__":
    app.run(debug=True)
