import os
import json
import uuid
import threading
from flask import Flask, render_template, request, jsonify
from facets import load_facets
from scorer import score_conversation
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
facets = load_facets()

# In-memory job store: {job_id: {"status": ..., "done": int, "total": int, "result": ...}}
jobs = {}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/score", methods=["POST"])
def score():
    """Starts scoring in a background thread and immediately returns a job_id.
    The frontend polls /status/<job_id> for progress instead of waiting on
    one long request (which would time out on a hosted server)."""
    data = request.json
    conversation = data.get("conversation", "").strip()
    if not conversation:
        return jsonify({"error": "No conversation provided"}), 400

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "running", "done": 0, "total": 0,
        "result": None, "error": None,
        "partial_results": {}, "stats": {"avg": 0, "high": 0, "low": 0, "count": 0},
    }

    def run_job():
        try:
            def progress_cb(done, total, batch_scores):
                jobs[job_id]["done"] = done
                jobs[job_id]["total"] = total
                # Append this batch's scores to the running partial results
                jobs[job_id]["partial_results"].update(batch_scores)
                # Recompute running avg/high/low over everything scored so far
                vals = [v["score"] for v in jobs[job_id]["partial_results"].values()]
                if vals:
                    jobs[job_id]["stats"] = {
                        "avg": round(sum(vals) / len(vals), 2),
                        "high": max(vals),
                        "low": min(vals),
                        "count": len(vals),
                    }

            scores = score_conversation(conversation, facets, progress_cb=progress_cb)
            sorted_scores = dict(
                sorted(scores.items(), key=lambda x: x[1]["score"], reverse=True)
            )
            jobs[job_id]["status"] = "complete"
            jobs[job_id]["result"] = {"scores": sorted_scores, "total": len(sorted_scores)}
        except Exception as e:
            import traceback
            traceback.print_exc()   # prints the full error to your terminal
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"] = str(e)

    threading.Thread(target=run_job, daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Unknown job_id"}), 404
    return jsonify(job)


@app.route("/results")
def results():
    try:
        with open("results/all_scores.json", "r") as f:
            data = json.load(f)
        return jsonify({"total": len(data), "results": data})
    except Exception:
        return jsonify({"total": 0, "results": []})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)