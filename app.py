import os
import json
from flask import Flask, render_template, request, jsonify
from facets import load_facets
from scorer import score_conversation
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
facets = load_facets()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/score", methods=["POST"])
def score():
    data = request.json
    conversation = data.get("conversation", "").strip()
    if not conversation:
        return jsonify({"error": "No conversation provided"}), 400
    try:
        scores = score_conversation(conversation, facets, batch_size=20)
        # Sort by score descending
        sorted_scores = dict(
            sorted(scores.items(), key=lambda x: x[1]["score"], reverse=True)
        )
        return jsonify({"scores": sorted_scores, "total": len(sorted_scores)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/results")
def results():
    try:
        with open("results/all_scores.json", "r") as f:
            data = json.load(f)
        return jsonify({"total": len(data), "results": data})
    except:
        return jsonify({"total": 0, "results": []})

if __name__ == "__main__":
    app.run(debug=True, port=5000)