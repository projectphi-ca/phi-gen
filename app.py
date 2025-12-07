import os
import pandas as pd
from flask import Flask, render_template, request, jsonify
from google.genai import Client
from dotenv import load_dotenv
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import re
import time

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("No GEMINI_API_KEY in environment.")

client = Client(api_key=api_key)

app = Flask(__name__)

limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["20 per minute"]
)

df = pd.read_csv("data/problems.csv")

difficulty_map = {
    "intro": "intro",
    "intermediate": "intermediate",
    "medium": "intermediate",
    "advanced": "advanced",
    "hard": "advanced"
}
df["level"] = df["level"].map(difficulty_map).fillna(df["level"])


def call_gemini_generate(prompt, num_problems, max_attempts=3, backoff=1.0):
    last_raw = ""

    for attempt in range(1, max_attempts + 1):
        try:
            # IMPORTANT: use contents= (google-genai 1.x)
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                stream=False
            )

            raw = response.text.strip()
            last_raw = raw

            problems = _block_parse(raw, num_problems)
            if len(problems) >= num_problems:
                return problems

        except Exception as e:
            print(f"[WARNING] Gemini API failed attempt {attempt}: {e}")
            time.sleep(backoff * attempt)

    return [{
        "title": "API Error",
        "statement": "Gemini returned an error or insufficient output.",
        "solution": "Try again later."
    }]


def _block_parse(txt, num):
    if not isinstance(txt, str) or not txt.strip():
        return []

    blocks = re.split(r"###\s*Problem\s*\d+", txt, flags=re.IGNORECASE)
    problems = []

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        # Extract statement & solution
        parts = re.split(r"(Statement|Solution)\s*:\s*", block, flags=re.IGNORECASE)

        statement = ""
        solution = ""

        for i in range(1, len(parts), 2):
            key = parts[i].lower()
            value = parts[i + 1].strip()

            if key == "statement":
                m = re.search(r"Solution\s*:\s*", value, flags=re.IGNORECASE)
                statement = value[:m.start()].strip() if m else value

            elif key == "solution":
                solution = value.strip()

        if not statement:
            statement = block

        statement = re.sub(r"^Problem\s*\d+", "", statement).strip()

        problems.append({
            "title": "",
            "statement": statement,
            "solution": solution
        })

        if len(problems) >= num:
            break

    return problems


@app.route("/")
def index():
    contests = sorted(df["contest"].unique())
    topics = sorted(df["topic"].unique())
    levels = ["intro", "intermediate", "advanced"]
    return render_template("index.html", contests=contests, topics=topics, levels=levels)


@app.route("/generate", methods=["POST"])
def generate():
    contest = request.form["contest"]
    topic = request.form["topic"]
    subtopic = request.form["subtopic"]
    difficulty = request.form["difficulty"]
    num_problems = int(request.form["num_problems"])

    filtered = df[df["contest"] == contest]

    if topic:
        filtered = filtered[filtered["topic"] == topic]
    if subtopic:
        filtered = filtered[filtered["subtopic"] == subtopic]

    if filtered.empty:
        sample_texts = "No example problems available."
    else:
        sample_texts = "\n\n".join(
            filtered["problem_text"].sample(min(5, len(filtered))).tolist()
        )

    prompt = f"""
You are an elite math contest problem-setter.

Generate EXACTLY {num_problems} NEW and ORIGINAL math contest problems with:

Contest style: {contest}
Topic: {topic or "any"}
Subtopic: {subtopic or "any"}
Difficulty level: {difficulty}

Learn style ONLY from these real problems:
{sample_texts}

REQUIREMENTS:
- Start DIRECTLY with "### Problem 1"
- No intro text.
- Short official contest wording.
- Solution contains answer.
- Statement must NOT include answer.
- Use $...$ or $$...$$ for LaTeX.
- STRICT FORMAT:

### Problem 1
Statement: ...
Solution: ...
###

(then Problem 2 ... etc)
"""

    generated = call_gemini_generate(prompt, num_problems)
    return jsonify(generated)


if __name__ == "__main__":
    app.run(debug=True)
