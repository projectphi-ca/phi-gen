import os
import pandas as pd
from flask import Flask, render_template, request, jsonify
from google.genai import Client          # <-- UPDATED
from dotenv import load_dotenv
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import json
import re
import time

load_dotenv()

# ---------------------------------------------------------
#  CONFIGURE GEMINI
# ---------------------------------------------------------
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("❌ No API key found. Put GEMINI_API_KEY in .env")

client = Client(api_key=api_key)          # <-- UPDATED

app = Flask(__name__)

limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["20 per minute"]
)

# ---------------------------------------------------------
#  LOAD DATASET
# ---------------------------------------------------------
df = pd.read_csv("data/problems.csv")

difficulty_map = {
    "intro": "intro",
    "intermediate": "intermediate",
    "medium": "intermediate",
    "advanced": "advanced",
    "hard": "advanced"
}
df["level"] = df["level"].map(difficulty_map).fillna(df["level"])

# =========================================================
#  GEMINI TEXT BLOCK PARSING + GENERATION
# =========================================================

def call_gemini_generate(prompt, num_problems, max_attempts=3, backoff=1.0):
    """
    Ask Gemini and parse the custom text block format.
    Updated to use google-genai 1.53 client API.
    """
    last_raw = ""

    for attempt in range(1, max_attempts + 1):
        raw = ""
        try:
            # NEW API CALL
            response = client.models.generate_content(
                model="gemini-2.5-flash",     # <-- UPDATED
                contents=prompt,
                max_output_tokens=1000
            )

            raw = response.text.strip()
            last_raw = raw

            problems = _block_parse(raw, num_problems)
            if len(problems) >= num_problems:
                return problems

        except Exception as e:
            print(f"[WARNING] Gemini API Call Failed (Attempt {attempt}). Error: {e}")
            if not last_raw:
                last_raw = ""

        time.sleep(backoff * attempt)

    if not last_raw:
        print("[WARNING] No output generated from Gemini. Returning empty list.")
        return [{
            "title": "API Error",
            "statement": "Could not generate problems. The server received an API error, possibly due to quota limits. Please try again in a few minutes.",
            "solution": "The model failed to return content. Check the console for details on the quota limit error (429)."
        }]

    return _block_parse(last_raw, num_problems)



def _block_parse(txt, num):
    """Parse ### Problem blocks (fail-safe)."""
    if not isinstance(txt, str) or not txt.strip():
        return []

    # FIX invalid escape by using raw string
    blocks = re.split(r"###\s*Problem\s*\d+", txt, flags=re.IGNORECASE)

    problems = []

    for block in blocks:
        b = block.strip()
        if not b:
            continue

        parts = re.split(r"(Statement|Solution)\s*:\s*", b, flags=re.DOTALL | re.IGNORECASE)

        statement = ""
        solution = ""

        for i in range(1, len(parts), 2):
            key = parts[i].lower().strip()
            value = parts[i+1].strip()

            if key == "statement":
                next_key_match = re.search(r"(Solution)\s*:\s*", value, flags=re.DOTALL | re.IGNORECASE)
                statement = value[:next_key_match.start()].strip() if next_key_match else value.strip()

            elif key == "solution":
                solution = value.strip()

        if not statement:
            stmt_match = re.search(
                r"(.*?)(?=Solution\s*:|$)",
                b, flags=re.DOTALL | re.IGNORECASE
            )
            statement = stmt_match.group(1).strip() if stmt_match else b

        if not solution:
            solution_match = re.search(
                r"Solution\s*:\s*(.*)$",
                b, flags=re.DOTALL | re.IGNORECASE
            )
            solution = solution_match.group(1).strip() if solution_match else ""

        statement = re.sub(r"###$", "", statement).strip()
        solution = re.sub(r"###$", "", solution).strip()

        statement = "\n".join(
            ln for ln in statement.splitlines()
            if not ln.lower().startswith("answer:")
        ).strip()

        statement = re.sub(r"^Problem\s*\d+\s*", "", statement).strip()

        problems.append({
            "title": "",
            "statement": statement,
            "solution": solution
        })

        if len(problems) >= num:
            break

    return problems


# =========================================================
#  ROUTES
# =========================================================

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
        sample_texts = "No specific examples available for this filter combination."
    else:
        sample_texts = "\n\n".join(
            filtered["problem_text"].sample(min(5, len(filtered))).tolist()
        )

    prompt = f"""
You are an elite math contest problem-setter.

Generate EXACTLY {num_problems} NEW and ORIGINAL math contest problems with:

Contest style: {contest}
Topic: {topic if topic else "any"}
Subtopic: {subtopic if subtopic else "any"}
Difficulty level: {difficulty}

Learn the writing style ONLY from these real problems:
{sample_texts}

REQUIREMENTS:
- DO NOT copy or modify any dataset problems.
- DO NOT include ANY introductory text.
- DO NOT say "Here are your problems", "Sure", "Okay", or anything else before Problem 1.
- Start DIRECTLY with "### Problem 1".
- Use SHORT, official contest-style wording (AMC/AIME/CMO style).
- The statement MUST NOT contain the answer.
- The answer must appear ONLY inside the solution.
- Use LaTeX notation for all math expressions, wrapped in inline ($...$) or display ($$...$$) delimiters (e.g., $x^2 + 1$, not just x^2 + 1).
- Keep LaTeX notation simple (like x^2, \sqrt{''}, \frac{''}{''}).
- No markdown formatting except the "###" separators.

STRICT OUTPUT FORMAT (NO EXTRA TEXT):

### Problem 1
Statement: ...
Solution: ...
###

### Problem 2
Statement: ...
Solution: ...
###

(Continue until Problem {num_problems})

Produce EXACTLY {num_problems} problems in this format.
No extra commentary. No explanation outside solutions.
"""

    generated = call_gemini_generate(prompt, num_problems)
    return jsonify(generated)


if __name__ == "__main__":
    app.run(debug=True)

