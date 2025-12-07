import os
import pandas as pd
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import json
import re
import time
import requests

load_dotenv()

# ---------------------------------------------------------
#  CONFIGURE HACKAI / OPENROUTER PROXY
# ---------------------------------------------------------
HACKAI_KEY = os.getenv("HACKAI_API_KEY")
if not HACKAI_KEY:
    raise RuntimeError("❌ No HACKAI_API_KEY found in .env")

HACKAI_URL = "https://ai.hackclub.com/proxy/v1/chat/completions"

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
#  GENERATION USING HACKAI (OpenRouter format)
# =========================================================

def call_hackai_generate(prompt, num_problems, max_attempts=3, backoff=1.0):
    """
    Uses HackAI proxy (OpenRouter-compatible) to generate text.
    """
    last_raw = ""

    for attempt in range(1, max_attempts + 1):
        try:
            payload = {
                "model": "deepseek/deepseek-r1-distill-qwen-32b",  # can change anytime
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "stream": False
            }

            headers = {
                "Authorization": f"Bearer {HACKAI_KEY}",
                "Content-Type": "application/json"
            }

            r = requests.post(HACKAI_URL, headers=headers, json=payload, timeout=60)

            if r.status_code != 200:
                print(f"[ERROR] HackAI returned HTTP {r.status_code}")
                print(r.text)
                time.sleep(backoff * attempt)
                continue

            data = r.json()
            raw = data["choices"][0]["message"]["content"]
            last_raw = raw

            problems = _block_parse(raw, num_problems)
            if len(problems) >= num_problems:
                return problems

        except Exception as e:
            print(f"[WARNING] HackAI failed (Try {attempt}): {e}")

        time.sleep(backoff * attempt)

    # Fallback
    if not last_raw:
        return [{
            "title": "API Error",
            "statement": "No response from HackAI (likely overload). Try again later.",
            "solution": "The model returned no output."
        }]

    return _block_parse(last_raw, num_problems)


# =========================================================
#  PARSER
# =========================================================

def _block_parse(txt, num):
    if not isinstance(txt, str) or not txt.strip():
        return []

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

CONTEST STYLE RULES (MANDATORY):

If the contest is AIME:
- Answer must be a positive integer from 0 to 999.
- No calculator use in this contest, so dont have heavy computation and approximation in these problems.
- Problems must rely on typical AIME structures: modular arithmetic, functional equations, tricky geometry ratios, combinatorics with structure, or clever integer reasoning.
- Avoid brute force search problems or problems requiring checking large ranges.
- Avoid problems where the answer is obviously > 999.
- Difficulty should resemble AIME problems #9–#15 if level = "advanced", or #1–#8 if level = "intermediate".
- Use classical AIME phrasing and conciseness.

If the contest is Euclid/CSIMC:
- Problems must be multi-step.
- Statement can be longer.
- As calculator is allowed in these contests, the solution can involve more heavy computation or approximation than AIME.
- May include proofs or partial reasoning.
- Should involve algebraic manipulation or geometry reasoning, etc., typical of Euclid.
- If it is an Advanced problem under this category: 
  - Very deep problem structure.
  - Often requires lemma-style insight.
  - Significantly more olympiad-level theory based.
  - Should have a highly inductive difficult transformation or trick.
  - Typically these problems are proof questions rather than expecting singular answers.

If the contest is AMC:
- Simple, elegant, one-step conceptual.
- Small numbers.
- Light computation.

Each contest has its OWN flavor.
As an elite, original, and creative problem-settor, YOU MUST match the targeted contest exactly and specific in tone, answer, problem style.
Study the given sample texts to notice possible strong patterns, and use intenret knowledge if necessary as well.


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

    generated = call_hackai_generate(prompt, num_problems)
    return jsonify(generated)


if __name__ == "__main__":
    app.run(debug=True)

