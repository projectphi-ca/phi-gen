# Phi-Gen: AI-Powered Contest Math Problem Generator

## 🌟 Overview

**Phi-Gen** is a web application designed to generate original, structured mathematics contest problems.

This tool was created by the team behind **Project Phi**.

It uses a dedicated dataset of past contest problems to **fine-tune** Hack AI's **Qwen 32b** model. The model is **trained on a dataset** to replicate the specific style, tone, and complexity required for competitive mathematics.

---

## ✨ Core Functionality

Phi-Gen is engineered to produce highly realistic and structured problems for competitive math environments:

* **Contest-Specific Style Replication:** The system generates problems and comprehensive solutions that effectively mimick the style, difficulty, and phrasing of established contest materials.
* **Targeted Generation:** Users can specify the contest, topic, subtopic, and difficulty level.
* **Supported Contests:** The platform currently supports generation in the style of **AIME**, **Euclid**, and **CSIMC** problems.
* **High-Quality Output:** Every generated problem is accompanied by a full reasoning and final answer, ensuring the output is immediately useful.
* **Professional Math Formatting:** All mathematical content in both statements and solutions is formatted using standard **LaTeX** notation (`$...$` for inline and `$$...$$` for display math) for clean rendering.

---

## 🛠️ Setup and Installation

### Dependencies
Ensure you have the following packages installed:
* Flask
* pandas
* google-genai
* python-dotenv
* Flask-Limiter

### Installation Steps

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Set up API Key:** Create a file named `.env` in the project's root directory and add your Gemini API key:
    ```
    GEMINI_API_KEY="YOUR_GEMINI_API_KEY_HERE"
    ```

3.  **Add Dataset:** Ensure your complete contest problem dataset is present at `data/problems.csv`.

---

## 🚀 Running Locally

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Flask Application:**
    ```bash
    python app.py
    ```

3.  **Access the App:** Open your web browser and navigate to: `http://127.0.0.1:5000`
