from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
from dotenv import load_dotenv

from google import genai

from utils import extract_text
from rag import build_index, retrieve

# ---------------- SETUP ----------------
load_dotenv()

app = Flask(__name__)
CORS(app, origins=["http://127.0.0.1:5500"])

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# 🔥 GLOBAL STORAGE
current_text = ""


# ---------------- HOME ----------------
@app.route("/")
def home():
    return jsonify({"message": "RAG Backend Running"})


# ---------------- UPLOAD ----------------
@app.route("/upload", methods=["POST"])
def upload_file():

    global current_text

    file = request.files["file"]
    filename = secure_filename(file.filename)

    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    text = extract_text(path)

    if not text.strip():
        return jsonify({"error": "No text found"}), 400

    current_text = text
    build_index(text)

    return jsonify({"message": "Document uploaded & indexed"})


# ---------------- SUMMARIZE ----------------
@app.route("/summarize", methods=["POST"])
def summarize():

    global current_text

    print("🔥 SUMMARIZE API CALLED")

    if not current_text.strip():
        return jsonify({"error": "No document uploaded"}), 400

    prompt = f"""
Summarize the document:

1. Short Summary
2. Key Points
3. Conclusion

Document:
{current_text}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return jsonify({"summary": response.text})


# ---------------- ASK (RAG) ----------------
@app.route("/ask", methods=["POST"])
def ask():

    print("🔥 ASK API CALLED")

    data = request.get_json()
    question = data.get("question", "")

    if not question.strip():
        return jsonify({"error": "Empty question"}), 400

    context_chunks = retrieve(question)
    context = "\n\n".join(context_chunks)

    print("CHUNKS:", len(context_chunks))

    if not context.strip():
        return jsonify({"answer": "No relevant context found"}), 200

    prompt = f"""
You are an AI assistant.
Answer ONLY using the context.

Context:
{context}

Question:
{question}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return jsonify({"answer": response.text})


# ---------------- RUN ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)