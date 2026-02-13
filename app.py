import os
import time
import json
from pathlib import Path
from typing import Optional

from urllib import request as urlrequest

from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore

from src.helper import download_embeddings
from src.prompt import system_prompt


app = Flask(__name__)


load_dotenv()
load_dotenv(Path.cwd() / ".env")
load_dotenv(Path.cwd().parent / ".env")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is missing. Add it to .env")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY is missing. Add it to .env")


embedding = download_embeddings()

index_name = "medical-chatbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embedding
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

FALLBACK_MODELS = [
    "meta-llama/llama-3.3-70b-instruct:free",
    "deepseek/deepseek-r1-distill-llama-70b:free",
    "qwen/qwen-2.5-72b-instruct:free",
]


def discover_openrouter_free_models(limit: int = 12) -> list[str]:
    """Fetch currently available free model IDs from OpenRouter."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "http://localhost",
        "X-Title": "Medbot",
    }
    req = urlrequest.Request("https://openrouter.ai/api/v1/models", headers=headers)
    try:
        with urlrequest.urlopen(req, timeout=10) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        data = payload.get("data", [])
        ids = [m.get("id") for m in data if isinstance(m, dict) and m.get("id")]
        free_ids = [m for m in ids if m.endswith(":free")]
        if free_ids:
            return free_ids[:limit]
    except Exception as exc:
        print(f"Could not auto-discover models from OpenRouter: {exc}")
    return []


def build_chat_model(model_name: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=model_name,
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        default_headers={
            "HTTP-Referer": "http://localhost",
            "X-Title": "Medbot",
        },
        temperature=0.1,
    )


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)


def _extract_sources(response: dict) -> list[str]:
    """Extract unique source names from the RAG response context documents."""
    sources = []
    seen = set()
    for doc in response.get("context", []):
        src = doc.metadata.get("source", "")
        if src:
            # Show just the filename, not the full path
            name = Path(src).name
            if name not in seen:
                seen.add(name)
                sources.append(name)
    return sources


def invoke_with_model_failover(user_input: str) -> Optional[dict]:
    """Try multiple models and return {"answer": ..., "sources": [...]} or None."""
    discovered = discover_openrouter_free_models()
    model_candidates = discovered + [m for m in FALLBACK_MODELS if m not in discovered]

    for model_name in model_candidates:
        try:
            chat_model = build_chat_model(model_name)
            qa_chain = create_stuff_documents_chain(chat_model, prompt)
            active_rag_chain = create_retrieval_chain(retriever, qa_chain)
            response = active_rag_chain.invoke({"input": user_input})
            print(f"Model used: {model_name}")
            return {
                "answer": str(response["answer"]),
                "sources": _extract_sources(response),
            }
        except Exception as exc:
            print(f"RAG error on model {model_name}: {exc}")
            if "429" in str(exc):
                time.sleep(1.2)
            continue
    return None


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg", "").strip()
    if not msg:
        return jsonify({"answer": "Please enter a question.", "sources": []})

    print(f"User: {msg}")
    result = invoke_with_model_failover(msg)

    if result is not None:
        print(f"Response: {result['answer']}")
        return jsonify(result)

    return jsonify({
        "answer": "I am not a doctor. The AI providers are currently busy. Please try again in a minute.",
        "sources": [],
    })


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
