import os
import time
import json
from pathlib import Path
from typing import Optional
from urllib import request as urlrequest

from dotenv import load_dotenv
from flask import Flask, render_template, request
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
# Optional CBM imports (commented out intentionally):
# from langchain.memory import ConversationBufferMemory
# from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain_community.chat_message_histories import ChatMessageHistory
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
# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embedding
)




retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

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
        # Requested invalid model kept for reference:
        # model="meta-llama/llama-3.2-70b-instruct:free",
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
        # Optional CBM placeholder (enable when memory is enabled):
        # ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ]
)

# Optional Conversation Buffer Memory (CBM) scaffold:
# _session_histories: dict[str, ChatMessageHistory] = {}
#
# def get_session_history(session_id: str) -> ChatMessageHistory:
#     if session_id not in _session_histories:
#         _session_histories[session_id] = ChatMessageHistory()
#     return _session_histories[session_id]
#
# To use CBM, wrap your chain with:
# rag_chain_with_memory = RunnableWithMessageHistory(
#     active_rag_chain,
#     get_session_history,
#     input_messages_key="input",
#     history_messages_key="chat_history",
# )

def invoke_with_model_failover(user_input: str) -> Optional[str]:
    discovered = discover_openrouter_free_models()
    model_candidates = discovered + [m for m in FALLBACK_MODELS if m not in discovered]

    for model_name in model_candidates:
        try:
            chat_model = build_chat_model(model_name)
            qa_chain = create_stuff_documents_chain(chat_model, prompt)
            active_rag_chain = create_retrieval_chain(retriever, qa_chain)
            response = active_rag_chain.invoke({"input": user_input})
            print(f"Model used: {model_name}")
            return str(response["answer"])
        except Exception as exc:
            print(f"RAG error on model {model_name}: {exc}")
            # Provider throttling is common on free tiers; short backoff helps.
            if "429" in str(exc):
                time.sleep(1.2)
            continue
    return None



@app.route("/")
def index():
    return render_template('chat.html')



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    # Optional: client can send a session id for memory-aware chats.
    # session_id = request.form.get("session_id", "default")
    print(msg)
    answer = invoke_with_model_failover(msg)
    if answer is not None:
        print("Response:", answer)
        return answer
    return "I am not a doctor. The AI providers are currently busy. Please try again in a minute."



if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
