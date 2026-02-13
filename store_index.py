import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from src.helper import (
    download_embeddings,
    filter_to_minimal_docs,
    load_pdf_files,
    text_split,
)

load_dotenv()
load_dotenv(Path.cwd() / ".env")
load_dotenv(Path.cwd().parent / ".env")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is missing. Add it to .env and retry.")

project_root = Path(__file__).resolve().parent
data_path = project_root / "data"
if not data_path.exists():
    raise FileNotFoundError(f"Data directory not found: {data_path}")

extracted_data = load_pdf_files(str(data_path))
minimal_docs = filter_to_minimal_docs(extracted_data)
texts_chunk = text_split(minimal_docs)

embedding = download_embeddings()

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medical-chatbot"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        # Previous dimension (all-MiniLM-L6-v2): 384
        dimension=1024,  # Current dimension for BAAI/bge-large-en-v1.5
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-2"),
    )

texts = [d.page_content for d in texts_chunk]
metadatas = []
for d in texts_chunk:
    md = {}
    source = d.metadata.get("source")
    page = d.metadata.get("page")
    if source is not None:
        md["source"] = str(source)
    if page is not None:
        md["page"] = int(page)
    metadatas.append(md)

docsearch = PineconeVectorStore.from_texts(
    texts=texts,
    embedding=embedding,
    metadatas=metadatas,
    index_name=index_name,
    batch_size=20,
    embeddings_chunk_size=50,
)

print(f"Indexed {len(texts)} chunks into Pinecone index '{index_name}'.") 