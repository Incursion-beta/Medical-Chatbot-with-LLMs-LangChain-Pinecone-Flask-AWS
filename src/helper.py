from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List
from langchain_core.documents import Document


# Extract text from PDF files
def load_pdf_files(data):
    loader = DirectoryLoader(
        data,
        glob="**/*.pdf",            # include PDFs in nested folders
        loader_cls=PyPDFLoader,
        silent_errors=True,          # skip unreadable/corrupt files
        show_progress=True,
        use_multithreading=True
    )

    documents = loader.load()
    return documents


def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Given a list of Document objects, return a new list of Document objects
    containing only 'source' in metadata and the original page_content.
    """
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )
    return minimal_docs



# Split the documents into smaller chunks
def text_split(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
    )
    texts_chunk = text_splitter.split_documents(minimal_docs)
    return texts_chunk


def download_embeddings():
    """
    Download embedding model for retrieval.
    """
    # Previous model (kept for comparison):
    # model_name = "sentence-transformers/all-MiniLM-L6-v2"

    # Current stronger model:
    model_name = "BAAI/bge-large-en-v1.5"

    # Previous init (kept for comparison):
    # embeddings = HuggingFaceEmbeddings(
    #     model_name=model_name
    # )

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": True}
    )
    return embeddings

