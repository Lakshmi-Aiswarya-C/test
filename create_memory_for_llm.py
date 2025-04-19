from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

## Uncomment the following files if you're not using pipenv as your virtual environment manager
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# === Step 1: Load PDFs with source tracking ===
DATA_PATH = "data/"

def load_pdf_files(data_path):
    all_docs = []
    for filename in os.listdir(data_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(data_path, filename)
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            # Tag source (e.g., "Gale", "DSM-5", "ICD-10") based on filename
            for doc in docs:
                if "gale" in filename.lower():
                    doc.metadata["source_name"] = "Gale Encyclopedia"
                elif "dsm" in filename.lower():
                    doc.metadata["source_name"] = "DSM-5"
                elif "icd" in filename.lower():
                    doc.metadata["source_name"] = "ICD-10"
                else:
                    doc.metadata["source_name"] = "Unknown"
            all_docs.extend(docs)
    return all_docs

documents = load_pdf_files(DATA_PATH)

# === Step 2: Split into chunks ===
def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks = create_chunks(documents)

# === Step 3: Embeddings ===
def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

embedding_model = get_embedding_model()

# === Step 4: Save to FAISS ===
DB_FAISS_PATH = "vectorstore/db_faiss"
db = FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_FAISS_PATH)