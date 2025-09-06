import os
from dotenv import load_dotenv
from typing import List
from pinecone import Pinecone, ServerlessSpec
import time

from langchain.docstore.document import Document
from langchain_community.document_loaders import PyPDFLoader 
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT")

def load_and_chunk_document(file_path: str) -> List[Document]:
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = text_splitter.split_documents(pages)
    
    for i, chunk in enumerate(chunks):
        chunk.metadata["source"] = os.path.basename(file_path)
        chunk.metadata["chunk_id"] = i
    
    return chunks

def load_and_chunk_document_from_string(text: str) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    # create_documents expects a list of strings
    chunks = text_splitter.create_documents([text])

    for i, chunk in enumerate(chunks):
        chunk.metadata["source"] = "Pasted Text"
        chunk.metadata["chunk_id"] = i

    return chunks

def create_embeddings_and_store(chunks: List[Document]) -> None:
    if not GOOGLE_API_KEY or not PINECONE_API_KEY or not PINECONE_ENV:
        raise ValueError("Environment variables for APIs not set correctly.")
        
    print(f"Creating embeddings for {len(chunks)} chunks...")
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
    
    if INDEX_NAME in pc.list_indexes().names():
        print(f"Index '{INDEX_NAME}' already exists. Deleting it...")
        pc.delete_index(INDEX_NAME)

        time.sleep(5)
    
    print(f"Creating a new index named '{INDEX_NAME}'...")
    pc.create_index(
        name=INDEX_NAME, 
        dimension=768, 
        metric="cosine",
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )
    
    while not pc.describe_index(INDEX_NAME).status['ready']:
        time.sleep(1)

    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embedding_model,
        index_name=INDEX_NAME
    )
    print(f"Successfully stored {len(chunks)} chunks in the '{INDEX_NAME}' index.")

if __name__ == "__main__":
    data_directory = "C:/Users/smrit/Desktop/Coding/Machine_Learning/RAG/backend/DOCS"
    
    if not os.path.isdir(data_directory):
        print(f"Error: The directory '{data_directory}' was not found. Please create it and add your documents.")
    else:
        print(f"Starting ingestion of documents from '{data_directory}'...")
        all_chunks = []
        for filename in os.listdir(data_directory):
            if filename.endswith(".pdf"):
                file_path = os.path.join(data_directory, filename)
                try:
                    chunks = load_and_chunk_document(file_path)
                    all_chunks.extend(chunks)
                    print(f"Loaded {len(chunks)} chunks from {filename}.")
                except Exception as e:
                    print(f"Failed to load {filename}: {e}")
        
        if all_chunks:
            create_embeddings_and_store(all_chunks)
            print("\nData ingestion pipeline completed successfully!")
        else:
            print("No text documents found to ingest.")