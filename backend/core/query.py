import os
import re
from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain.docstore.document import Document
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_cohere import CohereRerank
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

load_dotenv()

INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT")

def initialize_retriever() -> any:

    if not GOOGLE_API_KEY or not INDEX_NAME or not PINECONE_API_KEY or not PINECONE_ENV:
        raise ValueError("Pinecone or Google environment variables not set.")

    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    vector_store = PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME,
        embedding=embedding_model,
        namespace=None 
    )
    
    return vector_store.as_retriever(search_kwargs={"k": 20})

def retrieve_and_rerank(query: str, retriever: any) -> List[Document]:
    if not COHERE_API_KEY:
        raise ValueError("COHERE_API_KEY environment variable not set.")
        
    print(f"Retrieving initial documents for query: '{query}'")
    retrieved_docs = retriever.invoke(query)
    
    print(f"Retrieved {len(retrieved_docs)} documents. Now applying reranker...")
    reranker = CohereRerank(cohere_api_key=COHERE_API_KEY, top_n=5, model="rerank-english-v3.0")
    
    compressed_docs = reranker.compress_documents(
        documents=retrieved_docs,
        query=query
    )
    
    print(f"Reranker returned {len(compressed_docs)} top documents.")
    
    return compressed_docs

def generate_answer_with_citations(query: str, retrieved_docs: List[Document]) -> Dict[str, Any]:
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")

    system_prompt_template = """
    You are a helpful assistant for a Q&A application. Your task is to answer the user's question
    based ONLY on the provided context. If the answer is not available in the context,
    state that you cannot find the answer. Do not make up any information.
    
    For your answer, use inline citations in the format [source_id].
    For example: "The sky is blue [1]." The 'source_id' corresponds to the number of the source in the list below.
    
    Context:
    {context}
    
    Sources:
    {sources}
    """
    
    source_list = []
    formatted_context_for_llm = []
    
    for i, doc in enumerate(retrieved_docs):
        source_id = i + 1
        source_list.append(f"Source [{source_id}]: {doc.metadata.get('source', 'Unknown Source')}")
        formatted_context_for_llm.append(f"Chunk from Source [{source_id}]:\n{doc.page_content}\n")

    final_context = "\n".join(formatted_context_for_llm)
    final_sources = "\n".join(source_list)

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt_template),
        ("human", "Question: {query}")
    ])
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    rag_chain = prompt | llm | StrOutputParser()
    
    print("Generating answer with LLM...")
    llm_response = rag_chain.invoke({
        "context": final_context,
        "sources": final_sources,
        "query": query
    })
    
    lower_response = llm_response.lower()
    no_answer_keywords = ["not available", "cannot find", "do not have information", "not found"]
    if any(keyword in lower_response for keyword in no_answer_keywords):
        return {
            "answer": "Sorry, I couldn't find a definitive answer to that question in the provided documents. Please try a different query.",
            "sources": []
        }
    
    cited_source_ids = [int(s) for s in re.findall(r"\[(\d+)\]", llm_response)]
    
    cited_documents = []
    for doc in retrieved_docs:
        if (retrieved_docs.index(doc) + 1) in cited_source_ids:
            cited_documents.append({
                "source": doc.metadata.get("source", "Unknown"),
                "content": doc.page_content,
                "citation_id": retrieved_docs.index(doc) + 1
            })

    return {
        "answer": llm_response,
        "sources": cited_documents
    }