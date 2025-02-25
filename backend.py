from typing import List, Dict, Tuple
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import psycopg
from psycopg.rows import dict_row
import openai
import torch
import numpy as np

# Fix the  torch related bug
torch.classes.__path__ = []

# Load the embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2") # Load the embedding model

def extract_text_and_generate_embeddings(pdf_path: str) -> Tuple[List[Dict[str, str]], List[np.ndarray]]:
    """
    Extracts text from a PDF and generates embeddings from chunks.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        Tuple[List[Dict[str, str]], List[np.ndarray]]: A tuple containing a list of text chunks and their corresponding embeddings
    """
    loader = PyPDFLoader(pdf_path)
    docs = loader.lazy_load()
    
    # Split text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
    )
    
    chunks = []
    embeddings = []
    
    for doc in docs:
        if doc.page_content.strip():
            chunks_in_doc = text_splitter.split_documents([doc])
            chunks.extend(chunks_in_doc)
    
            # Generate embeddings
            embeddings_in_doc = [embedding_model.encode(chunk.page_content, dtype=np.float32) for chunk in chunks_in_doc]
            embeddings.extend(embeddings_in_doc)
    
    return chunks, embeddings

def clean_text(text: str) -> str:
    """Remove NUL (0x00) bytes from text"""
    return text.replace('\x00', "")
    
    
# Store embeddings in PostgreSQL
def store_embeddings(db_info: Dict[str, str], pdf_filename:str, 
                     chunks: List[Dict[str, str]], embeddings: List[np.ndarray]) -> int:
    """
    Stores text embeddings in PostgreSQL and associates with a PDF.
    
    Args:
        db_info (Dict[str, str]): Database connection parameters.
        pdf_filename (str)): Name of the uploaded PDF.
        chunks (List[Dict[str, str]]): List of text chunks
        embeddings (List[np.ndarray]): Corresponding embeddings for the chunks
        
    Returns:
        int: The ID of the stored PDF.
    """
    
    with psycopg.connect(**db_info) as conn:
        with conn.cursor() as cur:
            # Enable pgvector extension
            cur.execute("""
                CREATE EXTENSION IF NOT EXISTS vector
                """)
            
            # Create table for PDFs
            cur.execute("""
                CREATE TABLE IF NOT EXISTS pdfs (
                    id SERIAL PRIMARY KEY,
                    filename TEXT UNIQUE,
                    uploaded_at TIMESTAMP DEFAULT NOW())
                """)
            
            # Create table for embeddings
            cur.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    id SERIAL PRIMARY KEY,
                    pdf_id INTEGER REFERENCES pdfs(id) ON DELETE CASCADE,
                    text TEXT,
                    embedding vector(384))
                """)
            
            # Add an index for faster similarity search
            cur.execute("""
                CREATE INDEX IF NOT EXISTS embedding_idx
                ON embeddings USING ivfflat (embedding vector_l2_ops)
                WITH (lists = 100)
            """)
            
            # Check if PDF already exists
            cur.execute("SELECT id from pdfs WHERE filename = %s", (pdf_filename,))
            result = cur.fetchone()
            if result:
                pdf_id = result[0] # Reuse existing PDF ID
            else:
                cur.execute("INSERT INTO pdfs (filename) VALUES (%s) RETURNING id", (pdf_filename,))
                pdf_id = cur.fetchone()[0]
                
            for chunk, emb in zip(chunks, embeddings):
                cur.execute(
                    "INSERT INTO embeddings (pdf_id, text, embedding) VALUES (%s, %s, %s)",
                    (pdf_id, clean_text(chunk.page_content), emb.tolist())
                )
                
            conn.commit()
            return pdf_id
            
            
def query_similar_text(db_info: Dict[str, str], user_query: str, pdf_id: int, top_k: int=5) -> List[Dict[str, str]]:
    """
    Queries the database to find the most similar text chunks based on the user query.
    
    Args:
        db_info (Dict[str, str]): Database connection parameters.
        user_query (str): The user's search query.
        pdf_id (int): The ID of the PDF being searched.
        top_k (int, optional): Number of top results to result. Defaults to 5.
    
    Returns:
        List[Dict[str, str]]: A list of text chunks with similarity score
    """
    query_embedding = np.array(embedding_model.encode(user_query, dtype=np.float32)).tolist()
    
    with psycopg.connect(**db_info) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("""
                SELECT text, embedding <-> %s::vector AS distance
                FROM embeddings
                WHERE pdf_id = %s
                ORDER BY distance
                LIMIT %s
            """, (query_embedding, pdf_id, top_k))
            
            retrieved_results = cur.fetchall()
    return retrieved_results


def chat_with_llm(user_query: str, retrieved_text: List[str], api_key: str) -> str:
    """
    Generate response using llm based retrieved context
    
    Args:
        user_query (str): The user's query
        retrieved_text (List(str)): List of retrieved chunks from the database.
        api_key (str): OpenAI API key.
    """
    context = "\n\n".join(retrieved_text)
    openai.api_key = api_key
    
    prompt = f"""
    You are an AI assistant. Use the following context to answer the user's query. 
    If the context doesn't contain enough information, say "I don't know."

    Context:
    {context}

    User Query: {user_query}

    Answer:
    """
    
    completion = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ],
    )
    return completion.choices[0].message.content
