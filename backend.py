from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import psycopg
import openai
import numpy as np

embedding_model = SentenceTransformer("all-MiniLM-L6-v2") # Load the embedding model

def extract_text_and_generate_embeddings(pdf_path):
    # laod PDF
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    # Split text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    
    chunks = text_splitter.split_documents(docs)
    
    # Generate embeddings
    embeddings = [embedding_model.encode(chunk.page_content, dtype=np.float32) for chunk in chunks]
    
    return chunks, embeddings


# Store embeddings in PostgreSQL
def store_embeddings(db_info, pdf_filename, chunks, embeddings):
    
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
                ON embeddings USING ivfflat (embedding vector_cosine_ops)
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
                    (pdf_id, chunk.page_content, emb.tolist())
                )
                
            conn.commit()
            
            
def query_similar_text(pool, user_query, pdf_id, top_k=5):
    query_embedding = model.encode(user_query, convert_to_numpy=True).astype(float32).tolist()
    
    with pool.connection as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT text, embedding <=> %s AS similarity
                FROM embeddings
                WHERE pdf_id = %s
                ORDER BY similarity
                LIMIT %s
            """, (query_embedding, pdf_id, top_k))
            
            results = cur.fetchall()
    retrieved_vectors = [result[0] for result in results]
    return retrieved_vectors


def chat_with_llm(user_query, retrieved_vectors):
    """Generate response using llm from retrieved context"""
    context = "\n\n".join(retrieved_vectors)
    
    completion = openai.chat.completion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an AI assistant. Answer based on the provided context."},
            {"role": "user", "content": f"Context: {context}\n\nUser Question: {user_query}"},
        ]
    )
    return completion.choices[0].message.content
    