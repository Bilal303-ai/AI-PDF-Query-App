from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformers


def extract_text_and_generate_embeddings(pdf_path):
    # laod PDF
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    # Split text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    
    chunks = text_splitter.split_document(docs)
    
    # Generate embeddings
    embedding_model = SentenceTransformers("all-MiniLM-L6-v2")
    embeddings = [embedding_model.encode(chunk.page_content) for chunk in chunks]
    
    return chunks, embeddings


# Store embeddings in PostgreSQL
def store_embeddings(dbname, user, password, host, chunks, embeddings):
    with psycopg.connect("dbname=dbname user=user password=password host=host") as conn:
        with conn.cursor as cur:
            cur.execute("""
                CREATE EXTENSION IF NOT EXISTS vector
                """)
            
            cur.execute("""
                CREATE TABLE IF NOT EXISTS pdfs (
                    id SERIAL PRIMARY KEY,
                    filename TEXT UNIQUE,
                    uploaded_at TIMESTAMP DEFAULT NOW())
                """)
            
            cur.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    id SERIAL PRIMARY KEY,
                    pdf_id INTEGER REFERENCES pdfs(id) ON DELETE CASCADE,
                    text TEXT,
                    embedding vector(384))
                """)
            
            for chunk, emb in zip(chunks, embeddings):
                cur.execute(
                    "INSERT INTO embeddings (pdf_id, text, embedding) VALUES (%s, %s, %s)",
                    (pdf_id, chunk.page_content, emb.tolist())
                )
                
            conn.commit()