# PDF Sage

PDF Sage is an AI-powered application that allows users to upload PDFs, extract text, generate embeddings, and store them in a PostgreSQL database with pgvector. Users can then query the system, retrieving relevant document chunks, and interact with an LLM for enhanced responses.

## Features

- **PDF Upload & Processing**: Extracts text from PDFs and splits it into meaningful chunks.
- **Embedding Generation**: Uses SentenceTransformer and `all-MiniLM-L6-v2` model to generate vector embeddings for document chunks.
- **PostgreSQL with pgvector**: Stores embeddings efficiently for fast similarity search.
- **Semantic Search**: Retrieves the most relevant text chunks based on user queries.
- **LLM Integration**: Uses GPT-4 to generate answers based on retrieved content.

## Tech Stack

- **Backend**: Python, LangChain, psycopg, Sentence Transformers, OpenAI API
- **Database**: PostgreSQL with pgvector
- **Frontend**: Streamlit

## Installation

### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- PostgreSQL 15+ with `pgvector` extension

### Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Bilal303-ai/pdf-sage.git
   cd pdf-sage
   ```

2. **Create Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up PostgreSQL Database:**
   - Ensure PostgreSQL is running.
   - Create a database: `CREATE DATABASE pdf_sage;`
   - Enable pgvector: `CREATE EXTENSION IF NOT EXISTS vector;`

5. **Run the Application:**
   ```bash
   streamlit run app.py
   ```

## Usage

1. **Upload a PDF**: The app extracts and processes text.
2. **Query the Document**: Enter a question, and the system retrieves the most relevant chunks.
3. **LLM Response**: The retrieved text is passed to GPT-4 for a refined answer.

## Troubleshooting

- **Error: "extension \"vector\" is not available"**
  - Ensure pgvector is installed and enabled in PostgreSQL.

## Key Architectural Decisions and Trade-offs

### 1. **Choice of Embedding Model**

- **Decision**: `all-MiniLM-L6-v2` was chosen for its balance between efficiency and performance.
- **Trade-off**: More advanced models (e.g., `text-embedding-ada-002`) offer better accuracy but at a higher computational cost.

### 2. **Database for Storage**

- **Decision**: PostgreSQL with `pgvector` for scalable vector storage.
- **Trade-off**: PostgreSQL is robust but may not scale as well as dedicated vector databases (e.g., Pinecone, FAISS) for large-scale applications.

### 3. **Vector Similarity Search Method**

- **Decision**: `embedding <-> query_embedding` (L2 distance) for similarity ranking.
- **Trade-off**: Cosine similarity could also be used, but L2 distance aligns well with Sentence Transformers.

### 4. **Hybrid Search Approach**

- **Decision**: First retrieve relevant chunks, then use an LLM to generate a final response.
- **Trade-off**: Pure retrieval methods are faster, while LLM-based reasoning improves response quality but increases cost.

### 5. **Framework Choice**

- **Decision**: Python with `psycopg`, `sentence-transformers`, and `openai`.
- **Trade-off**: While effective, a LangChain-based approach might offer additional flexibility.





