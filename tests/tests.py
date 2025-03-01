import unittest
import os
import numpy as np
from langchain_core.documents import Document
from backend import extract_text_and_generate_embeddings, store_embeddings, query_similar_text, chat_with_llm


EMBEDDING_DIM = 384

class TestPDFProcessing(unittest.TestCase):
    

    def test_extract_text_and_generate_embeddings(self):
        """Test PDF extraction and embedding generation"""
        
        chunks, embeddings = extract_text_and_generate_embeddings(r"tests/sample.pdf")
        
        # Assertions
        self.assertEqual(len(chunks), len(embeddings)) # No. of chunks must be equal to no. of embeddings
        for chunk in chunks:
            self.assertLess(len(chunk.page_content), 1001) # Max size of a chunk should be 1000

        
class TestDatabase(unittest.TestCase):
    def setUp(self):
        self.db_url = os.getenv("DATABASE_URL")
        self.pdf_filename = "test.pdf"
        self.chunks = [Document(
            page_content="This is a test document."
        )]
        self.embeddings = [np.random.rand(EMBEDDING_DIM).astype(np.float32)]
        
    def test_store_embeddings(self):
        """Test that chunks and embeddings are stored in the DB"""
        test_pdf_id = store_embeddings(
            self.db_url,
            self.pdf_filename,
            self.chunks,
            self.embeddings,
        )
        
        # Check if tables 'pdf' and 'embeddings' exist
        with psycopg.connect(self.db_url) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                SELECT EXISTS (
                    SELECT FROM
                        pg_tables
                    WHERE
                        schemaname='public' AND
                        tablename='pdfs'
                )
                """)
                self.assertTrue(cur.fetchone()[0])
                
                cur.execute("""
                SELECT EXISTS (
                    SELECT FROM
                        pg_tables
                    WHERE
                        schemaname='public' AND
                        tablename='embeddings'
                )
                """)
                self.assertTrue(cur.fetchone()[0])
    
    def tearDown(self):
        with psycopg.connect(self.db_url) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                DROP TABLE pdfs, embeddings;            
                """)
        
        
if __name__ == "__main__":
    unittest.main()
