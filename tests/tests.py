import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from backend import extract_text_and_generate_embeddings, store_embeddings, query_similar_text, chat_with_llm


class TestPDFProcessing(unittest.TestCase):
    

    def test_extract_text_and_generate_embeddings(self):
        """Test PDF extraction and embedding generation"""
        
        chunks, embeddings = extract_text_and_generate_embeddings(r"C:\Users\Mohd Bilal Hasan\OneDrive\Desktop\Blue bash project\pdf-sage\tests\sample.pdf")
        
        # Assertions
        self.assertEqual(len(chunks), len(embeddings)) # No. of chunks must be equal to no. of embeddings
        for chunk in chunks:
            self.assertLess(len(chunk.page_content), 1001) # Max size of chunk should be 1000

        
class TestDatabase(unittest.TestCase):
    
    @patch("backend.psycopg.connect")
    def test_store_embedding(self, mock_connect):
        """Test storing embeddings in the database"""
        
        # Mock connection and cursor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        
        # Mock database insert operation
        chunks = [MagicMock(page_content='sample_text', metadata={})]
        embeddings = [np.random.rand(384)]
        
        store_embeddings({}, "test.pdf", chunks, embeddings)
        
        # Ensure INSERT was called
        mock_cursor.execute.assert_called()
    
    @patch("backend.psycopg.connect")    
    def test_query_similar_text(self, mock_connect):
        """Test querying similar text from database"""
        
        # Mock connection and cursor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        
        # Mock query results
        mock_cursor.fetchall.return_value = [{"text": "Test chunk", "distance": 0.2}]
        
        result = query_similar_text({}, "test_query", 1, top_k=3)
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["text"], "Test chunk")
        
        
if __name__ == "__main__":
    unittest.main()
