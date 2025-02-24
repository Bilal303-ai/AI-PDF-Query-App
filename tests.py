import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from sentence_transformers import SentenceTransformer
from backend import extract_text_and_generate_embeddings, store_embeddings, query_similar_text, chat_with_llm


class TestPDFProcessing(unittest.TestCase):
    
    @patch("backend.PyPDFLoader")
    @patch("backend.embedding_model.encode") 
    def test_extract_text_and_generate_embeddings(self, mock_encode, mock_loader):
        """Test PDF extraction and embedding generation"""
        
        # Mock extracted text
        mock_loader.return_value.load.return_value = [
            MagicMock(page_content="This is a test document", metadata={})
        ]
        
        # Mock embeddings
        mock_encode.return_value = np.random.rand(384)
        
        chunks, embeddings = extract_text_and_generate_embeddings("test.pdf")
        
        # Assertions
        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(embeddings), 1)
        self.assertEqual(len(embeddings[0]), 384) # Check embedding size
        
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
        
class TestLLMIntegration(unittest.TestCase):
    
    @patch("backend.openai.chat.completions.create")
    def test_chat_with_llm(self, mock_llm):
        """Test chat function with mocked OpenAI response"""
        
        # Mock llm response
        mock_llm.return_value = MagicMock(choices=[MagicMock(message=MagicMock(content="Mocked response"))])
        
        response = chat_with_llm("What is AI?", ["AI is artificial intelligence"], "fake_api_key")
        
        self.assertEqual(response, "Mocked response")
        
        
if __name__ == "__main__":
    unittest.main()