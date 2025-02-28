import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from backend import extract_text_and_generate_embeddings, store_embeddings, query_similar_text, chat_with_llm


class TestPDFProcessing(unittest.TestCase):
    

    def test_extract_text_and_generate_embeddings(self):
        """Test PDF extraction and embedding generation"""
        
        chunks, embeddings = extract_text_and_generate_embeddings(r"tests/sample.pdf")
        
        # Assertions
        self.assertEqual(len(chunks), len(embeddings)) # No. of chunks must be equal to no. of embeddings
        for chunk in chunks:
            self.assertLess(len(chunk.page_content), 1001) # Max size of a chunk should be 1000

        
class TestDatabase(unittest.TestCase):
    pass
        
        
if __name__ == "__main__":
    unittest.main()
