import unittest
from unittest.mock import patch, MagicMock
from app import app

class TestChatEndpoint(unittest.TestCase):
    def setUp(self):
        # Set up test client
        self.app = app.test_client()
        self.app.testing = True

    @patch('app.qa_chain')
    def test_valid_question(self, mock_qa_chain):
        # Mock the response from qa_chain
        mock_qa_chain.return_value = "This is a mock answer."

        response = self.app.post(
            '/chat',
            json={"question": "What is the capital of France?"}
        )
        
        self.assertEqual(response.status_code, 200)
        self.assertIn("question", response.json)
        self.assertIn("answer", response.json)
        self.assertEqual(response.json["question"], "What is the capital of France?")
        self.assertEqual(response.json["answer"], "This is a mock answer.")

    def test_missing_question(self):
        response = self.app.post(
            '/chat',
            json={}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json)
        self.assertEqual(response.json["error"], "Question is required")

    @patch('app.qa_chain')
    def test_qa_chain_exception(self, mock_qa_chain):
        # Mock qa_chain to raise an exception
        mock_qa_chain.side_effect = Exception("Something went wrong!")

        response = self.app.post(
            '/chat',
            json={"question": "What is the capital of France?"}
        )

        self.assertEqual(response.status_code, 500)
        self.assertIn("error", response.json)
        self.assertEqual(response.json["error"], "Something went wrong!")

    @patch('app.qa_chain')
    def test_large_input_question(self, mock_qa_chain):
        # Mock the response from qa_chain for large input
        mock_qa_chain.return_value = "This is a mock answer for a large question."

        large_question = "What is" + " very" * 1000 + " large?"
        response = self.app.post(
            '/chat',
            json={"question": large_question}
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("question", response.json)
        self.assertIn("answer", response.json)
        self.assertEqual(response.json["answer"], "This is a mock answer for a large question.")

if __name__ == '__main__':
    unittest.main()
