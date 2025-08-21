import time
import io
from unittest.mock import Mock, patch, MagicMock, mock_open  

import pytest
from fastapi.testclient import TestClient

from app import app_state, app


class TestAPI:
    
    @pytest.fixture
    def client(self):
        return TestClient(app)


    @pytest.fixture
    def mock_chatbot(self):
        mock_bot = Mock()
        mock_bot.initialize.return_value = True
        mock_bot.ask.return_value = 'This is a test response from the chatbot'
        
        mock_bot.rag = Mock()
        mock_bot.rag.get_status.return_value = {
            'initialized': True,                       
            'document_count': 5,                   
            'chunk_count': 12,                         
            'documents_path': '/tmp/test_documents', 
            'index_path': '/tmp/test_index'
        }
        mock_bot.rag.add_document = Mock()
        
        mock_bot.llm = Mock()
        mock_bot.llm.get_model_info.return_value = {
            'model_name': 'test-model',
            'device': 'cpu',
            'quantized': False,
            'dtype': 'float32',
            'context_length': 2048,
        }
        
        return mock_bot
    

    @pytest.fixture(autouse=True)
    def reset_app_state(self):
        original_state = {
            "chatbot": app_state.chatbot,
            "is_ready": app_state.is_ready,
            "initialization_error": app_state.initialization_error,
        }
        
        yield  
        
        app_state.chatbot = original_state["chatbot"]
        app_state.is_ready = original_state["is_ready"]
        app_state.initialization_error = original_state["initialization_error"]
    

    @pytest.fixture
    def app_with_ready_chatbot(self, mock_chatbot):
        app_state.chatbot = mock_chatbot
        app_state.is_ready = True
        app_state.initialization_error = None
        
        yield


    @pytest.fixture
    def app_with_failed_chatbot(self):
        app_state.chatbot = None
        app_state.is_ready = False
        app_state.initialization_error = 'Test initialization error'
        
        yield
        

    def test_root_endpoint_ready(self, client, app_with_ready_chatbot):
        '''
        Test root endpoint when chatbot is ready
        
        Args:
            client (TestClient): FastAPI test client for making requests
            app_with_ready_chatbot (fixture): App with a ready chatbot
        '''
        response = client.get('/')
        
        assert response.status_code == 200
        data = response.json()
        assert data['message'] == 'RAG+LLM ChatBot API'
        assert data['status'] == 'ready'
        assert data['docs'] == '/docs'


    def test_root_endpoint_not_ready(self, client, app_with_failed_chatbot):
        '''
        Test root endpoint when chatbot is not ready

        Args:
            client (TestClient): FastAPI test client for making requests
            app_with_failed_chatbot (fixture): App with a not ready chatbot
        '''
        response = client.get('/')
        
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'initializing'


    def test_health_endpoint_healthy(self, client, app_with_ready_chatbot):
        '''
        Test health endpoint when system is healthy

        Args:
            client (TestClient): FastAPI test client for making requests
            app_with_ready_chatbot (fixture): App with a ready chatbot
        '''
        response = client.get('/health')
        
        assert response.status_code == 200
        data = response.json()
        
        assert data['status'] == 'healthy'
        assert data['is_ready'] == True
        assert 'uptime' in data
        assert isinstance(data['uptime'], float)
        assert data['uptime'] >= 0
        
        assert 'rag_info' in data
        assert 'llm_info' in data
        assert data['rag_info']['document_count'] == 5
        assert data['llm_info']['model_name'] == 'test-model'


    def test_health_endpoint_unhealthy(self, client, app_with_failed_chatbot):
        '''
        Test health endpoint when system is unhealthy
        
        Args:
            client (TestClient): FastAPI test client for making requests
            app_with_failed_chatbot (fixture): App with a not ready chatbot
        '''
        response = client.get('/health')
        
        assert response.status_code == 200
        data = response.json()
        
        assert data['status'] == 'unhealthy'
        assert data['is_ready'] == False
        assert 'uptime' in data


    def test_ask_chatbot_success(self, client, app_with_ready_chatbot):
        '''
        Test successful chat query

        Args:
            client (TestClient): FastAPI test client for making requests
            app_with_ready_chatbot (fixture): App with a ready chatbot
        '''
        query_data = {'query': 'What is machine learning?'}
        response = client.post('/ask', json=query_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data['response'] == 'This is a test response from the chatbot'
        assert data['status'] == 'success'
        assert 'processing_time' in data
        assert isinstance(data['processing_time'], float)
        assert data['processing_time'] >= 0


    def test_ask_chatbot_empty_query(self, client, app_with_ready_chatbot):
        '''
        Test chat query with empty string

        Args:
            client (TestClient): FastAPI test client for making requests
            app_with_ready_chatbot (fixture): App with a ready chatbot
        '''
        query_data = {'query': ''}
        response = client.post('/ask', json=query_data)
        
        assert response.status_code == 200 


    def test_ask_chatbot_invalid_request(self, client, app_with_ready_chatbot):
        '''
        Test chat query with invalid request format
        
        Args:
            client (TestClient): FastAPI test client for making requests
            app_with_ready_chatbot (fixture): App with a ready chatbot
        '''
        response = client.post('/ask', json={})
        
        assert response.status_code == 422  


    def test_ask_chatbot_exception_handling(self, client, app_with_ready_chatbot, mock_chatbot):
        '''
        Test chat query when chatbot raises an exception
        
        Args:
            client (TestClient): FastAPI test client for making requests
            app_with_ready_chatbot (fixture): App with a ready chatbot
            mock_chatbot (fixture): Mocked chatbot instance
        '''
        mock_chatbot.ask.side_effect = Exception('Test exception')
        
        query_data = {'query': 'Test query'}
        response = client.post('/ask', json=query_data)
        
        assert response.status_code == 500
        data = response.json()
        assert 'Error processing query' in data['detail']


    def test_ask_chatbot_when_not_ready(self, client, app_with_failed_chatbot):
        '''
        Test chat query when chatbot is not ready
        
        Args:
            client (TestClient): FastAPI test client for making requests
            app_with_failed_chatbot (fixture): App with a not ready chatbot
        '''
        query_data = {'query': 'Test query'}
        response = client.post('/ask', json=query_data)
        
        assert response.status_code == 503
        data = response.json()
        assert 'Service unavailable' in data['detail']


    def test_upload_pdf_success(self, client):
        '''
        Test successful PDF upload

        Args: 
            client (TestClient): FastAPI test client for making requests
        '''
        pdf_content = b"%PDF-1.4\\n1 0 obj\\n<<\\n/Type /Catalog\\n>>\\nendobj\\nxref\\n0 1\\n0000000000 65535 f \\ntrailer\\n<<\\n/Size 1\\n/Root 1 0 R\\n>>\\nstartxref\\n9\\n%%EOF"
        
        files = {"file": ("test_document.pdf", io.BytesIO(pdf_content), "application/pdf")}
        
        with patch('os.makedirs'), patch('builtins.open', mock_open()) as mock_file:
            response = client.post("/upload_document", files=files)
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Document uploaded successfully"
        mock_file.assert_called_once()


    def test_upload_non_pdf_fails(self, client):
        '''
        Test upload with non-PDF file

        Args: 
            client (TestClient): FastAPI test client for making requests
        '''
        txt_content = b"This is a text file"
        files = {"file": ("document.txt", io.BytesIO(txt_content), "text/plain")}
        
        response = client.post("/upload_document", files=files)
        
        assert response.status_code == 400
        assert "Only PDF files are allowed" in response.json()["detail"]


    def test_upload_no_file(self, client):
        '''
        Test upload without file

        Args: 
            client (TestClient): FastAPI test client for making requests
        '''
        response = client.post("/upload_document")
        
        assert response.status_code == 422


    def test_upload_empty_filename(self, client):
        '''
        Test upload with empty filename

        Args: 
            client (TestClient): FastAPI test client for making requests
        '''
        pdf_content = b"%PDF-1.4\\nbasic pdf content"
        files = {"file": ("", io.BytesIO(pdf_content), "application/pdf")}
        
        response = client.post("/upload_document", files=files)
        
        assert response.status_code == 422

    @patch('os.makedirs')
    @patch('builtins.open')
    def test_upload_file_system_error(self, mock_open_func, mock_makedirs, client):

        # Test file system error during upload
        mock_open_func.side_effect = OSError("Disk full")
        
        pdf_content = b"%PDF-1.4\\nbasic pdf content"
        files = {"file": ("test.pdf", io.BytesIO(pdf_content), "application/pdf")}
        
        response = client.post("/upload_document", files=files)
        
        assert response.status_code == 500
        assert "Error uploading document" in response.json()["detail"]


    def test_add_document_success(self, client, app_with_ready_chatbot):
        '''
        Test successful document addition
        
        Args:
            client (TestClient): FastAPI test client for making requests
            app_with_ready_chatbot (fixture): App with a ready chatbot
        '''
        document_data = {'file_path': '/path/to/test/document.pdf'}
        response = client.post('/add_document', json=document_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data['message'] == 'Document addition started in background'
        assert data['file_path'] == '/path/to/test/document.pdf'
        assert data['status'] == 'processing'


    def test_add_document_invalid_request(self, client, app_with_ready_chatbot):
        '''
        Test document addition with invalid request format

        Args:
            client (TestClient): FastAPI test client for making requests
            app_with_ready_chatbot (fixture): App with a ready chatbot
        '''
        response = client.post('/add_document', json={})
        
        assert response.status_code == 422  


    def test_empty_file_path(self, client, app_with_ready_chatbot):
        '''
        Test document addition with empty file path
        
        Args:
            client (TestClient): FastAPI test client for making requests
            app_with_ready_chatbot (fixture): App with a ready chatbot
        '''
        document_data = {'file_path': ''}
        response = client.post('/add_document', json=document_data)
        
        assert response.status_code == 200


    def test_add_document_when_not_ready(self, client, app_with_failed_chatbot):
        '''
        Test document addition when chatbot is not ready

        Args:
            client (TestClient): FastAPI test client for making requests
            app_with_failed_chatbot (fixture): App with a not ready chatbot
        '''
        document_data = {'file_path': '/path/to/test/document.pdf'}
        response = client.post('/add_document', json=document_data)
        
        assert response.status_code == 503
        data = response.json()
        assert 'Service unavailable' in data['detail']


    @patch('app.backend.API.ChatBot')
    def test_reinitialize_success(self, mock_chatbot_class, client):
        ''''
        Test successful chatbot reinitialization
        
        Args:
            mock_chatbot_class (Mock): Patched ChatBot class returning a mocked instance
            client (TestClient): FastAPI test client for making requests
        '''
        mock_instance = Mock()
        mock_instance.initialize.return_value = True
        mock_chatbot_class.return_value = mock_instance
        
        response = client.post('/reinitialize')
        
        assert response.status_code == 200
        data = response.json()
        assert data['message'] == 'ChatBot reinitialized successfully'


    @patch('app.backend.API.ChatBot')
    def test_reinitialize_failure(self, mock_chatbot_class, client):
        '''
        Test failed chatbot reinitialization

        Args:
            mock_chatbot_class (Mock): Patched ChatBot class returning a mocked instance
            client (TestClient): FastAPI test client for making requests
        '''
        mock_instance = Mock()
        mock_instance.initialize.return_value = False
        mock_chatbot_class.return_value = mock_instance
        
        response = client.post('/reinitialize')
        
        assert response.status_code == 500
        data = response.json()
        assert 'Failed to reinitialize ChatBot' in data['detail']


    @patch('app.backend.API.ChatBot')
    def test_reinitialize_exception(self, mock_chatbot_class, client):
        '''
        Test chatbot reinitialization with exception
        
        Args:
            mock_chatbot_class (Mock): Patched ChatBot class returning a mocked instance
            client (TestClient): FastAPI test client for making requests
        '''
        mock_chatbot_class.side_effect = Exception('Test initialization exception')
        
        response = client.post('/reinitialize')
        
        assert response.status_code == 500


class TestEdgeCases:

    @pytest.fixture
    def client(self):
        return TestClient(app)
    

    @pytest.fixture
    def mock_chatbot(self):
        mock_bot = Mock()
        mock_bot.initialize.return_value = True
        mock_bot.ask.return_value = 'This is a test response from the chatbot'
        
        mock_bot.rag = Mock()
        mock_bot.rag.get_status.return_value = {
            'initialized': True,                       
            'document_count': 5,                   
            'chunk_count': 12,                         
            'documents_path': '/tmp/test_documents', 
            'index_path': '/tmp/test_index'
        }
        mock_bot.rag.add_document = Mock()
        
        mock_bot.llm = Mock()
        mock_bot.llm.get_model_info.return_value = {
            'model_name': 'test-model',
            'device': 'cpu',
            'quantized': False,
            'dtype': 'float32',
            'context_length': 2048,
        }
        
        return mock_bot
    

    @pytest.fixture(autouse=True)
    def reset_app_state(self):
        original_state = {
            "chatbot": app_state.chatbot,
            "is_ready": app_state.is_ready,
            "initialization_error": app_state.initialization_error,
        }
        
        yield  
        
        app_state.chatbot = original_state["chatbot"]
        app_state.is_ready = original_state["is_ready"]
        app_state.initialization_error = original_state["initialization_error"]
    
    
    @pytest.fixture
    def app_with_ready_chatbot(self, mock_chatbot):
        app_state.chatbot = mock_chatbot
        app_state.is_ready = True
        app_state.initialization_error = None
        
        yield
    

    def test_add_document_with_special_characters(self, client, app_with_ready_chatbot):
        '''
        Test document addition with special characters in path
        
        Args:
            client (TestClient): FastAPI test client for making requests
            app_with_ready_chatbot (fixture): App with a ready chatbot
        '''
        document_data = {'file_path': '/path/with spaces/document-with-特殊字符.pdf'}
        response = client.post('/add_document', json=document_data)
        
        assert response.status_code == 200  


    def test_add_document_with_very_long_path(self, client, app_with_ready_chatbot):
        '''
        Test document addition with very long file path
        
        Args:
            client (TestClient): FastAPI test client for making requests
            app_with_ready_chatbot (fixture): App with a ready chatbot
        '''
        long_path = '/very/long/path/' + 'directory/' * 50 + 'document.pdf'
        document_data = {'file_path': long_path}
        response = client.post('/add_document', json=document_data)
        
        assert response.status_code == 200  


    def test_404_error(self, client):
        '''
        Test 404 error handling
        
        Args:
            client (TestClient): FastAPI test client for making requests
        '''
        response = client.get('/nonexistent-endpoint')

        assert response.status_code == 404
        data = response.json()
        assert 'Not Found' == data['detail']
        assert 'timestamp' in data


    def test_405_error(self, client):
        '''
        Test 405 error handling
        
        Args:
            client (TestClient): FastAPI test client for making requests
        '''
        response = client.put('/')  
        
        assert response.status_code == 405
        data = response.json()
        assert 'Method Not Allowed' == data['detail']
        assert 'timestamp' in data


    def test_invalid_json_format(self, client, app_with_ready_chatbot):
        '''
        Test request with malformed JSON
        
        Args:
            client (TestClient): FastAPI test client for making requests
            app_with_ready_chatbot (fixture): App with a ready chatbot
        '''
        response = client.post('/ask', data='invalid json string', headers={'Content-Type': 'application/json'})
        
        assert response.status_code == 422


    def test_large_query_payload(self, client, app_with_ready_chatbot):
        '''
        Test with very large query payload

        Args:
            client (TestClient): FastAPI test client for making requests
            app_with_ready_chatbot (fixture): App with a ready chatbot
        '''
        large_query = 'x' * 10000 
        query_data = {'query': large_query}
        response = client.post('/ask', json=query_data)
        
        assert response.status_code == 200  


    def test_sql_injection_attempt(self, client, app_with_ready_chatbot):
        '''
        Test SQL injection attempt in query
        
        Args:
            client (TestClient): FastAPI test client for making requests
            app_with_ready_chatbot (fixture): App with a ready chatbot
        '''
        malicious_query = {'query': '; DROP TABLE users; --'}
        response = client.post('/ask', json=malicious_query)
        
        assert response.status_code == 200 


    def test_xss_attempt(self, client, app_with_ready_chatbot):
        '''
        Test XSS attempt in query

        Args:
            client (TestClient): FastAPI test client for making requests
            app_with_ready_chatbot (fixture): App with a ready chatbot
        '''
        xss_query = {'query': "<script>alert('xss')</script>"}
        response = client.post('/ask', json=xss_query)
        
        assert response.status_code == 200 


    def test_null_values_in_request(self, client, app_with_ready_chatbot):
        '''
        Test request with null values

        Args:
            client (TestClient): FastAPI test client for making requests
            app_with_ready_chatbot (fixture): App with a ready chatbot
        '''
        response = client.post('/ask', json={'query': None})
        
        assert response.status_code == 422 


    def test_unicode_characters_in_query(self, client, app_with_ready_chatbot):
        '''
        Test query with unicode characters

        Args:
            client (TestClient): FastAPI test client for making requests
            app_with_ready_chatbot (fixture): App with a ready chatbot
        '''
        unicode_query = {'query': '¿Cómo estás? 你好 🚀 测试'}
        response = client.post('/ask', json=unicode_query)
        
        assert response.status_code == 200 
