import os
from unittest.mock import Mock, patch, MagicMock, mock_open


import pytest
import torch
from tempfile import TemporaryDirectory

from core import ChatBot


class TestChatBot:
    
    @pytest.fixture
    def mock_env_vars(self):
        return {
            'DEVICE': 'cuda',
            'DOCUMENTS_PATH': '/path/to/docs',
            'FAISS_INDEX_PATH': '/path/to/index',
            'TOP_K': '5',
            'PDF_PROCESS_MAX_WORDS': '1000',
            'EMB_MODEL_NAME': 'sentence-transformers/all-MiniLM-L6-v2',
            'RAG_CHUNK_SIZE': '512',
            'RAG_FORCE_REBUILD': 'false',
            'RAG_THRESHOLD': '0.7',
            'RAG_MAX_TOKENS': '2000',
            'LLM_STOP_SEQUENCES': "['</s>', '<|endoftext|>']",
            'LLM_MODEL_NAME': 'microsoft/DialoGPT-medium',
            'LLM_LOAD_IN_4BIT': 'true',
            'LLM_TORCH_DTYPE': 'float16',
            'LLM_MAX_NEW_TOKENS': '512',
            'LLM_TEMPERATURE': '0.7',
            'LLM_TOP_P': '0.9',
            'LLM_TOP_K': '50',
            'LLM_REPETITION_PENALTY': '1.1',
            'LLM_STREAM': 'false'
        }
    

    @patch('core.chatbot.RAGPipeline')
    @patch('core.chatbot.LLM')
    def test_init_success(self, mock_llm_class, mock_rag_class, mock_env_vars):
        # Test initialization of chatbot
        with patch.dict(os.environ, mock_env_vars):
            mock_rag_instance = Mock()
            mock_rag_class.return_value = mock_rag_instance
            
            mock_llm_instance = Mock()
            mock_llm_class.return_value = mock_llm_instance
            
            chatbot = ChatBot()
            
            assert chatbot.device == 'cuda'
            assert chatbot.documents_path == '/path/to/docs'
            assert chatbot.top_k == 5
            assert chatbot.rag_force_rebuild is False
            assert chatbot.rag_threshold == 0.7
            assert chatbot.llm_stop_sequences == ['</s>', '<|endoftext|>']
            assert chatbot.llm_load_in_4bit is True
            assert chatbot.llm_torch_dtype == torch.float16
            assert chatbot.llm_temperature == 0.7
            assert chatbot.is_initialized is False
            
            mock_rag_class.assert_called_once_with(
                chunk_size=512,
                embedding_model='sentence-transformers/all-MiniLM-L6-v2',
                documents_path='/path/to/docs',
                index_path='/path/to/index',
                device='cuda'
            )
            
            mock_llm_class.assert_called_once_with(
                model_name='microsoft/DialoGPT-medium',
                device='cuda',
                load_in_4bit=True,
                torch_dtype=torch.float16
            )
    

    @patch('core.chatbot.load_dotenv')
    def test_env_file_loading(self, mock_load_dotenv, mock_env_vars):
        # Test loading of .env file
        with patch('core.chatbot.RAGPipeline'), patch('core.chatbot.LLM'), patch.dict(os.environ, mock_env_vars, clear=True):
            
            ChatBot()
            
            mock_load_dotenv.assert_called_once()
            called_path = mock_load_dotenv.call_args[0][0]
            assert os.path.basename(called_path) == '.env'

    
    @patch('core.chatbot.RAGPipeline')
    @patch('core.chatbot.LLM')
    def test_initialize_rag_failure(self, mock_llm_class, mock_rag_class):
        # Test RAG failed initialization
        with patch.dict(os.environ, {}, clear=True):
            mock_rag_instance = Mock()
            mock_rag_instance.initialize.return_value = False
            mock_rag_class.return_value = mock_rag_instance
            
            mock_llm_instance = Mock()
            mock_llm_class.return_value = mock_llm_instance
            
            chatbot = ChatBot()
            
            with patch('core.chatbot.logger') as mock_logger:
                result = chatbot.initialize()
                
                assert result is False
                assert chatbot.is_initialized is False
                
                mock_logger.error.assert_called_with('Failed to initialize RAG pipeline')
                mock_llm_instance.load_model.assert_not_called()

    
    @patch('core.chatbot.RAGPipeline')
    @patch('core.chatbot.LLM')
    def test_initialize_llm_failure(self, mock_llm_class, mock_rag_class):
        # Test LLM failed initialization
        with patch.dict(os.environ, {}, clear=True):
            mock_rag_instance = Mock()
            mock_rag_instance.initialize.return_value = True
            mock_rag_class.return_value = mock_rag_instance
            
            mock_llm_instance = Mock()
            mock_llm_instance.load_model.return_value = False
            mock_llm_class.return_value = mock_llm_instance
            
            chatbot = ChatBot()
            
            with patch('core.chatbot.logger') as mock_logger:
                result = chatbot.initialize()
                
                assert result is False
                assert chatbot.is_initialized is False
                
                mock_logger.error.assert_called_with('Failed to load LLM model')
    

    def test_ask_not_initialized(self, mock_env_vars):
        with patch('core.chatbot.load_dotenv'), \
            patch('core.chatbot.RAGPipeline'), \
            patch('core.chatbot.LLM'), \
            patch.dict(os.environ, mock_env_vars, clear=True):
            
            chatbot = ChatBot()
            
            with pytest.raises(ValueError, match='ChatBot not initialized'):
                chatbot.ask('What is AI?')


    @patch('core.chatbot.RAGPipeline')
    @patch('core.chatbot.LLM')
    def test_ask_success(self, mock_llm_class, mock_rag_class, mock_env_vars):
        # Test query processing and response generation
        with patch.dict(os.environ, mock_env_vars):
            mock_rag_instance = Mock()
            mock_rag_instance.get_context.return_value = 'Context about AI'
            mock_rag_class.return_value = mock_rag_instance
            
            mock_llm_instance = Mock()
            mock_llm_instance.generate_with_context.return_value = 'Answer:AI is artificial intelligence'
            mock_llm_class.return_value = mock_llm_instance
            
            chatbot = ChatBot()
            chatbot.is_initialized = True
            
            with patch('builtins.print'):  
                result = chatbot.ask('What is AI?')
            
            assert result == 'AI is artificial intelligence'
            
            mock_rag_instance.get_context.assert_called_once_with(
                'What is AI?',
                max_tokens=2000,
                k=5,
                score_threshold=0.7
            )
            
            mock_llm_instance.generate_with_context.assert_called_once_with(
                'What is AI?',
                'Context about AI',
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1,
                stop_sequences=['</s>', '<|endoftext|>'],
                stream=False
            )

    
    @patch('core.chatbot.RAGPipeline')
    @patch('core.chatbot.LLM')
    def test_ask_exception_handling(self, mock_llm_class, mock_rag_class):
        # Test exception handling during ask()
        with patch.dict(os.environ, {}, clear=True):
            mock_rag_instance = Mock()
            mock_rag_instance.get_context.side_effect = Exception('RAG error')
            mock_rag_class.return_value = mock_rag_instance
            
            mock_llm_instance = Mock()
            mock_llm_class.return_value = mock_llm_instance
            
            chatbot = ChatBot()
            chatbot.is_initialized = True
            
            with patch('core.chatbot.logger') as mock_logger:
                result = chatbot.ask('What is AI?')
                
                assert result == 'I apologize, but I encountered an error processing your question.'
                mock_logger.error.assert_called_once()
                assert 'Error processing query' in str(mock_logger.error.call_args)
    

    @patch('core.chatbot.RAGPipeline')
    @patch('core.chatbot.LLM')
    def test_add_document_success(self, mock_llm_class, mock_rag_class, mock_env_vars):
        # Test add document successfully 
        with patch.dict(os.environ, mock_env_vars):
            mock_rag_instance = Mock()
            mock_rag_instance.add_document.return_value = None  
            mock_rag_class.return_value = mock_rag_instance
            
            mock_llm_instance = Mock()
            mock_llm_class.return_value = mock_llm_instance
            
            chatbot = ChatBot()
            result = chatbot.add_document()
            
            assert result is True
            mock_rag_instance.add_document.assert_called_once_with(
                file_path='/path/to/docs',
                force_rebuild=False
            )
    

    @patch('core.chatbot.RAGPipeline')
    @patch('core.chatbot.LLM')
    def test_add_document_failure(self, mock_llm_class, mock_rag_class):
        # Test add document failure
        with patch.dict(os.environ, {}, clear=True):
            mock_rag_instance = Mock()
            mock_rag_instance.add_document.side_effect = Exception('Document error')
            mock_rag_class.return_value = mock_rag_instance
            
            mock_llm_instance = Mock()
            mock_llm_class.return_value = mock_llm_instance
            
            chatbot = ChatBot()
            
            with patch('core.chatbot.logger') as mock_logger:
                result = chatbot.add_document()
                
                assert result is False
                mock_logger.error.assert_called_once()
                assert 'Error adding document' in str(mock_logger.error.call_args)


class TestEdgeCases:
    
    @patch('core.chatbot.RAGPipeline')
    @patch('core.chatbot.LLM')
    def test_empty_query(self, mock_llm_class, mock_rag_class):
        # Test empty query
        with patch.dict(os.environ, {}, clear=True):
            mock_rag_instance = Mock()
            mock_rag_instance.get_context.return_value = ''
            mock_rag_class.return_value = mock_rag_instance
            
            mock_llm_instance = Mock()
            mock_llm_instance.generate_with_context.return_value = 'I need more information.'
            mock_llm_class.return_value = mock_llm_instance
            
            chatbot = ChatBot()
            chatbot.is_initialized = True
            
            with pytest.raises(ValueError, match='I need more information.'):
                chatbot.ask('')
            
            mock_rag_instance.get_context.assert_not_called()

    
    @patch('core.chatbot.RAGPipeline')
    @patch('core.chatbot.LLM')
    def test_very_long_query(self, mock_llm_class, mock_rag_class):
        # Test very long query
        with patch.dict(os.environ, {'RAG_MAX_TOKENS': '1000', 'TOP_K': '5', 'RAG_THRESHOLD': '0.7'}):
            mock_rag_instance = Mock()
            mock_rag_instance.get_context.return_value = 'Relevant context'
            mock_rag_class.return_value = mock_rag_instance
            
            mock_llm_instance = Mock()
            mock_llm_instance.generate_with_context.return_value = 'Answer:Processed long query'
            mock_llm_class.return_value = mock_llm_instance
            
            chatbot = ChatBot()
            chatbot.is_initialized = True
            
            long_query = 'What is AI? ' * 1000  
            
            with patch('builtins.print'):
                result = chatbot.ask(long_query)
            
            assert result == 'Processed long query'
            mock_rag_instance.get_context.assert_called_once_with(long_query, max_tokens=1000, k=5, score_threshold=0.7)
    

    def test_malformed_env_values(self):
        
        '''Test valores malformados en variables de entorno'''
        malformed_envs = {
            'TOP_K': 'not_a_number',
            'LLM_TEMPERATURE': 'invalid_float',
            'RAG_FORCE_REBUILD': 'maybe',
            'LLM_STOP_SEQUENCES': '[unclosed list',
            'DEVICE': 'cuda',
            'DOCUMENTS_PATH': '/path/to/docs',
            'FAISS_INDEX_PATH': '/path/to/index',
            'PDF_PROCESS_MAX_WORDS': '1000',
            'EMB_MODEL_NAME': 'sentence-transformers/all-MiniLM-L6-v2',
            'RAG_CHUNK_SIZE': '512',
            'RAG_THRESHOLD': '0.7',
            'RAG_MAX_TOKENS': '2000',
            'LLM_MODEL_NAME': 'microsoft/DialoGPT-medium',
            'LLM_LOAD_IN_4BIT': 'true',
            'LLM_TORCH_DTYPE': 'float16',
            'LLM_MAX_NEW_TOKENS': '512',
            'LLM_TOP_P': '0.9',
            'LLM_TOP_K': '50',
            'LLM_REPETITION_PENALTY': '1.1',
            'LLM_STREAM': 'false'
        }
        
        with patch('core.chatbot.load_dotenv'), \
             patch('core.chatbot.RAGPipeline'), \
             patch('core.chatbot.LLM'), \
             patch.dict(os.environ, malformed_envs, clear=True):
            
            chatbot = ChatBot()
            
            assert chatbot.top_k == 'not_a_number'
            assert chatbot.llm_temperature == 'invalid_float'
            assert chatbot.rag_force_rebuild == 'maybe'
            assert chatbot.llm_stop_sequences == '[unclosed list'
