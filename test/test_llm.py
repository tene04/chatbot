import threading
from unittest.mock import Mock, patch

import pytest
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from core import LLM, StopOnTokens


class TestStopOnTokens:
    
    def test_init_without_stop_sequence(self):
        # test initialization without stop sequences
        mock_tokenizer = Mock()

        stop_criteria = StopOnTokens(mock_tokenizer, [])
        
        assert stop_criteria.tokenizer == mock_tokenizer
        assert stop_criteria.stop_ids == []
        mock_tokenizer.encode.assert_not_called()

    
    def test_init_with_stop_sequences(self):
        # test initialization with stop sequences
        mock_tokenizer = Mock()
        mock_tokenizer.encode.side_effect = [[100], [200], [300]]
        
        stop_criteria = StopOnTokens(mock_tokenizer, ['stop', 'end', 'finish'])
        
        assert stop_criteria.stop_ids == [[100], [200], [300]]
        assert mock_tokenizer.encode.call_count == 3

    
    def test_call_should_stop(self):
        # test that stops when the last token matches a stop sequence
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [100]
        
        stop_criteria = StopOnTokens(mock_tokenizer, ['stop'])
        
        input_ids = torch.tensor([[1, 2, 3, 100]])
        
        result = stop_criteria(input_ids, None)
        assert result is True

    
    def test_call_should_not_stop(self):
        # test that does not stop when the last token does not match a stop sequence
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [100]
        
        stop_criteria = StopOnTokens(mock_tokenizer, ['stop'])
        
        input_ids = torch.tensor([[1, 2, 3, 99]])
        
        result = stop_criteria(input_ids, None)
        assert result is False


class TestLLM:
    
    def test_init_basic(self):
        # Test initialization
        llm = LLM('test-model', 'cuda', False, torch.float16)
        
        assert llm.model_name == 'test-model'
        assert llm.device == 'cuda'
        assert llm.load_in_4bit is False
        assert llm.torch_dtype == torch.float16
        assert llm.model is None
        assert llm.tokenizer is None
        assert llm.device_map == 'cuda'

    
    @patch('core.llm.AutoTokenizer.from_pretrained')
    @patch('core.llm.AutoModelForCausalLM.from_pretrained')
    @patch('core.llm.BitsAndBytesConfig')
    def test_load_model_success_with_quantization(self, mock_quant_config, mock_model, mock_tokenizer):
        # Test successful model loading with 4-bit quantization
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = '<eos>'
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        mock_model_instance = Mock()
        mock_model.return_value = mock_model_instance
        
        mock_quant_instance = Mock()
        mock_quant_config.return_value = mock_quant_instance
        
        llm = LLM('test-model', 'cuda', True, torch.float16)
        result = llm.load_model()
        
        assert result is True
        assert llm.tokenizer == mock_tokenizer_instance
        assert llm.model == mock_model_instance
        assert mock_tokenizer_instance.pad_token == '<eos>'
        
        mock_quant_config.assert_called_once_with(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
        )

    
    @patch('core.llm.AutoTokenizer.from_pretrained')
    def test_load_model_failure(self, mock_tokenizer):
        # Test error handling during model loading
        mock_tokenizer.side_effect = Exception('Error loading model')
        
        llm = LLM('invalid-model', 'cuda', False, torch.float16)
        
        with patch('core.llm.logger') as mock_logger:
            result = llm.load_model()
            
            assert result is False
            mock_logger.error.assert_called_once()

    
    def test_truncate_prompt(self):
        # Test truncation of a prompt
        llm = LLM('test-model', 'cuda', False, torch.float16)
        llm.tokenizer = Mock()
        llm.tokenizer.encode.return_value = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  
        llm.tokenizer.decode.return_value = 'truncated prompt'
        
        result = llm._truncate_prompt('very long test prompt', max_length=5)
        
        assert result == 'truncated prompt'
        llm.tokenizer.decode.assert_called_once_with([6, 7, 8, 9, 10], skip_special_tokens=True)

    
    def test_generate_response_model_not_loaded(self):
        # Test response generation without loaded a model
        llm = LLM('test-model', 'cuda', False, torch.float16)
        
        with pytest.raises(RuntimeError, match='Model not loaded'):
            llm.generate_response('test', 10, 0.7, 0.9, 50, 1.0, [], False)
    

    def test_generate_response_parameter_validation(self):
        # Test validation of parameters
        llm = LLM('test-model', 'cuda', False, torch.float16)
        llm.model = Mock()
        llm.tokenizer = Mock()
        
        with pytest.raises(AssertionError, match='Temperature must be between 0 and 1.5'):
            llm.generate_response('test', 10, 2.0, 0.9, 50, 1.0, [], False)
        
        with pytest.raises(AssertionError, match='top_p must be between 0 and 1'):
            llm.generate_response('test', 10, 0.7, 1.5, 50, 1.0, [], False)
        
        with pytest.raises(AssertionError, match='top_k must be >= 0'):
            llm.generate_response('test', 10, 0.7, 0.9, -1, 1.0, [], False)
    
    
    @patch('core.llm.TextIteratorStreamer')
    @patch('core.llm.threading.Thread')
    def test_streaming_response(self, mock_thread, mock_streamer_class):
        # Test streaming response functionality
        llm = LLM('test-model', 'cuda', False, torch.float16)
        llm.model = Mock()
        llm.tokenizer = Mock()
        llm.model.device = 'cpu'
        llm.model.config.max_position_embeddings = 4096
        llm.tokenizer.pad_token_id = 50256
        llm.tokenizer.encode.return_value = [1, 2, 3, 4] 
        
        mock_inputs = {
            'input_ids': Mock(),
            'attention_mask': Mock()
        }
        mock_inputs_to_device = Mock()
        mock_inputs_to_device.to.return_value = mock_inputs
        llm.tokenizer.return_value = mock_inputs_to_device
        
        llm._truncate_prompt = Mock(return_value='Test prompt')
        
        mock_streamer = Mock()
        mock_streamer_class.return_value = mock_streamer
        mock_streamer.__iter__ = Mock(return_value=iter(['Hello', ' world', '!']))
        
        mock_thread_instance = Mock()
        mock_thread.return_value = mock_thread_instance
        
        result_generator = llm.generate_response(
            prompt='Test prompt',
            max_new_tokens=50,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.0,
            stop_sequences=[],
            stream=True
        )
        
        mock_streamer_class.assert_called_once_with(
            llm.tokenizer, 
            skip_special_tokens=True
        )
        
        mock_thread.assert_called_once()
        mock_thread_instance.start.assert_called_once()
        
        tokens = list(result_generator)
        assert tokens == ['Hello', ' world', '!']
        
        mock_thread_instance.join.assert_called_once()
   

    @patch('torch.cuda.empty_cache')
    def test_destructor(self, mock_empty_cache):
        # Test destructor of the model
        llm = LLM('test-model', 'cuda', False, torch.float16)
        llm.model = Mock()
        llm.tokenizer = Mock()
        
        llm.__del__()
        
        mock_empty_cache.assert_called_once()
