import torch
import logging
import threading
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

logger = logging.getLogger(__name__)


class StopOnTokens(StoppingCriteria):

    def __init__(self, tokenizer, stop_sequences):
        '''
        Initialize with tokenizer and stop sequences
        
        Args:
            tokenizer (AutoTokenizer): Tokenizer instance to encode stop sequences
            stop_sequences (list[str]): List of strings to stop generation on
        '''
        self.tokenizer = tokenizer
        self.stop_ids = [tokenizer.encode(stop, add_special_tokens=False)[0] for stop in stop_sequences]


    def __call__(self, input_ids, scores, **kwargs):
        '''
        Check if the last generated token matches any stop sequence
        
        Args:
            input_ids (Tensor): Tensor of generated token IDs
            scores, **kwargs: Not used, but required by the interface
        '''
        for stop_ids in self.stop_ids:
            if input_ids[0][-len(stop_ids):].tolist() == stop_ids:
                return True
        return False
        

class HuggingFaceLLM:

    def __init__(self, model_name, device, load_in_4bit, torch_dtype):
        """
        Initialize the LLM with optional 4-bit quantization
        
        Args:
            model_name (str): HuggingFace model identifier
            device (str): "auto", "cuda", "cpu" or specific device
            load_in_4bit (bool): Use 4-bit quantization
            torch_dtype (torch.dtype): Precision dtype (float16, bfloat16, etc.)
        """
        self.model_name = model_name
        self.device = device
        self.load_in_4bit = load_in_4bit
        self.torch_dtype = torch_dtype
        self.model = None
        self.tokenizer = None
        self.device_map = device if device != "auto" else "auto"


    def load_model(self):
        """
        Load model and tokenizer with optional quantization
        """
        try:
            quant_config = None
            if self.load_in_4bit:
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=self.torch_dtype,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                padding_side="left"
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=self.device_map,
                quantization_config=quant_config,
                torch_dtype=self.torch_dtype,
                trust_remote_code=True
            )

            logger.info(f"Model '{self.model_name}' loaded successfully")
            return True
        
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False


    def _truncate_prompt(self, prompt, max_length):
        """
        Ensure prompt fits within model's context window

        Args:
            prompt (str): Input text to truncate
            max_length (int): Maximum allowed length for the prompt 
        """
        tokens = self.tokenizer.encode(prompt, truncation=False)
        if len(tokens) > max_length:
            truncated = tokens[-max_length:]
            return self.tokenizer.decode(truncated, skip_special_tokens=True)
        return prompt


    def generate_response(self, prompt, max_new_tokens, temperature, top_p, top_k, repetition_penalty, stop_sequences, stream):
        """
        Generate text completion with advanced controls
        
        Args:
            prompt (str): Input text prompt
            max_new_tokens (int): Maximum tokens to generate
            temperature (float): Creativity control
            top_p (float): Nucleus sampling threshold
            top_k (int): Top-k sampling
            repetition_penalty (float): Penalize repeats
            stop_sequences (list[str]): List of strings to stop generation
            stream (bool): Yield tokens as they're generated
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded, call load_model() first")
        
        assert 0 <= temperature <= 1.5, "Temperature must be between 0 and 1.5"
        assert 0 <= top_p <= 1.0, "top_p must be between 0 and 1"
        assert top_k >= 0, "top_k must be >= 0"

        max_length = getattr(self.model.config, "max_position_embeddings", 4096)
        prompt = self._truncate_prompt(prompt, max_length - max_new_tokens)

        stopping_criteria = None
        if stop_sequences:
            stopping_criteria = StoppingCriteriaList([
                StopOnTokens(self.tokenizer, stop_sequences)
            ])

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        generate_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": temperature > 0,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "stopping_criteria": stopping_criteria,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

        if stream:
            streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)
            generate_kwargs["streamer"] = streamer

            thread = threading.Thread(target=self.model.generate, kwargs=generate_kwargs)
            thread.start()

            def generator():
                for new_text in streamer:
                    yield new_text
                thread.join()

            return generator()
        
        outputs = self.model.generate(**generate_kwargs)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


    def generate_with_context(self, query, context, max_new_tokens, temperature, **kwargs):
        """
        RAG-optimized generation with context
        
        Args:
            query (str): User query
            context (str): Retrieved context
            max_new_tokens (int): Response length limit
            temperature (float): Lower for factual accuracy
            **kwargs: Additional generation parameters
        """
        prompt = f"""Context: {context}\n\nQuestion: {query}\n\nAnswer:"""
        return self.generate_response(prompt, max_new_tokens=max_new_tokens, temperature=temperature, **kwargs)


    def get_model_info(self):
        """
        Return modelconfiguration
        """
        if not self.model:
            return {"status": "not_loaded"}
        
        return {
            "model": self.model_name,
            "device": str(self.model.device),
            "quantized": self.load_in_4bit,
            "dtype": str(self.torch_dtype),
            "context_length": getattr(self.model.config, "max_position_embeddings", "unknown"),
        }


    def __del__(self):
        """
        Delete model and tokenizer to free resources
        """
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        torch.cuda.empty_cache()
