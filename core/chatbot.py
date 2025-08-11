import os
import torch
import ast
from dotenv import load_dotenv
from .rag import RAGPipeline
from .llm import HuggingFaceLLM
import logging

logger = logging.getLogger(__name__)


def parse_env_value(value):
    """
    Try to parse an environment variable value into its appropriate type
    """
    if value is None:
        return None
    val = value.strip()

    if val.lower() == 'true':
        return True
    if val.lower() == 'false':
        return False

    try:
        if '.' in val:
            return float(val)
        else:
            return int(val)
    except ValueError:
        pass

    try:
        parsed = ast.literal_eval(val)
        if isinstance(parsed, (list, tuple)):
            return list(parsed)
        return parsed
    except (ValueError, SyntaxError):
        pass

    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
        return val[1:-1]

    return val


class ChatBot:
    device: str
    documents_path: str
    faiss_index_path: str
    top_k: int
    pdf_process_max_words: int
    emb_model_name: str
    rag_chunk_size: int
    rag_force_rebuild: bool
    rag_threshold: float
    rag_max_tokens: int
    llm_stop_sequences: list
    llm_model_name: str
    llm_load_in_4bit: bool
    llm_torch_dtype: torch.dtype
    llm_max_new_tokens: int
    llm_temperature: float
    llm_top_p: float
    llm_top_k: int
    llm_repetition_penalty: float
    llm_stream: bool

    def __init__(self):
        """
        Initialize the ChatBot with RAG and LLM components
        """
        config_keys = [
            'DEVICE', 'DOCUMENTS_PATH', 'FAISS_INDEX_PATH', 'TOP_K',

            'PDF_PROCESS_MAX_WORDS',

            'EMB_MODEL_NAME',

            'RAG_CHUNK_SIZE', 'RAG_FORCE_REBUILD', 'RAG_THRESHOLD', 'RAG_MAX_TOKENS',

            'LLM_STOP_SEQUENCES', 'LLM_MODEL_NAME', 'LLM_LOAD_IN_4BIT', 'LLM_TORCH_DTYPE', 'LLM_MAX_NEW_TOKENS', 
            'LLM_TEMPERATURE', 'LLM_TOP_P', 'LLM_TOP_K', 'LLM_REPETITION_PENALTY', 'LLM_STREAM'
            ]
            

        env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
        load_dotenv(env_path)

        for key in config_keys:
            raw_val = os.getenv(key)
            if key == 'LLM_TORCH_DTYPE':
                setattr(self, key.lower(), getattr(torch, raw_val))
            else:
                setattr(self, key.lower(), parse_env_value(raw_val))

        self.rag = RAGPipeline(
            chunk_size=self.rag_chunk_size,
            embedding_model=self.emb_model_name,
            documents_path=self.documents_path,
            index_path=self.faiss_index_path,
            device=self.device 
        )
        
        self.llm = HuggingFaceLLM(
            model_name=self.llm_model_name,
            device=self.device,
            load_in_4bit=self.llm_load_in_4bit,
            torch_dtype=self.llm_torch_dtype
        )
        
        self.is_initialized = False


    def initialize(self):
        """
        Initialize both RAG and LLM components
        """
        logger.info("Initializing ChatBot...")
        
        if not self.rag.initialize(force_rebuild=self.rag_force_rebuild):
            logger.error("Failed to initialize RAG pipeline")
            return False
            
        if not self.llm.load_model():
            logger.error("Failed to load LLM model")
            return False
        
        self.is_initialized = True
        logger.info("ChatBot initialization complete!")

        rag_status = self.rag.get_status()
        logger.info(rag_status)
        llm_config = self.llm.get_model_info()
        logger.info(llm_config)

        return True


    def ask(self, query):
        """
        Process a query and generate a response using RAG + LLM
        
        Args:
            query (str): User's question
        """
        if not self.is_initialized:
            raise ValueError("ChatBot not initialized. Call initialize() first.")
        
        try:
            context = self.rag.get_context(query, max_tokens=self.rag_max_tokens, top_k=self.top_k, threshold=self.rag_threshold)
            response = self.llm.generate_with_context(query, context, 
                                                      max_new_tokens=self.llm_max_new_tokens, 
                                                      temperature=self.llm_temperature, 
                                                      top_p=self.llm_top_p, 
                                                      top_k=self.llm_top_k, 
                                                      repetition_penalty=self.llm_repetition_penalty, 
                                                      stop_sequences=self.llm_stop_sequences, 
                                                      stream=self.llm_stream)
            return response
        
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return "I apologize, but I encountered an error processing your question."


    def add_document(self):
        """
        Add a new document to the RAG system
        """
        try:
            self.rag.add_document(file_path=self.documents_path, force_rebuild=self.rag_force_rebuild)
            return True
        except Exception as e:
            logger.error(f"Error adding document: {e}")
            return False

