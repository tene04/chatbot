import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from .chatbot import ChatBot
from .rag import RAGPipeline
from .llm import HuggingFaceLLM
from .embeddings import EmbeddingManager
from .pdf_preprocess import full_clean, split_text
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

__all__ = [
    'ChatBot',
    'RAGPipeline',
    'HuggingFaceLLM',
    'EmbeddingManager',
    'full_clean',
    'split_text'
]