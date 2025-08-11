import os
import shutil
from pathlib import Path
from .pdf_preprocess import full_clean, split_text
from .embeddings import EmbeddingManager

import logging
logger = logging.getLogger(__name__)


class RAGPipeline:
    
    def __init__(self, chunk_size, embedding_model, documents_path, index_path, device):
        """
        Initializes the RAG pipeline with paths for documents and index,
        and the embedding model to use.

        Args:
            chunk_size (int): Size of text chunks to create from documents
            embedding_model (str): Name or path of the embedding model to use
            documents_path (str): Path to the directory containing documents
            index_path (str): Path to save the FAISS index and metadata
        """
        self.documents_path = documents_path
        self.index_path = index_path
        self.chunk_size = chunk_size
        self.embedding_manager = EmbeddingManager(embedding_model, device=device)
        self.is_initialized = False
        self.texts = []
    

    def _index_exists(self):
        """
        Check if a valid index exists
        """
        index_file = Path(self.index_path) / "index.faiss"
        texts_file = Path(self.index_path) / "texts.pkl"
        
        return all(f.exists() for f in [index_file, texts_file])
    

    def _build_new_index(self):
        """
        Build a new index from the documents 
        """
        os.makedirs(self.documents_path, exist_ok=True)

        logger.info(f"Preprocessing documents in {self.documents_path}...")
        self.texts = []
        for filename in os.listdir(self.documents_path):
            if filename.lower().endswith('.pdf'):
                file_path = Path(self.documents_path) / filename
                cleaned_text = full_clean(file_path)
                chunks = split_text(cleaned_text, max_words=self.chunk_size)
                self.texts.extend(chunks)
        
        if not self.texts:
            raise ValueError(f"No documents were found in {self.documents_path}")

        logger.info("Creating embedding index...")
        self.embedding_manager.build_index(self.texts, self.index_path)
    

    def _load_existing_index(self):
        """
        Load the existing FAISS index
        """
        self.embedding_manager.load_index(self.index_path)
        self.texts = self.embedding_manager.texts   
        

    def initialize(self, force_rebuild):
        """Initialize the RAG pipeline, loading or building the index as needed
        
        Args:
            force_rebuild (bool): If True, rebuild the index even if it exists
        """
        try:
            if not force_rebuild and self._index_exists():
                logger.info("Loading existing index...")
                self._load_existing_index()
            else:
                logger.info("Building new index...")
                self._build_new_index()
            
            self.is_initialized = True
            logger.info(f"RAG Pipeline initialized with {len(self.texts)} chunks")
            return True
            
        except Exception as e:
            logger.info(f"Error initializing RAG Pipeline: {e}")
            return False


    def search(self, query, k, score_threshold):
        """
        Search for relevant documents based on a query
        
        Args:
            query (str): Query text to search for
            k (int): Number of top results to return
            score_threshold (float): Minimum score to consider a result relevant
        """
        if not self.is_initialized:
            raise ValueError("RAG Pipeline not initialized. Run initialize() first.")
        
        raw_results = self.embedding_manager.search(query, k)

        filtered_results = [{'content': content, 'score': score} for content, score in raw_results
                            if score >= score_threshold]

        return filtered_results


    def get_context(self, query, max_tokens, k, score_threshold):
        """
        Obtain relevant context for a query, formatted for LLM input
        
        Args:
            query (str): User's query to search for
            max_tokens (int): Maximum number of tokens for the context
            k (int): Number of top results to return
            score_threshold (float): Minimum score to consider a result relevant
        """
        results = self.search(query, k, score_threshold)

        context_parts = []
        current_tokens = 0
        
        for result in results:
            chunk = result['content']
            chunk_tokens = len(chunk.split())
            if current_tokens + chunk_tokens > max_tokens:
                break
            context_parts.append(chunk)
            current_tokens += chunk_tokens

        return "\n\n".join(context_parts)
    
    
    def add_document(self, file_path, force_rebuild):
        """
        Add a new document to the system
        
        Args:
            file_path (str): Path to the PDF file to add
            force_rebuild (bool): If True, rebuild the index after adding
        """
        if not self.is_initialized:
            raise ValueError("RAG Pipeline not initialized. Run initialize() first.")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        os.makedirs(self.documents_path, exist_ok=True)

        file_name = Path(file_path).name
        destination = Path(self.documents_path) / file_name
        logger.info(f"Adding new document: {file_name}")
        shutil.copy(file_path, destination)

        cleaned_text = full_clean(destination)
        chunks = split_text(cleaned_text, max_words=self.chunk_size)

        if force_rebuild:
            logger.info("Rebuilding entire index...")
            self._build_new_index()
        else:
            logger.info("Adding to existing index...")
            self.embedding_manager.add_to_index(chunks, self.index_path)
            self.texts.extend(chunks)


    def get_status(self):
        """
        Get current status of the RAG pipeline
        """
        return {
            "initialized": self.is_initialized,
            "document_count": len(os.listdir(self.documents_path)),
            "chunk_count": len(self.texts) if self.texts else 0,
            "documents_path": str(self.documents_path),
            "index_path": str(self.index_path)
        }
