
import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer

import logging
logger = logging.getLogger(__name__)


class EmbeddingManager:

    def __init__(self, model_name, device):
        """
        Initializes the EmbeddingManager, loading the SentenceTransformer 
        model and preparing the FAISS index

        Args:
            model_name (str): Name or path of the embedding model to use
            device (str): Device to run the model on, e.g., "cpu", "cuda"
        """
        self.model = SentenceTransformer(model_name, device=device)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = None
        self.texts = []
    

    def create_embeddings(self, texts):
        """
        Creates embeddings for a list of texts using the loaded SentenceTransformer model

        Args:
            texts (List[str]): List of texts to encode as embeddings
        """
        return self.model.encode(texts, convert_to_numpy=True)


    def build_index(self, texts, save_path):
        """
        Builds a FAISS index from the provided texts and their embeddings

        Args:
            texts (List[str]): List of texts to index
        """
        try:
            logger.info(f"Creating embeddings for {len(texts)} documents...")
            embeddings = self.create_embeddings(texts)
            
            logger.debug("Initializing FAISS index...")
            self.index = faiss.IndexFlatIP(self.dimension)
            
            logger.debug("Normalizing embeddings...")
            faiss.normalize_L2(embeddings)
            
            logger.debug("Adding embeddings to index...")
            self.index.add(embeddings)
            self.texts = texts

            self.save_index(save_path)
            logger.info(f"Index created successfully with {self.index.ntotal} documents")
        except Exception as e:
            logger.error(f"Failed to build index: {str(e)}")
            raise


    def add_to_index(self, new_texts, save_path):
        """
        Add new texts to the existing FAISS index without rebuilding it from scratch
        
        Args:
            new_texts (List[str]): List of new text chunks to add
            save_path (str): Path to save the updated index and texts
        """
        if self.index is None:
            raise ValueError("Index not initialized. Load or build the index first.")

        if not new_texts:
            logger.info("No new texts provided to add.")
            return

        logger.info(f"Creating embeddings for {len(new_texts)} new docs...")
        new_embeddings = self.create_embeddings(new_texts)
        faiss.normalize_L2(new_embeddings)

        self.index.add(new_embeddings) 
        self.texts.extend(new_texts)  

        self.save_index(save_path)      
        logger.info(f"Index updated. Now contains {self.index.ntotal} docs.")


    def search(self, query, k):
        """
        Search for similar documents
        
        Args:
            query (str): Query text to search for
            k (int): Number of top results to return
        """
        if not self.index:
            raise ValueError("Index not initialized. Use build_index() first.")
        
        if len(self.texts) == 0:
            raise ValueError("No texts indexed. Please build the index first.")

        query_embedding = self.create_embeddings([query])
        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding, k)
        
        results = [(self.texts[index], float(score)) \
                   for score, index in zip(scores[0], indices[0]) \
                   if index < len(self.texts)]

        return results
    

    def save_index(self, path):
        """
        Save index and texts

        Args:
            path (str): Path to save the index and texts
        """
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        
        with open(os.path.join(path, "texts.pkl"), "wb") as f:
            pickle.dump(self.texts, f)
    

    def load_index(self, path):
        """
        Load index and texts

        Args:
            path (str): Path to load the index and texts from
        """
        index_path = os.path.join(path, "index.faiss")
        texts_path = os.path.join(path, "texts.pkl")
        
        if not os.path.exists(index_path) or not os.path.exists(texts_path):
            raise FileNotFoundError(f"Index or texts not found in {path}")
        
        self.index = faiss.read_index(index_path)
        
        with open(texts_path, "rb") as f:
            self.texts = pickle.load(f)

        logger.info(f"Index loaded with {len(self.texts)} docs")


    def mean_score(self, query, k):
        """
        Returns the mean similarity score of the top-k retrieved documents for a query
    
        Args:
            query (str): Query text to search for
            k (int): Number of top results to consider
        """
        results = self.search(query, k)
        if not results:
            return 0.0
        scores = [score for _, score in results]
        return sum(scores) / len(scores)