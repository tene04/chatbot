
import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from typing import List

class EmbeddingManager:

    def __init__(self, model_name, device):
        """
        Initializes the EmbeddingManager, loading the SentenceTransformer 
        model and preparing the FAISS index

        Args:
            model_name (str): Name or path of the embedding model to use
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


    def build_index(self, texts, save_path=None):
        """
        Builds a FAISS index from the provided texts and their embeddings

        Args:
            texts (List[str]): List of texts to index
        """
        print(f"Creating embeddings for {len(texts)} docs...")

        embeddings = self.create_embeddings(texts)
        self.index = faiss.IndexFlatIP(self.dimension)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.texts = texts
        
        if save_path is None:
            save_path = '../data/faiss_index'

        self.save_index(save_path)

        print(f"Index created with {self.index.ntotal} docs.")


    def search(self, query, k=5):
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
    

    def load_index(self, path: str):
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

        print(f"Index loaded with {len(self.texts)} docs")


    def mean_score(self, query, k=5):
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