import pytest
import os
import shutil
import tempfile
import numpy as np
import faiss
import pickle
from unittest.mock import Mock, patch
from sentence_transformers import SentenceTransformer
from core import EmbeddingManager


class TestEmbeddingInitialization:
    
    @patch('core.embeddings.SentenceTransformer')
    def test_init_success(self, mock_sentence_transformer):
        # Test successful initialization

        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 1
        mock_sentence_transformer.return_value = mock_model
        
        em = EmbeddingManager("test-model", "cpu")
        
        assert em.model == mock_model
        assert em.dimension == 1
        assert em.index is None
        assert em.texts == []
        mock_sentence_transformer.assert_called_once_with("test-model", device="cpu")


class TestCreateEmbeddings:
    
    @pytest.fixture
    def start_embedding(self):
        # Fixture to create an instance of EmbeddingManager with a mock model
        with patch('core.embeddings.SentenceTransformer') as mock_st:
            mock_model = Mock()
            mock_model.get_sentence_embedding_dimension.return_value = 1
            mock_st.return_value = mock_model
            return EmbeddingManager("test-model", "cpu")
    

    def test_create_embeddings(self, start_embedding):
        # Test embedding creation with a list of texts
        texts = ["Hello world", "This is a test"]
        expected_embeddings = np.random.random((2, 1))
        start_embedding.model.encode.return_value = expected_embeddings

        result = start_embedding.create_embeddings(texts)
        
        np.testing.assert_array_equal(result, expected_embeddings)
        start_embedding.model.encode.assert_called_once_with(texts, convert_to_numpy=True)


    def test_create_embeddings_empty_list(self, start_embedding):
        # Test embedding creation with an empty list
        texts = []
        expected_embeddings = np.array([]).reshape(0, 1)
        start_embedding.model.encode.return_value = expected_embeddings

        result = start_embedding.create_embeddings(texts)

        np.testing.assert_array_equal(result, expected_embeddings)
        start_embedding.model.encode.assert_called_once_with(texts, convert_to_numpy=True)


class TestBuildIndex:
    
    @pytest.fixture
    def start_embedding(self):
        # the same as before, creating an instance of EmbeddingManager with a mock model
        with patch('core.embeddings.SentenceTransformer') as mock_st:
            mock_model = Mock()
            mock_model.get_sentence_embedding_dimension.return_value = 1
            mock_st.return_value = mock_model
            return EmbeddingManager("test-model", "cpu")
    

    @pytest.fixture
    def temp_dir(self):
        # Fixture to create a temporary directory for index saving
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    

    def test_build_index_success(self, start_embedding, temp_dir):
        # Test successful index building
        texts = ["Hello world", "This is a test", "Another document"]
        embeddings = np.random.random((3, 1)).astype(np.float32)
        start_embedding.model.encode.return_value = embeddings

        with patch('core.embeddings.faiss') as mock_faiss:
            mock_index = Mock()
            mock_index.ntotal = 3
            mock_faiss.IndexFlatIP.return_value = mock_index
            mock_faiss.normalize_L2 = Mock()
        
            with patch.object(start_embedding, 'save_index') as mock_save:
                start_embedding.build_index(texts, temp_dir)
                
                mock_faiss.IndexFlatIP.assert_called_once_with(1)
                mock_faiss.normalize_L2.assert_called_once()
                mock_index.add.assert_called_once()
                mock_save.assert_called_once_with(temp_dir)
                assert start_embedding.texts == texts
                assert start_embedding.index == mock_index
    

    def test_build_index_empty_texts(self, start_embedding, temp_dir):
        # Test building index with empty texts list
        texts = []
        embeddings = np.array([]).reshape(0, 1).astype(np.float32)
        start_embedding.model.encode.return_value = embeddings

        with patch('core.embeddings.faiss') as mock_faiss:
            mock_index = Mock()
            mock_index.ntotal = 0
            mock_faiss.IndexFlatIP.return_value = mock_index

            with patch.object(start_embedding, 'save_index'):
                start_embedding.build_index(texts, temp_dir)

                assert start_embedding.texts == []
                assert start_embedding.index == mock_index


class TestAddToIndex:
    
    @pytest.fixture
    def start_embedding_with_index(self):
        with patch('core.embeddings.SentenceTransformer') as mock_st:
            mock_model = Mock()
            mock_model.get_sentence_embedding_dimension.return_value = 1
            mock_st.return_value = mock_model
            
            em = EmbeddingManager("test-model", "cpu")
            em.index = Mock()
            em.index.ntotal = 5
            em.texts = ["doc1", "doc2", "doc3"]
            return em
    

    @pytest.fixture
    def temp_dir(self):
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    

    def test_add_to_index_success(self, start_embedding_with_index, temp_dir):
        # Test successful addition to index
        em = start_embedding_with_index
        new_texts = ["new doc 1", "new doc 2"]
        new_embeddings = np.random.random((2, 1)).astype(np.float32)
        em.model.encode.return_value = new_embeddings

        with patch('core.embeddings.faiss.normalize_L2') as mock_normalize, patch.object(em, 'save_index') as mock_save:
            em.add_to_index(new_texts, temp_dir)
            
            em.model.encode.assert_called_once_with(new_texts, convert_to_numpy=True)
            mock_normalize.assert_called_once()
            em.index.add.assert_called_once()
            assert em.texts == ["doc1", "doc2", "doc3", "new doc 1", "new doc 2"]
            mock_save.assert_called_once_with(temp_dir)
    

    def test_add_to_index_no_index(self):
        # Test adding to index when no index exists
        with patch('core.embeddings.SentenceTransformer') as mock_st:
            mock_model = Mock()
            mock_model.get_sentence_embedding_dimension.return_value = 1
            mock_st.return_value = mock_model
            
            em = EmbeddingManager("test-model", "cpu")
            
            with pytest.raises(ValueError, match="Index not initialized"):
                em.add_to_index(["new text"], "/tmp/test")


    def test_add_to_index_empty_texts(self, start_embedding_with_index, temp_dir):
        # Test adding empty list to index
        em = start_embedding_with_index
        original_texts = em.texts.copy()
        
        em.add_to_index([], temp_dir)
        
        assert em.texts == original_texts
        em.model.encode.assert_not_called()


class TestSearch:
    
    @pytest.fixture
    def start_embedding_with_data(self):
        with patch('core.embeddings.SentenceTransformer') as mock_st:
            mock_model = Mock()
            mock_model.get_sentence_embedding_dimension.return_value = 1
            mock_st.return_value = mock_model
            
            em = EmbeddingManager("test-model", "cpu")
            em.index = Mock()
            em.texts = ["Document 1", "Document 2", "Document 3"]
            return em
    

    def test_search_success(self, start_embedding_with_data):
        # Test successful search
        em = start_embedding_with_data
        query = "test query"
        k = 2
        
        query_embedding = np.random.random((1, 1)).astype(np.float32)
        em.model.encode.return_value = query_embedding
        
        scores = np.array([[0.9, 0.7]])
        indices = np.array([[0, 2]])
        em.index.search.return_value = (scores, indices)

        with patch('core.embeddings.faiss.normalize_L2'):
            results = em.search(query, k)
            
            expected_results = [("Document 1", 0.9), ("Document 3", 0.7)]
            assert results == expected_results
            em.model.encode.assert_called_once_with([query], convert_to_numpy=True)
            em.index.search.assert_called_once_with(query_embedding, k)
    

    def test_search_no_index(self):
        # Test search when no index exists
        with patch('core.embeddings.SentenceTransformer') as mock_st:
            mock_model = Mock()
            mock_model.get_sentence_embedding_dimension.return_value = 1
            mock_st.return_value = mock_model
            
            em = EmbeddingManager("test-model", "cpu")
            
            with pytest.raises(ValueError, match="Index not initialized"):
                em.search("query", 5)


    def test_search_no_texts(self, start_embedding_with_data):
        # Test search when no texts are indexed
        em = start_embedding_with_data
        em.texts = [] 
        
        with pytest.raises(ValueError, match="No texts indexed"):
            em.search("query", 5)


    def test_search_invalid_indices(self, start_embedding_with_data):
        # Test search with invalid indices returned
        em = start_embedding_with_data
        query = "test query"
        k = 2

        query_embedding = np.random.random((1, 1)).astype(np.float32)
        em.model.encode.return_value = query_embedding
        
        scores = np.array([[0.9, 0.7]])
        indices = np.array([[0, 100]])  
        em.index.search.return_value = (scores, indices)
        
        with patch('core.embeddings.faiss.normalize_L2'):
            results = em.search(query, k)
            
            expected_results = [("Document 1", 0.9)]
            assert results == expected_results


    def test_search_zero_k(self, start_embedding_with_data):
        # Test search with k=0
        em = start_embedding_with_data
        query_embedding = np.random.random((1, 1)).astype(np.float32)
        em.model.encode.return_value = query_embedding
        
        scores = np.array([[]])
        indices = np.array([[]])
        em.index.search.return_value = (scores, indices)

        with patch('core.embeddings.faiss.normalize_L2'):
            results = em.search("query", 0)
            assert results == []


class TestSaveAndLoadIndex:
    
    @pytest.fixture
    def start_embedding(self):
        with patch('core.embeddings.SentenceTransformer') as mock_st:
            mock_model = Mock()
            mock_model.get_sentence_embedding_dimension.return_value = 1
            mock_st.return_value = mock_model
            return EmbeddingManager("test-model", "cpu")
    
    
    @pytest.fixture
    def temp_dir(self):
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    

    @patch('core.embeddings.faiss')
    @patch('core.embeddings.pickle')
    def test_save_index_success(self, mock_pickle, mock_faiss, start_embedding, temp_dir):
        # Test successful index saving
        start_embedding.index = Mock()
        start_embedding.texts = ["doc1", "doc2"]
        
        mock_open = mock_pickle.Mock()
        
        with patch('builtins.open', mock_open):
            start_embedding.save_index(temp_dir)
            
            mock_faiss.write_index.assert_called_once()
            mock_pickle.dump.assert_called_once()
    

    @patch('core.embeddings.faiss')
    @patch('core.embeddings.pickle')
    def test_load_index_success(self, mock_pickle, mock_faiss, start_embedding, temp_dir):
        # Test successful index loading
        index_path = os.path.join(temp_dir, "index.faiss")
        texts_path = os.path.join(temp_dir, "texts.pkl")
        
        os.makedirs(temp_dir, exist_ok=True)
        with open(index_path, 'w') as f:
            f.write("mock index")
        with open(texts_path, 'w') as f:
            f.write("mock texts")
        
        mock_index = Mock()
        mock_faiss.read_index.return_value = mock_index
        mock_pickle.load.return_value = ["doc1", "doc2"]
        
        with patch('builtins.open', mock_pickle.Mock()):
            start_embedding.load_index(temp_dir)
            
            assert start_embedding.index == mock_index
            assert start_embedding.texts == ["doc1", "doc2"]
    

    def test_load_index_missing_files(self, start_embedding):
        # Test loading index when files don't exist
        non_existent_path = "/path/that/does/not/exist"
        
        with pytest.raises(FileNotFoundError):
            start_embedding.load_index(non_existent_path)


    def test_load_index_missing_index_file(self, start_embedding, temp_dir):
        # Test loading when only texts file exists
        texts_path = os.path.join(temp_dir, "texts.pkl")
        with open(texts_path, 'w') as f:
            f.write("mock texts")
        
        with pytest.raises(FileNotFoundError):
            start_embedding.load_index(temp_dir)
    

    def test_load_index_missing_texts_file(self, start_embedding, temp_dir):
        # Test loading when only index file exists
        index_path = os.path.join(temp_dir, "index.faiss")
        with open(index_path, 'w') as f:
            f.write("mock index")
        
        with pytest.raises(FileNotFoundError):
            start_embedding.load_index(temp_dir)


class TestMeanScore:
    
    @pytest.fixture
    def start_embedding_with_search(self):
        with patch('core.embeddings.SentenceTransformer') as mock_st:
            mock_model = Mock()
            mock_model.get_sentence_embedding_dimension.return_value = 1
            mock_st.return_value = mock_model
            
            em = EmbeddingManager("test-model", "cpu")
            return em


    def test_mean_score_success(self, start_embedding_with_search):
        # Test successful mean score calculation
        em = start_embedding_with_search
        
        with patch.object(em, 'search') as mock_search:
            mock_search.return_value = [
                ("doc1", 0.9),
                ("doc2", 0.7),
                ("doc3", 0.5)
            ]
            
            mean = em.mean_score("query", 3)
            expected_mean = (0.9 + 0.7 + 0.5) / 3
            assert mean == expected_mean
    

    def test_mean_score_no_results(self, start_embedding_with_search):
        # Test mean score when no results are returned
        em = start_embedding_with_search
        
        with patch.object(em, 'search') as mock_search:
            mock_search.return_value = []
            
            mean = em.mean_score("query", 3)
            assert mean == 0.0




class TestEdgeCases:
    
    @pytest.fixture
    def start_embedding(self):
        with patch('core.embeddings.SentenceTransformer') as mock_st:
            mock_model = Mock()
            mock_model.get_sentence_embedding_dimension.return_value = 1
            mock_st.return_value = mock_model
            return EmbeddingManager("test-model", "cpu")
    

    def test_unicode_text_handling(self, start_embedding):
        # Test handling of Unicode text
        texts = ["Hello 世界", "Café ñoño", "🚀 emoji text"]
        embeddings = np.random.random((3, 1)).astype(np.float32)
        start_embedding.model.encode.return_value = embeddings
        
        result = start_embedding.create_embeddings(texts)
        assert result.shape == (3, 1)
    

    def test_very_long_text(self, start_embedding):
        # Test handling of very long text
        long_text = "word " * 10000  
        embeddings = np.random.random((1, 1)).astype(np.float32)
        start_embedding.model.encode.return_value = embeddings
        
        result = start_embedding.create_embeddings([long_text])
        assert result.shape == (1, 1)

    def test_special_characters(self, start_embedding):
        # Test handling of special characters
        texts = ["", "   ", "\n\t\r", "!@#$%^&*()"]
        embeddings = np.random.random((4, 1)).astype(np.float32)
        start_embedding.model.encode.return_value = embeddings
        
        result = start_embedding.create_embeddings(texts)
        assert result.shape == (4, 1)


# at the root of your project directory, run:
# pytest test/test_embedding.py -s --disable-warnings --log-cli-level=INFO