import pytest
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import json
from core import RAGPipeline, EmbeddingManager


class TestRAGInitialization:
    
    @patch('core.rag.EmbeddingManager')
    def test_init_success(self, mock_embedding_manager):
        # Test successful initialization
        mock_em = Mock()
        mock_embedding_manager.return_value = mock_em
        
        rag = RAGPipeline(
            chunk_size=500,
            embedding_model="test-model",
            documents_path="/test/docs",
            index_path="/test/index",
            device="cpu"
        )
        
        assert rag.documents_path == "/test/docs"
        assert rag.index_path == "/test/index"
        assert rag.chunk_size == 500
        assert rag.embedding_manager == mock_em
        assert rag.is_initialized is False
        assert rag.texts == []
        mock_embedding_manager.assert_called_once_with("test-model", device="cpu")
    

class TestIndexExists:
    
    @pytest.fixture
    def rag_pipeline(self):
        with patch('core.rag.EmbeddingManager'):
            return RAGPipeline(500, "test-model", "/docs", "/index", "cpu")
    

    @pytest.fixture
    def temp_dir(self):
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    
    def test_index_exists_both_files_present(self, rag_pipeline, temp_dir):
        # Test when both index and texts files exist
        rag_pipeline.index_path = temp_dir
        
        (Path(temp_dir) / "index.faiss").touch()
        (Path(temp_dir) / "texts.pkl").touch()
        
        assert rag_pipeline._index_exists() is True
    

    def test_index_exists_missing_index_file(self, rag_pipeline, temp_dir):
        # Test when index file is missing
        rag_pipeline.index_path = temp_dir
        
        (Path(temp_dir) / "texts.pkl").touch()
        
        assert rag_pipeline._index_exists() is False

    
    def test_index_exists_missing_texts_file(self, rag_pipeline, temp_dir):
        # Test when texts file is missing
        rag_pipeline.index_path = temp_dir
        
        (Path(temp_dir) / "index.faiss").touch()
        
        assert rag_pipeline._index_exists() is False
    

    def test_index_exists_no_files(self, rag_pipeline, temp_dir):
        # Test when no files exist
        rag_pipeline.index_path = temp_dir
        
        assert rag_pipeline._index_exists() is False
    
    def test_index_exists_nonexistent_directory(self, rag_pipeline):
        # Test when index directory doesn't exist
        rag_pipeline.index_path = "/nonexistent/directory"
        
        assert rag_pipeline._index_exists() is False


class TestBuildNewIndex:

    @pytest.fixture
    def rag_pipeline(self):
        with patch('core.rag.EmbeddingManager') as mock_em:
            mock_em.return_value = Mock()
            return RAGPipeline(500, "test-model", "/docs", "/index", "cpu")
    

    @pytest.fixture
    def temp_dir(self):
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    

    @patch('core.rag.full_clean')
    @patch('core.rag.split_text')
    @patch('core.rag.os.listdir')
    def test_build_new_index_success(self, mock_listdir, mock_split_text, mock_full_clean, rag_pipeline, temp_dir):
        # Test successful index building
        rag_pipeline.documents_path = temp_dir
        
        mock_listdir.return_value = ["doc1.pdf", "doc2.pdf", "ignored.txt"]
        mock_full_clean.side_effect = ["cleaned text 1", "cleaned text 2"]
        mock_split_text.side_effect = [["chunk1", "chunk2"], ["chunk3", "chunk4"]]
        rag_pipeline._build_new_index()
        
        assert rag_pipeline.texts == ["chunk1", "chunk2", "chunk3", "chunk4"]
        assert mock_full_clean.call_count == 2
        assert mock_split_text.call_count == 2
        rag_pipeline.embedding_manager.build_index.assert_called_once_with(["chunk1", "chunk2", "chunk3", "chunk4"], "/index")
    

    @patch('core.rag.os.listdir')
    def test_build_new_index_no_pdfs(self, mock_listdir, rag_pipeline, temp_dir):
        # Test building index when no PDF files exist
        rag_pipeline.documents_path = temp_dir
        mock_listdir.return_value = ["file1.txt", "file2.doc"]
        
        with pytest.raises(ValueError, match="No documents were found"):
            rag_pipeline._build_new_index()


    @patch('core.rag.os.listdir')
    def test_build_new_index_empty_directory(self, mock_listdir, rag_pipeline, temp_dir):
        # Test building index with empty directory
        rag_pipeline.documents_path = temp_dir
        mock_listdir.return_value = []
        
        with pytest.raises(ValueError, match="No documents were found"):
            rag_pipeline._build_new_index()


class TestLoadExistingIndex:
    
    @pytest.fixture
    def rag_pipeline(self):
        with patch('core.rag.EmbeddingManager') as mock_em:
            mock_instance = Mock()
            mock_instance.texts = ["loaded_chunk1", "loaded_chunk2"]
            mock_em.return_value = mock_instance
            return RAGPipeline(500, "test-model", "/docs", "/index", "cpu")
    
    
    def test_load_existing_index_success(self, rag_pipeline):
        # Test successful loading of existing index
        rag_pipeline._load_existing_index()
        
        rag_pipeline.embedding_manager.load_index.assert_called_once_with("/index")
        assert rag_pipeline.texts == ["loaded_chunk1", "loaded_chunk2"]
    

    def test_load_existing_index_failure(self, rag_pipeline):
        # Test loading when index loading fails
        rag_pipeline.embedding_manager.load_index.side_effect = Exception("Load failed")
        
        with pytest.raises(Exception, match="Load failed"):
            rag_pipeline._load_existing_index()


class TestInitialize:
    
    @pytest.fixture
    def rag_pipeline(self):
        with patch('core.rag.EmbeddingManager'):
            return RAGPipeline(500, "test-model", "/docs", "/index", "cpu")
    

    def test_initialize_force_rebuild_true(self, rag_pipeline):
        # Test initialization with force_rebuild=True
        with patch.object(rag_pipeline, '_index_exists', return_value=True), \
             patch.object(rag_pipeline, '_build_new_index') as mock_build, \
             patch.object(rag_pipeline, '_load_existing_index') as mock_load:
            
            result = rag_pipeline.initialize(force_rebuild=True)
            
            assert result is True
            assert rag_pipeline.is_initialized is True
            mock_build.assert_called_once()
            mock_load.assert_not_called()
    

    def test_initialize_existing_index_no_force(self, rag_pipeline):
        # Test initialization with existing index and no force rebuild
        with patch.object(rag_pipeline, '_index_exists', return_value=True), \
             patch.object(rag_pipeline, '_build_new_index') as mock_build, \
             patch.object(rag_pipeline, '_load_existing_index') as mock_load:
            
            result = rag_pipeline.initialize(force_rebuild=False)
            
            assert result is True
            assert rag_pipeline.is_initialized is True
            mock_load.assert_called_once()
            mock_build.assert_not_called()
    

    def test_initialize_no_existing_index(self, rag_pipeline):
        # Test initialization with no existing index
        with patch.object(rag_pipeline, '_index_exists', return_value=False), \
             patch.object(rag_pipeline, '_build_new_index') as mock_build, \
             patch.object(rag_pipeline, '_load_existing_index') as mock_load:
            
            result = rag_pipeline.initialize(force_rebuild=False)
            
            assert result is True
            assert rag_pipeline.is_initialized is True
            mock_build.assert_called_once()
            mock_load.assert_not_called()
    

class TestSearch:
    
    @pytest.fixture
    def initialized_rag(self):
        with patch('core.rag.EmbeddingManager') as mock_em:
            mock_instance = Mock()
            mock_instance.search.return_value = [
                ("relevant text 1", 0.9),
                ("relevant text 2", 0.7),
                ("not so relevant text", 0.4)
            ]
            mock_em.return_value = mock_instance
            
            rag = RAGPipeline(500, "test-model", "/docs", "/index", "cpu")
            rag.is_initialized = True
            return rag
    
  
    def test_search_no_results_above_threshold(self, initialized_rag):
        # Test search when no results meet threshold
        initialized_rag.embedding_manager.search.return_value = [("low score text", 0.3), ("another low score", 0.2)]
        
        results = initialized_rag.search("test query", k=2, score_threshold=0.5)
        
        assert results == []
    

    def test_search_not_initialized(self):
        # Test search when pipeline is not initialized
        with patch('core.rag.EmbeddingManager'):
            rag = RAGPipeline(500, "test-model", "/docs", "/index", "cpu")
            
            with pytest.raises(ValueError, match="RAG Pipeline not initialized"):
                rag.search("query", k=5, score_threshold=0.5)
    
    
    def test_search_empty_results(self, initialized_rag):
        # Test search with empty results from embedding manager
        initialized_rag.embedding_manager.search.return_value = []
        
        results = initialized_rag.search("test query", k=5, score_threshold=0.5)
        
        assert results == []
    
    
class TestGetContext:
    
    @pytest.fixture
    def initialized_rag(self):
        with patch('core.rag.EmbeddingManager'):
            rag = RAGPipeline(500, "test-model", "/docs", "/index", "cpu")
            rag.is_initialized = True
            return rag
    
    def test_get_context_success(self, initialized_rag):
        # Test successful context retrieval
        mock_results = [
            {"content": "Short text", "score": 0.9},
            {"content": "Another short text", "score": 0.8},
            {"content": "Third piece of text", "score": 0.7}
        ]
        
        with patch.object(initialized_rag, 'search', return_value=mock_results):
            context = initialized_rag.get_context("query", max_tokens=100, k=3)
            
            expected_context = "Short text\n\nAnother short text\n\nThird piece of text"
            assert context == expected_context
    

    def test_get_context_token_limit(self, initialized_rag):
        # Test context retrieval with token limit
        long_text = " ".join(["word"] * 50)  
        mock_results = [
            {"content": "Short text", "score": 0.9},  
            {"content": long_text, "score": 0.8},    
            {"content": "More text", "score": 0.7}   
        ]
        
        with patch.object(initialized_rag, 'search', return_value=mock_results):
            context = initialized_rag.get_context("query", max_tokens=10, k=3)
            
            assert context == "Short text"
    
    
    def test_get_context_empty_results(self, initialized_rag):
        # Test context retrieval with empty search results
        with patch.object(initialized_rag, 'search', return_value=[]):
            context = initialized_rag.get_context("query", max_tokens=100, k=3)
            
            assert context == ""
    

class TestAddDocument:
    
    @pytest.fixture
    def initialized_rag(self):
        with patch('core.rag.EmbeddingManager') as mock_em:
            mock_instance = Mock()
            mock_em.return_value = mock_instance
            
            rag = RAGPipeline(500, "test-model", "/docs", "/index", "cpu")
            rag.is_initialized = True
            rag.texts = ["existing_chunk1", "existing_chunk2"]
            return rag
        
    
    @pytest.fixture
    def temp_dir(self):
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    

    @pytest.fixture
    def sample_pdf_file(self, temp_dir):
        pdf_file = Path(temp_dir) / "test.pdf"
        pdf_file.write_text("dummy pdf content")  
        return str(pdf_file)


    @patch('core.rag.full_clean')
    @patch('core.rag.split_text')
    @patch('core.rag.shutil.copy')
    def test_add_document_no_rebuild(self, mock_copy, mock_split_text, mock_full_clean, initialized_rag, temp_dir, sample_pdf_file):
        # Test adding document without rebuilding index
        initialized_rag.documents_path = temp_dir
        
        mock_full_clean.return_value = "cleaned new document text"
        mock_split_text.return_value = ["new_chunk1", "new_chunk2"]
        
        initialized_rag.add_document(sample_pdf_file, force_rebuild=False)

        mock_copy.assert_called_once()
        mock_full_clean.assert_called_once()
        mock_split_text.assert_called_once_with("cleaned new document text", max_words=500)
        initialized_rag.embedding_manager.add_to_index.assert_called_once_with(["new_chunk1", "new_chunk2"], "/index")
        expected_texts = ["existing_chunk1", "existing_chunk2", "new_chunk1", "new_chunk2"]
        assert initialized_rag.texts == expected_texts
    

    def test_add_document_not_initialized(self):
        # Test adding document when pipeline is not initialized
        with patch('core.rag.EmbeddingManager'):
            rag = RAGPipeline(500, "test-model", "/docs", "/index", "cpu")
            
            with pytest.raises(ValueError, match="RAG Pipeline not initialized"):
                rag.add_document("/path/to/file.pdf", force_rebuild=False)
    

    def test_add_document_file_not_found(self, initialized_rag):
        # Test adding document when file doesn't exist
        with pytest.raises(FileNotFoundError, match="File not found"):
            initialized_rag.add_document("/nonexistent/file.pdf", force_rebuild=False)


class TestEdgeCases:
    
    @pytest.fixture
    def temp_dir(self):
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_case_insensitive_pdf_detection(self, temp_dir):
        # Test that PDF files with different cases are detected
        with patch('core.rag.EmbeddingManager'):
            rag = RAGPipeline(500, "test-model", temp_dir, "/index", "cpu")
            
            (Path(temp_dir) / "file1.PDF").touch()
            (Path(temp_dir) / "file2.pdf").touch()
            (Path(temp_dir) / "file3.Pdf").touch()
            (Path(temp_dir) / "file4.txt").touch() 
            
            with patch('core.rag.full_clean', return_value="text"), \
                 patch('core.rag.split_text', return_value=["chunk"]):

                rag._build_new_index()
                
                assert len(rag.texts) == 3


    @patch('core.rag.full_clean')
    @patch('core.rag.split_text')
    def test_empty_chunks_after_processing(self, mock_split_text, mock_full_clean, temp_dir):
        # Test behavior when documents produce no chunks
        with patch('core.rag.EmbeddingManager'):
            rag = RAGPipeline(500, "test-model", temp_dir, "/index", "cpu")
            
            (Path(temp_dir) / "empty.pdf").touch()

            mock_full_clean.return_value = ""


# at the root of your project directory, run:
# pytest test/test_rag.py -s --disable-warnings --log-cli-level=INFO