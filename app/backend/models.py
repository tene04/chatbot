import time
from typing import Optional, Dict, Any

from pydantic import BaseModel, Field

from core import ChatBot


# Set Pydantic models for API
class QueryRequest(BaseModel):
    query: str = Field(..., description='User question')

class QueryResponse(BaseModel):
    response: str
    processing_time: float
    status: str = 'success'

class AddDocumentRequest(BaseModel):
    file_path: str = Field(..., description='Specific file path to add')

class StatusResponse(BaseModel):
    status: str
    is_ready: bool
    uptime: float
    rag_info: Optional[Dict[str, Any]] = None  
    llm_info: Optional[Dict[str, Any]] = None 

class ErrorResponse(BaseModel):
    error: str
    detail: str
    timestamp: float

class AppState:
    def __init__(self):
        self.chatbot: ChatBot = None
        self.is_ready: bool = False
        self.startup_time: time = time.time()
        self.initialization_error: str = None