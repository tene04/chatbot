import logging
import time
from pathlib import Path
from contextlib import asynccontextmanager
import os
from datetime import datetime
import shutil

import uvicorn
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from core import ChatBot
from .models import QueryRequest, QueryResponse, AddDocumentRequest, StatusResponse, ErrorResponse, AppState

logger = logging.getLogger(__name__)
app_state = AppState()


# ==============================
# 1. FASTAPI APP & MIDDLEWARE
# ==============================
# Manage app lifecycle
@asynccontextmanager
async def lifespan(app):
    '''
    Used by FastAPI to manage startup and shutdown events

    Args:
        app (FastAPI): The FastAPI application instance. Passed automatically by FastAPI,
                       not used directly in this function.
    '''
    logger.info('Starting ChatBot API...')
    try:
        app_state.chatbot = ChatBot()
        success = app_state.chatbot.initialize()
        if success:
            app_state.is_ready = True
            logger.info('ChatBot initialized successfully')
        else:
            app_state.initialization_error = 'Failed to initialize ChatBot components'
            logger.error(app_state.initialization_error)
    except Exception as e:
        app_state.initialization_error = f'Startup error: {str(e)}'
        logger.error(f'Failed to initialize ChatBot: {e}')
    
    yield
    
    logger.info('Shutting down ChatBot API...')
    if app_state.chatbot:
        del app_state.chatbot

# Create FastAPI app with custom lifecycle management
app = FastAPI(
    title='RAG+LLM ChatBot API',
    description='API for RAG-enhanced chatbot with document processing capabilities',
    lifespan=lifespan
)

# Define CORS middleware to manage communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],  
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# Dependenct to verify chatbot is ready
async def get_ready_chatbot():
    if not app_state.is_ready or not app_state.chatbot:
        error = app_state.initialization_error or 'ChatBot not prepared'
        raise HTTPException(
            status_code=503, 
            detail=f'Service unavailable: {error}'
        )
    return app_state.chatbot

# Middleware to logging requests and response
@app.middleware('http')
async def log_requests(request, call_next):
    start_time = time.time()

    logger.info(f'Request: {request.method} {request.url}')
    response = await call_next(request)

    process_time = time.time() - start_time
    logger.info(f'Response: {response.status_code} - {process_time:.3f}s')
    
    return response

# ==============================
# 2. ENDPOINTS
# ==============================
@app.get('/', tags=['Health'])
async def root():
    # Root endpoint with API information
    return {
        'message': 'RAG+LLM ChatBot API',
        'status': 'ready' if app_state.is_ready else 'initializing',
        'docs': '/docs'
    }

current_dir = Path(__file__).parent
app.mount("/frontend", StaticFiles(directory=current_dir.parent / "frontend"), name="frontend")

@app.get('/health', response_model=StatusResponse, tags=['Health'])
async def health_check():
    # Health check endpoint with detailed status
    uptime = time.time() - app_state.startup_time
    
    status_info = StatusResponse(
        status='healthy' if app_state.is_ready else 'unhealthy',
        is_ready=app_state.is_ready,
        uptime=uptime
    )
    
    if app_state.is_ready and app_state.chatbot:
        try:
            status_info.rag_info = app_state.chatbot.rag.get_status()
            status_info.llm_info = app_state.chatbot.llm.get_model_info()
        except Exception as e:
            logger.warning(f'Could not get rag or llm status: {e}')
    
    return status_info

@app.post('/ask', response_model=QueryResponse, tags=['Chat'])
async def ask_chatbot(request: QueryRequest, chatbot=Depends(get_ready_chatbot)):
    '''
    Ask a question to the chatbot
    
    Args:
        request (QueryRequest): The user's query wrapped in a Pydantic model
        chatbot (ChatBot, dependency): The ChatBot instance, provided by the 
            `get_ready_chatbot` dependency. Ensures the chatbot is initialized 
            and ready to respond
    '''
    start_time = time.time()
    
    try:
        logger.info(f'Processing query...')
        response = chatbot.ask(request.query)
        
        processing_time = time.time() - start_time
        logger.info(f'Query processed successfully in {processing_time:.3f}s')
        
        return QueryResponse(
            response=response,
            processing_time=processing_time,
            status='success'
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f'Error processing query: {e}')
        raise HTTPException(
            status_code=500, 
            detail=f'Error processing query: {str(e)}'
        )
    
@app.post('/upload_document', tags=['Documents'])
async def upload_document(file: UploadFile = File(...)):
    '''
    Upload a PDF document to the server
    
    Args:
        file (UploadFile): The PDF file to upload
    '''
    try:
        await file.seek(0)
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400, 
                detail='Only PDF files are allowed'
            )
        
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', '..'))
        documents_dir = os.path.join(ROOT_DIR, 'data', 'documents')
        os.makedirs(documents_dir, exist_ok=True)  
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        clean_name = file.filename.replace(' ', '_')
        unique_filename = f"{timestamp}_{clean_name}"
        file_path = os.path.join(documents_dir, unique_filename)
        
        with open(file_path, 'wb') as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f'Document uploaded successfully: {file_path}')
        
        return {'message': 'Document uploaded successfully', 'file_path': file_path}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Error uploading document: {e}')
        raise HTTPException(
            status_code=500,
            detail=f'Error uploading document: {str(e)}'
        )

@app.post('/add_document', tags=['Documents'])
async def add_document(request: AddDocumentRequest, background_tasks: BackgroundTasks, chatbot=Depends(get_ready_chatbot)):
    '''
    Add a document to the RAG system
    
    Args:
        request (AddDocumentRequest): Incoming request containing the document file path
        background_tasks (BackgroundTasks): FastAPI BackgroundTasks object used to run 
                          document ingestion asynchronously
        chatbot (ChatBot, dependency): The ChatBot instance, provided by the 
            `get_ready_chatbot` dependency. Ensures the chatbot is initialized 
            and ready to respond
    '''
    try:
        logger.info(f'Adding document: {request.file_path}')
        def add_doc_task():
            try:
                chatbot.rag.add_document(
                    file_path=request.file_path,
                    force_rebuild=False  
                )
                logger.info(f'Document {request.file_path} added successfully')
            except Exception as e:
                logger.error(f'Background task failed for {request.file_path}: {e}')
        
        background_tasks.add_task(add_doc_task)
        
        return {
            'message': 'Document addition started in background',
            'file_path': request.file_path,
            'status': 'processing'
        }
        
    except Exception as e:
        logger.error(f'Error adding document: {e}')
        raise HTTPException(
            status_code=500,
            detail=f'Error adding document: {str(e)}'
        )

@app.post('/reinitialize', tags=['Admin'])
async def reinitialize_chatbot():
    # Reinitialize the chatbot (admin endpoint)
    try:
        logger.info('Reinitializing chatbot...')
        
        app_state.is_ready = False
        app_state.initialization_error = None
        
        if app_state.chatbot:
            del app_state.chatbot
        
        app_state.chatbot = ChatBot()
        success = app_state.chatbot.initialize()
        
        if success:
            app_state.is_ready = True
            logger.info('ChatBot reinitialized successfully')
            return {'message': 'ChatBot reinitialized successfully'}
        else:
            app_state.initialization_error = 'Failed to reinitialize ChatBot'
            logger.error(app_state.initialization_error)
            raise HTTPException(status_code=500, detail='Failed to reinitialize ChatBot')
            
    except Exception as e:
        app_state.initialization_error = f'Reinitialization error: {str(e)}'
        logger.error(f'Failed to reinitialize ChatBot: {e}')
        raise HTTPException(status_code=500, detail=str(e))


# ==============================
# 3. EXCEPTION HANDLERS
# ==============================
@app.exception_handler(StarletteHTTPException)
async def starlette_http_exception_handler(request, exc):
    logger.warning(f'HTTP {exc.status_code}: {exc.detail}')
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=f'HTTP {exc.status_code}',
            detail=exc.detail,
            timestamp=time.time()
        ).model_dump()
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    logger.warning(f'HTTP {exc.status_code}: {exc.detail}')
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=f'HTTP {exc.status_code}',
            detail=exc.detail,
            timestamp=time.time()
        ).model_dump()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f'Unexpected error: {exc}')
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error='Internal Server Error',
            detail='An unexpected error occurred',
            timestamp=time.time()
        ).model_dump()
    )


# ==============================
# 4. RUN SERVICE
# ==============================
if __name__ == '__main__':
    uvicorn.run(
        'app.backend.API:app',  
        host='0.0.0.0',
        port=8000,
        reload=True,
        log_level='info'
    )