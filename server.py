from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import io
from dotenv import load_dotenv
import traceback
import signal
import sys
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

load_dotenv()

from utils.excel_handler import ExcelHandler
from utils.pdf_handler import PDFHandler
from utils.openai_handler import OpenAIHandler
from utils.column_processor import ColumnProcessor

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up server...")
    yield
    print("Shutting down server...")

app = FastAPI(title="AI Excel Column Mapper", lifespan=lifespan)

excel_handler = ExcelHandler()
pdf_handler = PDFHandler()
openai_handler = OpenAIHandler()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:3000",
        "http://localhost:3000",
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "*"  # Allow all origins during development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

@app.middleware("http")
async def debug_middleware(request, call_next):
    """Enhanced debug middleware with more detailed logging"""
    print(f"\n>>> Request: {request.method} {request.url.path}")
    print(f">>> Headers: {dict(request.headers)}")
    print(f">>> Origin: {request.headers.get('origin')}")
    
    try:
        response = await call_next(request)
        print(f">>> Response Status: {response.status_code}")
        if response.status_code >= 400:
            # Handle streaming response properly
            if hasattr(response, 'body'):
                print(f">>> Error Response: {response.body}")
            else:
                print(">>> Error Response: <streaming response>")
        return response
    except Exception as e:
        print(f">>> Error in request processing: {str(e)}")
        traceback.print_exc()
        raise

os.makedirs("static/assets", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return FileResponse("static/index.html")

DATA_DIR = Path(__file__).parent / 'data'
VECTOR_STORE_DIR = DATA_DIR / 'vector_store'
DEFAULT_PDF_PATH = DATA_DIR / os.getenv('DEFAULT_PDF_PATH', 'column_mapping_sov_reference.pdf')

# Create necessary directories
DATA_DIR.mkdir(exist_ok=True)
VECTOR_STORE_DIR.mkdir(exist_ok=True)

# Add in-memory state management
class ProcessingState:
    def __init__(self):
        self.processor = None
        self.columns = []
        self.samples = {}
        self.processed_columns = {}
        self.pdf_text = ""

app.state.processing = ProcessingState()

@app.post("/api/init_processing")
async def initialize_processing(
    excel_file: UploadFile = File(...),
):
    """Initialize processing with Excel file upload"""
    print("Initializing processing...")
    try:
        # Read Excel file
        excel_content = await excel_file.read()
        excel_io = io.BytesIO(excel_content)
        
        print("Reading Excel columns...")
        columns, samples = excel_handler.read_excel_columns(excel_io)

        print(f"Found {len(columns)} columns")

        # Read and process PDF with LangChain
        print(f"Reading PDF from {DEFAULT_PDF_PATH}...")
        with open(DEFAULT_PDF_PATH, 'rb') as pdf_file:
            try:
                pdf_text = pdf_handler.extract_text(pdf_file)
                print("PDF processed and indexed successfully")
            except Exception as e:
                print(f"PDF processing error: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Error processing PDF: {str(e)}"
                )

        # Initialize processing state
        print("Initializing state...")
        app.state.processing = ProcessingState()
        app.state.processing.columns = columns
        app.state.processing.samples = samples
        app.state.processing.pdf_text = pdf_text
        app.state.processing.processor = ColumnProcessor(openai_handler, pdf_text)

        return JSONResponse(content={
            "status": "initialized",
            "total_columns": len(columns),
            "columns": columns,
            "vector_store": str(VECTOR_STORE_DIR)
        })
    except Exception as e:
        print(f"Initialization error: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/process_column")
async def process_column(column_data: dict):
    try:
        if not app.state.processing.processor:
            raise HTTPException(status_code=400, detail="Processing not initialized")

        column_name = column_data.get("column_name")
        if not column_name:
            raise HTTPException(status_code=400, detail="Column name required")

        if column_name in app.state.processing.processed_columns:
            return app.state.processing.processed_columns[column_name]

        samples = app.state.processing.samples.get(column_name, [])
        result = await app.state.processing.processor.process_column(column_name, samples)
        
        # Store result
        app.state.processing.processed_columns[column_name] = result
        
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/progress")
async def get_progress():
    """Get processing progress with vector store status"""
    if not hasattr(app.state, 'processing'):
        raise HTTPException(status_code=400, detail="Processing not initialized")
    
    total = len(app.state.processing.columns)
    processed = len(app.state.processing.processed_columns)
    
    # Add vector store status
    vector_store_exists = (VECTOR_STORE_DIR / "index.faiss").exists()
    
    return {
        "total": total,
        "processed": processed,
        "percentage": (processed / total * 100) if total > 0 else 0,
        "processed_columns": list(app.state.processing.processed_columns.keys()),
        "vector_store_ready": vector_store_exists
    }

@app.post("/api/upload")
async def upload_files(excel_file: UploadFile = File(...)):
    try:
        # Initialize processing
        init_result = await initialize_processing(excel_file)
        
        if init_result["status"] != "initialized":
            raise HTTPException(status_code=500, detail="Failed to initialize processing")
        
        # For backwards compatibility, process all columns immediately
        results = {}
        validations = {}
        
        for column in init_result["columns"]:
            result = await process_column({"column_name": column})
            if result.get("error"):
                continue
                
            results[result["original_name"]] = result["mapped_name"]
            validations[result["mapped_name"]] = result["validation"]
        
        return {
            "mappings": results,
            "validation": validations,
            "samples": app.state.samples
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def handle_exit(signum, frame):
    print("\nReceived signal to terminate")
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)
    
    os.makedirs("static/assets", exist_ok=True)
    
    print("Starting server on http://localhost:3000")
    config = uvicorn.Config(
        app=app,
        host="127.0.0.1",
        port=3000,
        reload=True,
        reload_delay=1,
        log_level="info",
        workers=1
    )
    server = uvicorn.Server(config)
    try:
        asyncio.run(server.serve())
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        print(f"Error during server execution: {e}")
        traceback.print_exc()
    finally:
        print("Server shutdown complete")
