from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException
from fastapi.responses import Response
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
import os
import logging
from pathlib import Path
from typing import List
import uuid
from datetime import datetime

from models import AnalysisResult, StoredAnalysis
from analysis_engine import ContractAnalyzer


ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB removed: API is fully stateless and returns results directly.

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Add your routes to the router instead of directly to app
@api_router.get("/")
async def root():
    return {"message": "ContractNetAI API is running"}

@api_router.post("/analyze", response_model=AnalysisResult)
async def analyze_contracts(file: UploadFile = File(...)):
    """
    Upload CSV file and analyze contract relationships
    """
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")
        
        # Read file content
        content = await file.read()
        csv_content = content.decode('utf-8')
        
        # Initialize analyzer
        analyzer = ContractAnalyzer()
        
        # Run analysis
        results = analyzer.analyze_csv(csv_content)
        
        # Stateless response; no persistence
        return results
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.error(f"Analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@api_router.get("/download/csv/{analysis_id}")
async def download_csv(analysis_id: str):
    """
    Download CSV report for a specific analysis
    """
    raise HTTPException(status_code=404, detail="Downloads disabled (no persistence enabled)")

@api_router.get("/download/txt/{analysis_id}")
async def download_txt(analysis_id: str):
    """
    Download TXT report for a specific analysis
    """
    raise HTTPException(status_code=404, detail="Downloads disabled (no persistence enabled)")

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# No DB to shut down