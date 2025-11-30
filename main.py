"""
Invoice Extraction API - HackRx Datathon Version
Robust bill/invoice parser with Gemini Vision

Endpoint: POST /extract-bill-data
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from typing import List, Optional
import uvicorn
import os
import logging
import time
import sys

from invoice_extractor import InvoiceExtractor

# Configure logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Get API key from environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

if not GEMINI_API_KEY:
    logger.warning("=" * 60)
    logger.warning("GEMINI_API_KEY not set!")
    logger.warning("Get your FREE API key from: https://makersuite.google.com/app/apikey")
    logger.warning("=" * 60)

app = FastAPI(
    title="Invoice Data Extraction API",
    description="Extract line items from invoice images/PDFs using Google Gemini Vision",
    version="2.1.0"
)

# Global variables
extractor = None
last_response = None

def get_extractor():
    global extractor
    if extractor is None:
        if not GEMINI_API_KEY:
            raise HTTPException(
                status_code=500, 
                detail="GEMINI_API_KEY not configured. Set it as an environment variable."
            )
        extractor = InvoiceExtractor(GEMINI_API_KEY)
        logger.info("InvoiceExtractor initialized successfully")
    return extractor

# ============== Request/Response Models ==============

class ExtractionRequest(BaseModel):
    document: HttpUrl

class BillItem(BaseModel):
    item_name: str
    item_amount: float
    item_rate: Optional[float] = None
    item_quantity: Optional[float] = None

class PageLineItems(BaseModel):
    page_no: str
    page_type: str
    bill_items: List[BillItem]

class TokenUsage(BaseModel):
    total_tokens: int
    input_tokens: int
    output_tokens: int

class ExtractionData(BaseModel):
    pagewise_line_items: List[PageLineItems]
    total_item_count: int

class ExtractionResponse(BaseModel):
    is_success: bool
    token_usage: TokenUsage
    data: Optional[ExtractionData] = None
    error: Optional[str] = None

# ============== API Endpoints ==============

@app.get("/")
async def root():
    return {
        "message": "Invoice Extraction API (HackRx Datathon)",
        "version": "2.1.0",
        "status": "running",
        "endpoints": {
            "extract": "POST /extract-bill-data",
            "health": "GET /health",
            "last_response": "GET /last-response"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "invoice-extraction",
        "model": "gemini-2.5-flash",
        "api_key_configured": bool(GEMINI_API_KEY),
        "version": "2.1.0"
    }

@app.get("/last-response")
async def get_last_response():
    """Get the last extraction response for debugging"""
    if last_response:
        return last_response
    return {"message": "No extraction performed yet"}

@app.post("/extract-bill-data", response_model=ExtractionResponse)
async def extract_bill_data(request: ExtractionRequest):
    """
    Extract line items and amounts from invoice document.
    
    Accepts PDF or image URLs. Returns structured line item data.
    """
    global last_response
    
    start_time = time.time()
    document_url = str(request.document)
    
    logger.info("=" * 60)
    logger.info(f"NEW REQUEST: {document_url[:100]}...")
    logger.info("=" * 60)
    
    try:
        # Get extractor instance
        ext = get_extractor()
        
        # Perform extraction
        result = ext.extract_from_url(document_url)
        
        elapsed = time.time() - start_time
        logger.info(f"Extraction completed in {elapsed:.1f}s")
        
        if not result:
            raise Exception("Extraction returned empty result")
        
        # Get token usage
        token_usage = ext.get_token_usage()
        
        # Build pagewise line items
        pagewise_items = []
        for page in result.get("pagewise_line_items", []):
            page_items = PageLineItems(
                page_no=str(page.get("page_no", "1")),
                page_type=page.get("page_type", "Bill Detail"),
                bill_items=[
                    BillItem(
                        item_name=item["item_name"],
                        item_amount=float(item["item_amount"]),
                        item_rate=float(item["item_rate"]) if item.get("item_rate") else None,
                        item_quantity=float(item["item_quantity"]) if item.get("item_quantity") else None
                    )
                    for item in page.get("bill_items", [])
                ]
            )
            pagewise_items.append(page_items)
        
        total_items = result.get("total_item_count", 0)
        
        # Build response
        response_data = ExtractionResponse(
            is_success=True,
            token_usage=TokenUsage(
                total_tokens=token_usage["total_tokens"],
                input_tokens=token_usage["input_tokens"],
                output_tokens=token_usage["output_tokens"]
            ),
            data=ExtractionData(
                pagewise_line_items=pagewise_items,
                total_item_count=total_items
            )
        )
        
        # Store for debugging
        last_response = response_data.model_dump()
        
        logger.info(f"SUCCESS: {total_items} items across {len(pagewise_items)} pages")
        logger.info("=" * 60)
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        elapsed = time.time() - start_time
        error_msg = str(e)
        
        logger.error(f"FAILED after {elapsed:.1f}s: {error_msg}")
        logger.exception("Full traceback:")
        logger.info("=" * 60)
        
        # Return error response (keep schema intact)
        error_response = ExtractionResponse(
            is_success=False,
            token_usage=TokenUsage(total_tokens=0, input_tokens=0, output_tokens=0),
            data=None,
            error=error_msg
        )
        
        last_response = error_response.model_dump()
        
        return error_response

# ============== Run Server ==============

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    
    logger.info("=" * 60)
    logger.info("Starting Invoice Extraction API")
    logger.info(f"Port: {port}")
    logger.info(f"API Key configured: {bool(GEMINI_API_KEY)}")
    logger.info("=" * 60)
    
    if not GEMINI_API_KEY:
        print("\n" + "=" * 60)
        print("WARNING: GEMINI_API_KEY not set!")
        print("Get your FREE API key from: https://makersuite.google.com/app/apikey")
        print("Set it: export GEMINI_API_KEY='your-key-here'")
        print("=" * 60 + "\n")
    
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=port, 
        reload=False,  # Disable reload for production
        log_level="info"
    )