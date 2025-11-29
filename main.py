"""
Invoice Extraction API using Google Gemini Vision (FREE)
Endpoint: POST /extract-bill-data
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from typing import List, Optional
import uvicorn
import os
import logging

from invoice_extractor import InvoiceExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get API key from environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY not set! Set it via environment variable.")

app = FastAPI(
    title="Invoice Data Extraction API",
    description="Extract line items from invoice images using Google Gemini Vision (FREE)",
    version="2.0.0"
)

# Initialize extractor (will be initialized per request if API key changes)
extractor = None

def get_extractor():
    global extractor
    if extractor is None:
        if not GEMINI_API_KEY:
            raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured")
        extractor = InvoiceExtractor(GEMINI_API_KEY)
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
    page_type: str  # "Bill Detail" | "Final Bill" | "Pharmacy"
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
        "message": "Invoice Extraction API (Gemini-Powered)",
        "version": "2.0.0",
        "endpoints": {
            "extract": "POST /extract-bill-data",
            "health": "GET /health"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "invoice-extraction",
        "model": "gemini-2.5-flash",
        "api_key_configured": bool(GEMINI_API_KEY)
    }

@app.post("/extract-bill-data", response_model=ExtractionResponse)
async def extract_bill_data(request: ExtractionRequest):
    """
    Extract line items and amounts from invoice document.
    
    Args:
        request: Contains document URL (image or PDF)
        
    Returns:
        Extracted bill data with line items and token usage
        
    Note: Large PDFs (5+ pages) may take 1-3 minutes to process.
    """
    import time
    start_time = time.time()
    
    try:
        logger.info(f"Processing document: {request.document}")
        
        # Get extractor instance
        ext = get_extractor()
        
        # Extract data
        result = ext.extract_from_url(str(request.document))
        
        elapsed = time.time() - start_time
        logger.info(f"Extraction completed in {elapsed:.1f}s")
        
        if not result:
            raise HTTPException(
                status_code=500,
                detail="Failed to extract data from document"
            )
        
        # Get token usage
        token_usage = ext.get_token_usage()
        
        # Format pagewise line items
        pagewise_items = []
        for page in result.get("pagewise_line_items", []):
            page_items = PageLineItems(
                page_no=page.get("page_no", "1"),
                page_type=page.get("page_type", "Bill Detail"),
                bill_items=[
                    BillItem(
                        item_name=item["item_name"],
                        item_amount=item["item_amount"],
                        item_rate=item.get("item_rate"),
                        item_quantity=item.get("item_quantity")
                    )
                    for item in page.get("bill_items", [])
                ]
            )
            pagewise_items.append(page_items)
        
        # Build response
        return ExtractionResponse(
            is_success=True,
            token_usage=TokenUsage(
                total_tokens=token_usage["total_tokens"],
                input_tokens=token_usage["input_tokens"],
                output_tokens=token_usage["output_tokens"]
            ),
            data=ExtractionData(
                pagewise_line_items=pagewise_items,
                total_item_count=result.get("total_item_count", 0)
            )
        )
        
    except HTTPException:
        raise
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Extraction failed after {elapsed:.1f}s: {str(e)}", exc_info=True)
        return ExtractionResponse(
            is_success=False,
            token_usage=TokenUsage(total_tokens=0, input_tokens=0, output_tokens=0),
            data=None,
            error=str(e)
        )

# ============== Run Server ==============

if __name__ == "__main__":
    # Check for API key
    if not GEMINI_API_KEY:
        print("\n" + "="*60)
        print("WARNING: GEMINI_API_KEY not set!")
        print("Get your FREE API key from: https://makersuite.google.com/app/apikey")
        print("Then set it: export GEMINI_API_KEY='your-api-key'")
        print("="*60 + "\n")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )