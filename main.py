"""
Invoice Extraction API - HackRx Datathon High Performance Version
Target: <90s for multi-page PDFs, <10s for digital PDFs

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
import asyncio
from concurrent.futures import ThreadPoolExecutor

from invoice_extractor import InvoiceExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Performance constants
REQUEST_TIMEOUT = 120  # Hard timeout for entire request

# Get API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

if not GEMINI_API_KEY:
    logger.warning("=" * 50)
    logger.warning("GEMINI_API_KEY not set!")
    logger.warning("=" * 50)

app = FastAPI(
    title="Invoice Extraction API - HackRx",
    description="High-performance bill extraction using Gemini Vision",
    version="3.0.0"
)

# Thread pool for running extraction
executor = ThreadPoolExecutor(max_workers=2)

# Global state
extractor = None
last_response = None


def get_extractor():
    global extractor
    if extractor is None:
        if not GEMINI_API_KEY:
            raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured")
        extractor = InvoiceExtractor(GEMINI_API_KEY)
        logger.info("Extractor initialized")
    return extractor


# ============== Models ==============

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


# ============== Endpoints ==============

@app.get("/")
async def root():
    return {
        "message": "Invoice Extraction API - HackRx Datathon",
        "version": "3.0.0 (High Performance)",
        "features": ["Parallel processing", "Digital PDF fast-path", "Adaptive resolution"],
        "endpoints": {
            "extract": "POST /extract-bill-data",
            "health": "GET /health"
        }
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model": "gemini-2.5-flash",
        "api_key_set": bool(GEMINI_API_KEY),
        "version": "3.0.0"
    }

@app.get("/last-response")
async def get_last():
    return last_response if last_response else {"message": "No extractions yet"}


def run_extraction(url: str) -> dict:
    """Run extraction in thread (blocking)"""
    ext = get_extractor()
    return ext.extract_from_url(url)


@app.post("/extract-bill-data", response_model=ExtractionResponse)
async def extract_bill_data(request: ExtractionRequest):
    """Extract line items from invoice with timeout protection"""
    global last_response
    
    start_time = time.time()
    url = str(request.document)
    
    logger.info("=" * 60)
    logger.info(f"[REQUEST] {url[:80]}...")
    
    try:
        # Run extraction with timeout
        loop = asyncio.get_event_loop()
        
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(executor, run_extraction, url),
                timeout=REQUEST_TIMEOUT
            )
        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            logger.error(f"[TIMEOUT] Request exceeded {REQUEST_TIMEOUT}s (actual: {elapsed:.1f}s)")
            
            error_response = ExtractionResponse(
                is_success=False,
                token_usage=TokenUsage(total_tokens=0, input_tokens=0, output_tokens=0),
                error=f"Request timeout after {REQUEST_TIMEOUT}s"
            )
            last_response = error_response.model_dump()
            return error_response
        
        elapsed = time.time() - start_time
        
        if not result:
            raise Exception("Empty extraction result")
        
        # Get token usage
        ext = get_extractor()
        tokens = ext.get_token_usage()
        
        # Build response
        pagewise = []
        for page in result.get("pagewise_line_items", []):
            pagewise.append(PageLineItems(
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
            ))
        
        total_items = result.get("total_item_count", 0)
        
        response = ExtractionResponse(
            is_success=True,
            token_usage=TokenUsage(
                total_tokens=tokens["total_tokens"],
                input_tokens=tokens["input_tokens"],
                output_tokens=tokens["output_tokens"]
            ),
            data=ExtractionData(
                pagewise_line_items=pagewise,
                total_item_count=total_items
            )
        )
        
        last_response = response.model_dump()
        
        logger.info(f"[SUCCESS] {total_items} items, {len(pagewise)} pages, {elapsed:.1f}s")
        logger.info("=" * 60)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        elapsed = time.time() - start_time
        error_msg = str(e)
        
        logger.error(f"[FAILED] {error_msg} ({elapsed:.1f}s)")
        logger.info("=" * 60)
        
        error_response = ExtractionResponse(
            is_success=False,
            token_usage=TokenUsage(total_tokens=0, input_tokens=0, output_tokens=0),
            error=error_msg
        )
        last_response = error_response.model_dump()
        return error_response


# ============== Run ==============

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    
    logger.info("=" * 60)
    logger.info("Invoice Extraction API v3.0.0")
    logger.info(f"Port: {port}")
    logger.info(f"API Key: {'SET' if GEMINI_API_KEY else 'NOT SET'}")
    logger.info(f"Request Timeout: {REQUEST_TIMEOUT}s")
    logger.info("=" * 60)
    
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False, log_level="info")