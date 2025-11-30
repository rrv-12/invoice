"""
main.py - FastAPI application for medical invoice extraction

Features:
- POST /extract-bill-data endpoint
- Comprehensive error handling
- Request timeout protection
- Structured logging
- Debug endpoints
"""

import asyncio
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl, Field

from invoice_extractor import InvoiceExtractor

# ============== Logging Configuration ==============

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Reduce noise from third-party libraries
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('google').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# ============== Configuration ==============

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
REQUEST_TIMEOUT = 150  # seconds - hard timeout for API requests
VERSION = "4.0.0"

if not GEMINI_API_KEY:
    logger.warning("=" * 60)
    logger.warning("GEMINI_API_KEY environment variable not set!")
    logger.warning("Get your API key from: https://makersuite.google.com/app/apikey")
    logger.warning("=" * 60)

# ============== FastAPI App ==============

app = FastAPI(
    title="Medical Invoice Extraction API",
    description="Extract line items from medical bills using Gemini Vision",
    version=VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thread pool for extraction
executor = ThreadPoolExecutor(max_workers=2)

# Global state
_extractor: Optional[InvoiceExtractor] = None
_last_response: Optional[dict] = None

# ============== Request/Response Models ==============

class ExtractionRequest(BaseModel):
    """Request model for extraction endpoint."""
    document: HttpUrl = Field(..., description="URL to PDF or image document")


class BillItem(BaseModel):
    """Single line item from invoice."""
    item_name: str = Field(..., description="Item/service description")
    item_amount: float = Field(..., ge=0, description="Net amount")
    item_rate: Optional[float] = Field(None, ge=0, description="Unit rate")
    item_quantity: Optional[float] = Field(None, ge=0, description="Quantity")


class PageLineItems(BaseModel):
    """Extraction results for a single page."""
    page_no: str = Field(..., description="Page number")
    page_type: str = Field(..., description="Type of page content")
    bill_items: List[BillItem] = Field(default_factory=list)


class TokenUsage(BaseModel):
    """Token usage statistics."""
    total_tokens: int = Field(default=0)
    input_tokens: int = Field(default=0)
    output_tokens: int = Field(default=0)


class ExtractionData(BaseModel):
    """Extracted data container."""
    pagewise_line_items: List[PageLineItems] = Field(default_factory=list)
    total_item_count: int = Field(default=0, ge=0)


class ExtractionResponse(BaseModel):
    """Response model for extraction endpoint."""
    is_success: bool = Field(..., description="Whether extraction succeeded")
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    data: Optional[ExtractionData] = Field(None, description="Extracted data")
    error: Optional[str] = Field(None, description="Error message if failed")


# ============== Helper Functions ==============

def get_extractor() -> InvoiceExtractor:
    """Get or create extractor instance."""
    global _extractor
    
    if _extractor is None:
        if not GEMINI_API_KEY:
            raise HTTPException(
                status_code=500,
                detail="GEMINI_API_KEY not configured"
            )
        _extractor = InvoiceExtractor(GEMINI_API_KEY)
        logger.info("InvoiceExtractor instance created")
    
    return _extractor


def run_extraction(url: str) -> dict:
    """Run extraction in thread pool."""
    extractor = get_extractor()
    return extractor.extract_from_url(url)


def build_response(result: dict, extractor: InvoiceExtractor) -> ExtractionResponse:
    """Build response from extraction result."""
    token_usage = extractor.get_token_usage()
    
    pagewise_items = []
    for page in result.get("pagewise_line_items", []):
        items = []
        for item in page.get("bill_items", []):
            items.append(BillItem(
                item_name=item.get("item_name", ""),
                item_amount=float(item.get("item_amount", 0)),
                item_rate=float(item["item_rate"]) if item.get("item_rate") else None,
                item_quantity=float(item["item_quantity"]) if item.get("item_quantity") else None
            ))
        
        pagewise_items.append(PageLineItems(
            page_no=str(page.get("page_no", "1")),
            page_type=page.get("page_type", "Bill Detail"),
            bill_items=items
        ))
    
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


# ============== API Endpoints ==============

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Medical Invoice Extraction API",
        "version": VERSION,
        "status": "running",
        "api_key_configured": bool(GEMINI_API_KEY),
        "endpoints": {
            "extract": "POST /extract-bill-data",
            "health": "GET /health",
            "docs": "GET /docs",
            "last_response": "GET /last-response"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "invoice-extraction",
        "version": VERSION,
        "model": "gemini-2.5-flash",
        "api_key_configured": bool(GEMINI_API_KEY)
    }


@app.get("/last-response")
async def get_last_response():
    """Get last extraction response for debugging."""
    if _last_response:
        return _last_response
    return {"message": "No extraction performed yet"}


@app.post("/extract-bill-data", response_model=ExtractionResponse)
async def extract_bill_data(request: ExtractionRequest):
    """
    Extract line items from medical invoice.
    
    Accepts PDF or image URLs. Returns structured line item data.
    
    - **document**: URL to the invoice document (PDF or image)
    
    Returns extracted line items organized by page.
    """
    global _last_response
    
    start_time = time.time()
    document_url = str(request.document)
    
    logger.info("=" * 70)
    logger.info(f"[REQUEST] New extraction request")
    logger.info(f"[REQUEST] URL: {document_url[:100]}{'...' if len(document_url) > 100 else ''}")
    
    try:
        # Run extraction with timeout
        loop = asyncio.get_event_loop()
        
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(executor, run_extraction, document_url),
                timeout=REQUEST_TIMEOUT
            )
        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            logger.error(f"[TIMEOUT] Request exceeded {REQUEST_TIMEOUT}s (actual: {elapsed:.1f}s)")
            
            error_response = ExtractionResponse(
                is_success=False,
                token_usage=TokenUsage(),
                error=f"Request timeout after {REQUEST_TIMEOUT}s"
            )
            _last_response = error_response.model_dump()
            return error_response
        
        elapsed = time.time() - start_time
        
        if not result:
            raise Exception("Extraction returned empty result")
        
        # Build response
        extractor = get_extractor()
        response = build_response(result, extractor)
        
        # Store for debugging
        _last_response = response.model_dump()
        
        total_items = result.get("total_item_count", 0)
        num_pages = len(result.get("pagewise_line_items", []))
        
        logger.info(f"[SUCCESS] Extracted {total_items} items from {num_pages} pages in {elapsed:.1f}s")
        logger.info("=" * 70)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        elapsed = time.time() - start_time
        error_msg = str(e)
        
        logger.error(f"[FAILED] {error_msg} (after {elapsed:.1f}s)")
        logger.exception("Full traceback:")
        logger.info("=" * 70)
        
        error_response = ExtractionResponse(
            is_success=False,
            token_usage=TokenUsage(),
            error=error_msg
        )
        
        _last_response = error_response.model_dump()
        return error_response


# ============== Application Entry Point ==============

def main():
    """Run the application."""
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info("=" * 70)
    logger.info(f"Starting Medical Invoice Extraction API v{VERSION}")
    logger.info(f"Host: {host}, Port: {port}")
    logger.info(f"API Key configured: {bool(GEMINI_API_KEY)}")
    logger.info(f"Request timeout: {REQUEST_TIMEOUT}s")
    logger.info("=" * 70)
    
    if not GEMINI_API_KEY:
        print("\n" + "=" * 60)
        print("WARNING: GEMINI_API_KEY not set!")
        print("Set it: export GEMINI_API_KEY='your-api-key'")
        print("Get key: https://makersuite.google.com/app/apikey")
        print("=" * 60 + "\n")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
        access_log=True
    )


if __name__ == "__main__":
    main()