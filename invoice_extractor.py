"""
invoice_extractor.py - Production-grade medical invoice extraction pipeline

Architecture:
- ImagePreprocessor: Image enhancement for optimal OCR/Vision
- JSONParser: Robust JSON parsing with multiple fallback strategies
- ResponseValidator: Validates and cleans extracted data
- InvoiceExtractor: Main orchestrator with retry logic

Optimizations:
- Deterministic extraction (temperature=0)
- Multi-strategy JSON parsing
- Cross-validation of extracted values
- Parallel page processing support
- Comprehensive logging
"""

import logging
import time
import threading
from io import BytesIO
from typing import Dict, Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import requests
from PIL import Image

from preprocessor import ImagePreprocessor, PDFPageConverter
from parser import JSONParser, ResponseValidator
from prompts import (
    EXTRACTION_PROMPT_V1,
    RETRY_PROMPT,
    select_prompt,
    get_text_enhanced_prompt,
    GENERATION_CONFIG,
    RETRY_GENERATION_CONFIG
)

logger = logging.getLogger(__name__)

# Configuration constants
MAX_PAGES = 25
MAX_REQUEST_TIMEOUT = 180  # seconds
PAGE_TIMEOUT = 30  # seconds per page
DOWNLOAD_TIMEOUT = 60  # seconds
MAX_WORKERS = 3
API_DELAY = 1.0  # Delay between API calls
MAX_RETRIES = 2


class InvoiceExtractor:
    """
    Main extraction orchestrator for medical invoices.
    
    Provides:
    - URL-based extraction (PDF and images)
    - Multi-page PDF support with parallel processing
    - Retry logic with varied strategies
    - Comprehensive token tracking
    """
    
    def __init__(self, api_key: str):
        """
        Initialize the extractor with Gemini API key.
        
        Args:
            api_key: Google Gemini API key
        """
        # Configure Gemini
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Initialize components
        self.preprocessor = ImagePreprocessor(target_max_dim=1600)
        self.parser = JSONParser()
        self.validator = ResponseValidator()
        
        # Token tracking (thread-safe)
        self._token_lock = threading.Lock()
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        
        # Request timing
        self._request_start = None
        
        # Safety settings - disable all filters for medical content
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        
        logger.info("InvoiceExtractor initialized with Gemini 2.5 Flash")
    
    def reset_token_count(self):
        """Reset token counters for new request."""
        with self._token_lock:
            self.total_input_tokens = 0
            self.total_output_tokens = 0
    
    def _add_tokens(self, input_tokens: int, output_tokens: int):
        """Thread-safe token addition."""
        with self._token_lock:
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
    
    def get_token_usage(self) -> Dict[str, int]:
        """Get current token usage."""
        with self._token_lock:
            return {
                "input_tokens": self.total_input_tokens,
                "output_tokens": self.total_output_tokens,
                "total_tokens": self.total_input_tokens + self.total_output_tokens
            }
    
    def _check_timeout(self, stage: str = "") -> bool:
        """Check if approaching request timeout."""
        if self._request_start is None:
            return False
        elapsed = time.time() - self._request_start
        if elapsed > MAX_REQUEST_TIMEOUT - 20:
            logger.warning(f"Approaching timeout at {stage}: {elapsed:.1f}s")
            return True
        return False
    
    def extract_from_url(self, url: str) -> Dict:
        """
        Main entry point: Extract from document URL.
        
        Args:
            url: URL to PDF or image document
            
        Returns:
            Extraction results dict with pagewise_line_items and total_item_count
        """
        self.reset_token_count()
        self._request_start = time.time()
        
        timings = {}
        
        try:
            # Stage 1: Download document
            t0 = time.time()
            logger.info(f"[DOWNLOAD] Starting download...")
            
            response = requests.get(url, timeout=DOWNLOAD_TIMEOUT, stream=True)
            response.raise_for_status()
            
            content = response.content
            content_type = response.headers.get('Content-Type', '').lower()
            
            timings['download'] = time.time() - t0
            logger.info(f"[DOWNLOAD] Completed in {timings['download']:.1f}s "
                       f"({len(content)/1024:.1f}KB, type: {content_type})")
            
            # Stage 2: Detect file type and extract
            is_pdf = self._is_pdf(url, content, content_type)
            
            if is_pdf:
                logger.info("[DETECT] PDF document detected")
                result = self._extract_from_pdf(content, timings)
            else:
                logger.info("[DETECT] Image document detected")
                result = self._extract_from_image(content, timings)
            
            # Log final timings
            total_time = time.time() - self._request_start
            logger.info(f"[COMPLETE] Total time: {total_time:.1f}s, "
                       f"Items: {result.get('total_item_count', 0)}, "
                       f"Pages: {len(result.get('pagewise_line_items', []))}")
            
            return result
            
        except requests.exceptions.Timeout:
            logger.error(f"[ERROR] Download timeout after {DOWNLOAD_TIMEOUT}s")
            raise Exception(f"Document download timeout ({DOWNLOAD_TIMEOUT}s)")
        except requests.exceptions.RequestException as e:
            logger.error(f"[ERROR] Download failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"[ERROR] Extraction failed: {str(e)}")
            raise
    
    def _is_pdf(self, url: str, content: bytes, content_type: str) -> bool:
        """Determine if document is PDF."""
        return (
            'application/pdf' in content_type or
            url.lower().split('?')[0].endswith('.pdf') or
            content[:4] == b'%PDF'
        )
    
    def _extract_from_pdf(self, pdf_content: bytes, timings: dict) -> Dict:
        """
        Extract from PDF document with parallel page processing.
        """
        try:
            import fitz  # PyMuPDF
            
            t0 = time.time()
            pdf = fitz.open(stream=pdf_content, filetype="pdf")
            num_pages = min(len(pdf), MAX_PAGES)
            
            if len(pdf) > MAX_PAGES:
                logger.warning(f"[PDF] Truncating from {len(pdf)} to {MAX_PAGES} pages")
            
            logger.info(f"[PDF] Processing {num_pages} pages")
            
            # Initialize PDF converter
            converter = PDFPageConverter(zoom=2.0, max_dim=1600)
            
            # Convert all pages to images first
            page_data = []
            for page_num in range(num_pages):
                if self._check_timeout("conversion"):
                    logger.warning(f"[PDF] Timeout during conversion at page {page_num + 1}")
                    break
                
                pdf_page = pdf[page_num]
                img, text = converter.convert_page(pdf_page, page_num + 1)
                page_data.append({
                    'page_num': page_num + 1,
                    'image': img,
                    'text': text,
                    'is_digital': len(text) > 100
                })
                
                logger.debug(f"[PDF] Page {page_num + 1}: {img.size[0]}x{img.size[1]}, "
                           f"text: {len(text)} chars")
            
            pdf.close()
            timings['conversion'] = time.time() - t0
            logger.info(f"[PDF] Conversion completed in {timings['conversion']:.1f}s")
            
            # Extract from pages (parallel or sequential based on count)
            t0 = time.time()
            
            if num_pages >= 4:
                # Use parallel processing for larger PDFs
                results = self._extract_pages_parallel(page_data)
            else:
                # Sequential for small PDFs
                results = self._extract_pages_sequential(page_data)
            
            timings['extraction'] = time.time() - t0
            
            # Aggregate results
            all_pages = []
            total_items = 0
            
            for page_result in results:
                if page_result and page_result.get('bill_items'):
                    all_pages.append(page_result)
                    total_items += len(page_result['bill_items'])
            
            logger.info(f"[PDF] Extraction completed in {timings['extraction']:.1f}s: "
                       f"{total_items} items across {len(all_pages)} pages")
            
            return {
                "pagewise_line_items": all_pages,
                "total_item_count": total_items
            }
            
        except ImportError:
            logger.error("[ERROR] PyMuPDF not installed")
            raise Exception("PDF processing requires PyMuPDF")
        except Exception as e:
            logger.error(f"[ERROR] PDF processing failed: {str(e)}")
            raise
    
    def _extract_pages_parallel(self, page_data: List[dict]) -> List[dict]:
        """Extract from pages in parallel."""
        results = [None] * len(page_data)
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {}
            
            for i, data in enumerate(page_data):
                if self._check_timeout("parallel_submit"):
                    break
                
                future = executor.submit(
                    self._extract_single_page,
                    data['image'],
                    data['page_num'],
                    data['text'] if data['is_digital'] else None
                )
                futures[future] = i
                time.sleep(API_DELAY)  # Stagger API calls
            
            for future in futures:
                idx = futures[future]
                try:
                    result = future.result(timeout=PAGE_TIMEOUT)
                    results[idx] = result
                    
                    if result and result.get('bill_items'):
                        logger.info(f"[PAGE {page_data[idx]['page_num']}] "
                                   f"Extracted {len(result['bill_items'])} items")
                    else:
                        logger.info(f"[PAGE {page_data[idx]['page_num']}] No items found")
                        
                except FuturesTimeoutError:
                    logger.warning(f"[PAGE {page_data[idx]['page_num']}] "
                                  f"Timeout after {PAGE_TIMEOUT}s")
                except Exception as e:
                    logger.error(f"[PAGE {page_data[idx]['page_num']}] Error: {str(e)}")
        
        return [r for r in results if r is not None]
    
    def _extract_pages_sequential(self, page_data: List[dict]) -> List[dict]:
        """Extract from pages sequentially."""
        results = []
        
        for data in page_data:
            if self._check_timeout("sequential"):
                break
            
            result = self._extract_single_page(
                data['image'],
                data['page_num'],
                data['text'] if data['is_digital'] else None
            )
            
            if result:
                results.append(result)
                
                if result.get('bill_items'):
                    logger.info(f"[PAGE {data['page_num']}] "
                               f"Extracted {len(result['bill_items'])} items")
                else:
                    logger.info(f"[PAGE {data['page_num']}] No items found")
            
            time.sleep(API_DELAY)
        
        return results
    
    def _extract_from_image(self, image_content: bytes, timings: dict) -> Dict:
        """Extract from single image."""
        try:
            t0 = time.time()
            
            # Load and preprocess image
            img = Image.open(BytesIO(image_content))
            img = self.preprocessor.process(img, page_num=1)
            
            timings['conversion'] = time.time() - t0
            logger.info(f"[IMAGE] Size: {img.size[0]}x{img.size[1]}")
            
            # Extract
            t0 = time.time()
            result = self._extract_single_page(img, 1, None)
            timings['extraction'] = time.time() - t0
            
            items_count = len(result.get('bill_items', [])) if result else 0
            
            return {
                "pagewise_line_items": [result] if result and items_count > 0 else [],
                "total_item_count": items_count
            }
            
        except Exception as e:
            logger.error(f"[ERROR] Image processing failed: {str(e)}")
            raise
    
    def _extract_single_page(self, image: Image.Image, page_num: int,
                             page_text: Optional[str] = None) -> Optional[Dict]:
        """
        Extract from a single page with retry logic.
        
        Args:
            image: Preprocessed PIL Image
            page_num: Page number (1-indexed)
            page_text: Extracted text for digital PDFs
            
        Returns:
            Page result dict or None
        """
        empty_result = {
            "page_no": str(page_num),
            "page_type": "Bill Detail",
            "bill_items": []
        }
        
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                result = self._call_gemini(image, page_num, page_text, attempt)
                
                if result and result.get('bill_items'):
                    result['page_no'] = str(page_num)
                    return result
                
                # If no items and not last attempt, retry
                if attempt < MAX_RETRIES:
                    logger.debug(f"[PAGE {page_num}] Attempt {attempt}: No items, retrying...")
                    time.sleep(1.0)
                    
            except Exception as e:
                logger.warning(f"[PAGE {page_num}] Attempt {attempt} error: {str(e)}")
                if attempt < MAX_RETRIES:
                    time.sleep(2.0)
        
        return empty_result
    
    def _call_gemini(self, image: Image.Image, page_num: int,
                     page_text: Optional[str], attempt: int) -> Optional[Dict]:
        """
        Make a single Gemini API call with parsing.
        
        Args:
            image: PIL Image
            page_num: Page number
            page_text: Optional text context
            attempt: Attempt number (1, 2, ...)
            
        Returns:
            Parsed and validated result dict
        """
        try:
            # Select appropriate prompt
            prompt = select_prompt(page_text or "", attempt)
            
            # Select generation config
            gen_config = GENERATION_CONFIG if attempt == 1 else RETRY_GENERATION_CONFIG
            
            # Make API call
            response = self.model.generate_content(
                [prompt, image],
                generation_config=genai.types.GenerationConfig(**gen_config),
                safety_settings=self.safety_settings
            )
            
            # Track tokens
            if hasattr(response, 'usage_metadata'):
                self._add_tokens(
                    getattr(response.usage_metadata, 'prompt_token_count', 0),
                    getattr(response.usage_metadata, 'candidates_token_count', 0)
                )
            else:
                self._add_tokens(500, 200)  # Estimate
            
            # Extract response text
            text = self._get_response_text(response)
            if not text:
                logger.warning(f"[PAGE {page_num}] Empty response from Gemini")
                return None
            
            logger.debug(f"[PAGE {page_num}] Response length: {len(text)} chars")
            
            # Parse JSON
            parsed = self.parser.parse(text, page_num)
            if not parsed:
                logger.warning(f"[PAGE {page_num}] JSON parsing failed")
                return None
            
            # Validate and clean
            validated = self.validator.validate_and_clean(parsed, page_num)
            
            return validated
            
        except Exception as e:
            logger.error(f"[PAGE {page_num}] Gemini call failed: {str(e)}")
            raise
    
    def _get_response_text(self, response) -> Optional[str]:
        """Safely extract text from Gemini response."""
        try:
            if not response:
                return None
            
            if not hasattr(response, 'candidates') or not response.candidates:
                # Check for blocking
                if hasattr(response, 'prompt_feedback'):
                    feedback = response.prompt_feedback
                    if hasattr(feedback, 'block_reason'):
                        logger.warning(f"Response blocked: {feedback.block_reason}")
                return None
            
            candidate = response.candidates[0]
            
            # Check finish reason
            if hasattr(candidate, 'finish_reason'):
                reason = candidate.finish_reason
                reason_val = getattr(reason, 'value', reason)
                if reason_val in [3, 4]:  # SAFETY, RECITATION
                    logger.warning(f"Response blocked with reason: {reason_val}")
                    return None
            
            # Extract text
            if not hasattr(candidate, 'content') or not candidate.content:
                return None
            
            if not hasattr(candidate.content, 'parts') or not candidate.content.parts:
                return None
            
            text = candidate.content.parts[0].text
            return text.strip() if text else None
            
        except Exception as e:
            logger.error(f"Error extracting response text: {str(e)}")
            return None


# Convenience function for direct usage
def extract_invoice(api_key: str, url: str) -> Dict:
    """
    Convenience function to extract invoice from URL.
    
    Args:
        api_key: Gemini API key
        url: Document URL
        
    Returns:
        Extraction results
    """
    extractor = InvoiceExtractor(api_key)
    return extractor.extract_from_url(url)