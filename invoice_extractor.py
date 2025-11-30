"""
Invoice Extractor - High Performance Version for HackRx Datathon
Target: <90s for multi-page PDFs, <10s for digital PDFs

Key optimizations:
- Parallel page processing (3-4 concurrent workers)
- Adaptive resolution (1200px default, 1400px retry)
- Digital PDF fast-path (skip image conversion)
- Per-page and request-level timeouts
- Reduced delays between API calls
"""

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import requests
from io import BytesIO
from PIL import Image
import json
import re
import logging
import time
from typing import Dict, Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from functools import partial
import threading

logger = logging.getLogger(__name__)

# Performance constants
MAX_PAGES = 20  # Hard limit on pages to process
MAX_REQUEST_TIME = 120  # seconds - fail gracefully after this
PAGE_TIMEOUT = 25  # seconds per page (including retries)
DOWNLOAD_TIMEOUT = 30  # seconds for document download
MAX_WORKERS = 4  # Parallel page processing workers
DEFAULT_MAX_DIM = 1200  # Default image resolution (faster)
RETRY_MAX_DIM = 1400  # Higher resolution for retry
DEFAULT_ZOOM = 1.5  # PDF to image zoom factor
API_DELAY = 0.8  # Reduced delay between Gemini calls (was 1.5-2.0)


class InvoiceExtractor:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self._token_lock = threading.Lock()
        
        # Request start time for timeout tracking
        self._request_start = None
        
        # Disable all safety filters
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        
        # Optimized prompt - shorter for faster processing
        self.prompt = """Extract ALL bill line items as JSON. Output ONLY valid JSON:
{"page_type":"Bill Detail","bill_items":[{"item_name":"X","item_amount":100.0,"item_rate":100.0,"item_quantity":1}]}

Rules:
- item_amount = Net/total amount (rightmost column)
- item_rate = Unit price per item
- item_quantity = Number only (ignore "No", "Nos")
- page_type: "Pharmacy" for medicines, "Final Bill" for summary, else "Bill Detail"
- SKIP totals, subtotals, headers, taxes
- Return empty bill_items if no line items found"""

    def reset_token_count(self):
        with self._token_lock:
            self.total_input_tokens = 0
            self.total_output_tokens = 0
    
    def _add_tokens(self, input_tokens: int, output_tokens: int):
        with self._token_lock:
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
    
    def get_token_usage(self) -> Dict[str, int]:
        with self._token_lock:
            return {
                "input_tokens": self.total_input_tokens,
                "output_tokens": self.total_output_tokens,
                "total_tokens": self.total_input_tokens + self.total_output_tokens
            }
    
    def _check_timeout(self, stage: str = "") -> bool:
        """Check if we're approaching the request timeout"""
        if self._request_start is None:
            return False
        elapsed = time.time() - self._request_start
        if elapsed > MAX_REQUEST_TIME - 10:  # Leave 10s buffer
            logger.warning(f"Approaching timeout at {stage}: {elapsed:.1f}s elapsed")
            return True
        return False
    
    def extract_from_url(self, url: str) -> Dict:
        """Main entry point with timing and timeout tracking"""
        self.reset_token_count()
        self._request_start = time.time()
        
        timings = {}
        
        try:
            # Stage 1: Download
            t0 = time.time()
            logger.info(f"[DOWNLOAD] Starting download...")
            
            response = requests.get(url, timeout=DOWNLOAD_TIMEOUT)
            response.raise_for_status()
            content = response.content
            content_type = response.headers.get('Content-Type', '').lower()
            
            timings['download'] = time.time() - t0
            logger.info(f"[DOWNLOAD] Completed in {timings['download']:.1f}s ({len(content)/1024:.0f}KB)")
            
            # Detect file type
            is_pdf = (
                'application/pdf' in content_type or 
                url.lower().split('?')[0].endswith('.pdf') or
                content[:4] == b'%PDF'
            )
            
            # Stage 2: Extract
            if is_pdf:
                result = self.extract_from_pdf(content, timings)
            else:
                result = self.extract_from_image(content, timings)
            
            # Log final timings
            total_time = time.time() - self._request_start
            logger.info(f"[TIMINGS] Download: {timings.get('download', 0):.1f}s, "
                       f"Conversion: {timings.get('conversion', 0):.1f}s, "
                       f"Extraction: {timings.get('extraction', 0):.1f}s, "
                       f"Total: {total_time:.1f}s")
            
            return result
            
        except requests.exceptions.Timeout:
            logger.error(f"[DOWNLOAD] Timeout after {DOWNLOAD_TIMEOUT}s")
            raise Exception(f"Document download timeout ({DOWNLOAD_TIMEOUT}s)")
        except Exception as e:
            logger.error(f"[ERROR] {str(e)}")
            raise
    
    def extract_from_pdf(self, pdf_content: bytes, timings: dict) -> Dict:
        """Extract from PDF with parallel processing"""
        try:
            import fitz
            
            t0 = time.time()
            pdf = fitz.open(stream=pdf_content, filetype="pdf")
            num_pages = min(len(pdf), MAX_PAGES)
            
            if len(pdf) > MAX_PAGES:
                logger.warning(f"PDF has {len(pdf)} pages, processing only first {MAX_PAGES}")
            
            logger.info(f"[PDF] {num_pages} pages to process")
            
            # Check if digital PDF (has selectable text)
            is_digital = self._check_if_digital_pdf(pdf, num_pages)
            
            if is_digital:
                logger.info("[PDF] Detected DIGITAL PDF - using fast text extraction")
                result = self._extract_digital_pdf(pdf, num_pages, timings)
            else:
                logger.info("[PDF] Detected SCANNED PDF - using vision extraction")
                result = self._extract_scanned_pdf(pdf, num_pages, timings)
            
            pdf.close()
            return result
            
        except ImportError:
            logger.error("PyMuPDF (fitz) not installed")
            raise Exception("PDF processing library not available")
    
    def _check_if_digital_pdf(self, pdf, num_pages: int) -> bool:
        """Check if PDF has selectable text (digital) vs scanned images"""
        total_text = 0
        pages_to_check = min(3, num_pages)  # Check first 3 pages
        
        for i in range(pages_to_check):
            text = pdf[i].get_text("text").strip()
            total_text += len(text)
        
        # If average text per page > 500 chars, likely digital
        avg_text = total_text / pages_to_check
        is_digital = avg_text > 500
        
        logger.info(f"[PDF] Text check: {avg_text:.0f} avg chars/page -> {'DIGITAL' if is_digital else 'SCANNED'}")
        return is_digital
    
    def _extract_digital_pdf(self, pdf, num_pages: int, timings: dict) -> Dict:
        """Fast extraction for digital PDFs using text + minimal vision"""
        import fitz
        
        t0 = time.time()
        all_pages = []
        total_items = 0
        
        # Process pages with text-assisted vision (parallel)
        page_data = []
        for page_num in range(num_pages):
            page = pdf[page_num]
            text = page.get_text("text").strip()
            
            # Lower resolution for digital PDFs since text is clear
            pix = page.get_pixmap(matrix=fitz.Matrix(1.2, 1.2))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Resize to smaller dimension for digital PDFs
            max_dim = 1000
            if max(img.size) > max_dim:
                ratio = max_dim / max(img.size)
                img = img.resize((int(img.size[0] * ratio), int(img.size[1] * ratio)), Image.LANCZOS)
            
            page_data.append((page_num + 1, img, text))
        
        timings['conversion'] = time.time() - t0
        logger.info(f"[CONVERSION] {num_pages} pages converted in {timings['conversion']:.1f}s")
        
        # Parallel extraction
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            for page_num, img, text in page_data:
                if self._check_timeout("digital_extract"):
                    break
                future = executor.submit(self._extract_single_page_safe, img, page_num, text, DEFAULT_MAX_DIM)
                futures.append((page_num, future))
                time.sleep(API_DELAY)  # Small delay between submissions
            
            for page_num, future in futures:
                try:
                    result = future.result(timeout=PAGE_TIMEOUT)
                    if result and result.get('bill_items'):
                        all_pages.append(result)
                        total_items += len(result['bill_items'])
                        logger.info(f"[PAGE {page_num}] Extracted {len(result['bill_items'])} items")
                    else:
                        logger.info(f"[PAGE {page_num}] No items found")
                except FuturesTimeoutError:
                    logger.warning(f"[PAGE {page_num}] Timeout after {PAGE_TIMEOUT}s")
                except Exception as e:
                    logger.error(f"[PAGE {page_num}] Error: {str(e)}")
        
        timings['extraction'] = time.time() - t0
        logger.info(f"[EXTRACTION] Completed in {timings['extraction']:.1f}s: {total_items} items across {len(all_pages)} pages")
        
        return {
            "pagewise_line_items": all_pages,
            "total_item_count": total_items
        }
    
    def _extract_scanned_pdf(self, pdf, num_pages: int, timings: dict) -> Dict:
        """Optimized extraction for scanned PDFs with parallel processing"""
        import fitz
        
        t0 = time.time()
        
        # Convert all pages to images first (faster than interleaved)
        page_images = []
        for page_num in range(num_pages):
            if self._check_timeout("conversion"):
                logger.warning(f"Timeout during conversion, processed {page_num}/{num_pages} pages")
                break
                
            page = pdf[page_num]
            
            # Adaptive zoom - standard resolution
            pix = page.get_pixmap(matrix=fitz.Matrix(DEFAULT_ZOOM, DEFAULT_ZOOM))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Resize to target resolution
            if max(img.size) > DEFAULT_MAX_DIM:
                ratio = DEFAULT_MAX_DIM / max(img.size)
                img = img.resize((int(img.size[0] * ratio), int(img.size[1] * ratio)), Image.LANCZOS)
            
            page_images.append((page_num + 1, img))
            logger.debug(f"[CONVERT] Page {page_num + 1}: {img.size[0]}x{img.size[1]}")
        
        timings['conversion'] = time.time() - t0
        logger.info(f"[CONVERSION] {len(page_images)} pages in {timings['conversion']:.1f}s")
        
        # Parallel extraction with ThreadPoolExecutor
        t0 = time.time()
        all_pages = []
        total_items = 0
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit all pages
            futures = []
            for page_num, img in page_images:
                if self._check_timeout("submit"):
                    break
                future = executor.submit(self._extract_single_page_safe, img, page_num, None, DEFAULT_MAX_DIM)
                futures.append((page_num, future))
                time.sleep(API_DELAY)  # Stagger API calls slightly
            
            # Collect results
            for page_num, future in futures:
                try:
                    result = future.result(timeout=PAGE_TIMEOUT)
                    if result and result.get('bill_items'):
                        all_pages.append(result)
                        items_count = len(result['bill_items'])
                        total_items += items_count
                        logger.info(f"[PAGE {page_num}] Extracted {items_count} items")
                    else:
                        logger.info(f"[PAGE {page_num}] No items (blank/header page)")
                except FuturesTimeoutError:
                    logger.warning(f"[PAGE {page_num}] Timeout after {PAGE_TIMEOUT}s - skipping")
                except Exception as e:
                    logger.error(f"[PAGE {page_num}] Error: {str(e)}")
        
        timings['extraction'] = time.time() - t0
        logger.info(f"[EXTRACTION] Completed in {timings['extraction']:.1f}s: {total_items} items across {len(all_pages)} pages")
        
        return {
            "pagewise_line_items": all_pages,
            "total_item_count": total_items
        }
    
    def _extract_single_page_safe(self, image: Image.Image, page_num: int, 
                                   text_hint: Optional[str], max_dim: int) -> Dict:
        """Thread-safe page extraction with retry"""
        empty_result = {"page_no": str(page_num), "page_type": "Bill Detail", "bill_items": []}
        
        try:
            # First attempt
            result = self._call_gemini(image, page_num, text_hint)
            
            if result and result.get('bill_items'):
                return result
            
            # One retry with slight delay (no resolution increase to save time)
            time.sleep(1.0)
            result = self._call_gemini(image, page_num, text_hint)
            
            return result if result else empty_result
            
        except Exception as e:
            logger.warning(f"[PAGE {page_num}] Extraction failed: {str(e)}")
            return empty_result
    
    def _call_gemini(self, image: Image.Image, page_num: int, 
                     text_hint: Optional[str] = None) -> Optional[Dict]:
        """Single Gemini API call with parsing"""
        try:
            # Build prompt with optional text context
            prompt = self.prompt
            if text_hint and len(text_hint) > 100:
                prompt += f"\n\nPage text for reference:\n{text_hint[:1500]}"
            
            response = self.model.generate_content(
                [prompt, image],
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=3000,  # Reduced for speed
                ),
                safety_settings=self.safety_settings
            )
            
            # Track tokens (thread-safe)
            if hasattr(response, 'usage_metadata'):
                self._add_tokens(
                    getattr(response.usage_metadata, 'prompt_token_count', 0),
                    getattr(response.usage_metadata, 'candidates_token_count', 0)
                )
            else:
                self._add_tokens(400, 150)
            
            # Get response text
            text = self._safe_get_text(response)
            if not text:
                return None
            
            # Parse response
            return self._parse_response(text, page_num)
            
        except Exception as e:
            logger.debug(f"[PAGE {page_num}] Gemini call error: {str(e)}")
            return None
    
    def _safe_get_text(self, response) -> Optional[str]:
        """Extract text from Gemini response"""
        try:
            if not response or not response.candidates:
                return None
            
            candidate = response.candidates[0]
            
            if hasattr(candidate, 'finish_reason'):
                reason = getattr(candidate.finish_reason, 'value', candidate.finish_reason)
                if reason in [3, 4]:  # SAFETY, RECITATION
                    return None
            
            if not hasattr(candidate, 'content') or not candidate.content:
                return None
            
            if not hasattr(candidate.content, 'parts') or not candidate.content.parts:
                return None
            
            return candidate.content.parts[0].text.strip()
        except:
            return None
    
    def _parse_response(self, text: str, page_num: int) -> Optional[Dict]:
        """Parse JSON response with fallback regex"""
        try:
            # Clean markdown
            text = re.sub(r'^```json?\s*', '', text, flags=re.IGNORECASE)
            text = re.sub(r'\s*```$', '', text)
            text = text.strip()
            
            # Try JSON parse
            try:
                match = re.search(r'\{[\s\S]*\}', text)
                if match:
                    json_text = self._fix_json(match.group())
                    data = json.loads(json_text)
                    return self._build_result(data, page_num)
            except json.JSONDecodeError:
                pass
            
            # Fallback to regex extraction
            return self._extract_items_regex(text, page_num)
            
        except Exception as e:
            logger.debug(f"[PAGE {page_num}] Parse error: {e}")
            return self._extract_items_regex(text, page_num)
    
    def _fix_json(self, text: str) -> str:
        """Fix common JSON issues"""
        # Remove newlines in strings
        result = []
        in_string = False
        escape = False
        for char in text:
            if escape:
                result.append(char)
                escape = False
                continue
            if char == '\\':
                result.append(char)
                escape = True
                continue
            if char == '"':
                in_string = not in_string
            if char in '\n\r' and in_string:
                result.append(' ')
            else:
                result.append(char)
        text = ''.join(result)
        
        # Fix trailing commas
        text = re.sub(r',\s*}', '}', text)
        text = re.sub(r',\s*]', ']', text)
        
        # Close unclosed brackets
        open_braces = text.count('{') - text.count('}')
        open_brackets = text.count('[') - text.count(']')
        
        if open_braces > 0 or open_brackets > 0:
            # Remove incomplete last item
            last_comma = text.rfind(',')
            last_item = text.rfind('{"item')
            if last_item > last_comma and last_item > 0:
                text = text[:last_item].rstrip(',').rstrip()
            
            # Recount and close
            open_braces = text.count('{') - text.count('}')
            open_brackets = text.count('[') - text.count(']')
            text += ']' * max(0, open_brackets) + '}' * max(0, open_braces)
        
        return text
    
    def _extract_items_regex(self, text: str, page_num: int) -> Optional[Dict]:
        """Extract items using regex for malformed responses"""
        try:
            items = []
            
            # Pattern: item_name then item_amount
            pattern = r'"item_name"\s*:\s*"([^"]+)"[^}]*?"item_amount"\s*:\s*([\d.]+)'
            matches = re.findall(pattern, text, re.DOTALL)
            
            for name, amount in matches:
                if not name.strip() or float(amount) <= 0:
                    continue
                
                item = {"item_name": name.strip(), "item_amount": float(amount)}
                
                # Try to find rate and quantity
                section = text[text.find(f'"{name}"'):text.find(f'"{name}"') + 250]
                
                rate_m = re.search(r'"item_rate"\s*:\s*([\d.]+)', section)
                if rate_m:
                    item["item_rate"] = float(rate_m.group(1))
                
                qty_m = re.search(r'"item_quantity"\s*:\s*([\d.]+)', section)
                if qty_m:
                    item["item_quantity"] = float(qty_m.group(1))
                
                items.append(item)
            
            # Reverse pattern: amount before name
            if not items:
                pattern2 = r'"item_amount"\s*:\s*([\d.]+)[^}]*?"item_name"\s*:\s*"([^"]+)"'
                for amount, name in re.findall(pattern2, text, re.DOTALL):
                    if float(amount) > 0 and name.strip():
                        items.append({"item_name": name.strip(), "item_amount": float(amount)})
            
            if items:
                # Deduplicate
                seen = set()
                unique = []
                for item in items:
                    key = (item["item_name"].lower(), item["item_amount"])
                    if key not in seen:
                        seen.add(key)
                        unique.append(item)
                
                # Detect page type
                page_type = "Bill Detail"
                text_lower = text.lower()
                if any(k in text_lower for k in ['pharmacy', 'medicine', 'tablet', 'capsule']):
                    page_type = "Pharmacy"
                elif any(k in text_lower for k in ['final bill', 'grand total']):
                    page_type = "Final Bill"
                
                logger.info(f"[PAGE {page_num}] Regex extracted {len(unique)} items")
                return {"page_no": str(page_num), "page_type": page_type, "bill_items": unique}
            
            return None
        except:
            return None
    
    def _build_result(self, data: dict, page_num: int) -> Dict:
        """Build result from parsed JSON"""
        result = {
            "page_no": str(page_num),
            "page_type": data.get("page_type", "Bill Detail"),
            "bill_items": []
        }
        
        if result["page_type"] not in ["Bill Detail", "Final Bill", "Pharmacy"]:
            result["page_type"] = "Bill Detail"
        
        for item in data.get("bill_items", []):
            name = str(item.get("item_name", "")).strip()
            amount = self._parse_num(item.get("item_amount"))
            
            if not name or amount <= 0:
                continue
            
            bill_item = {"item_name": name, "item_amount": amount}
            
            rate = self._parse_num(item.get("item_rate"))
            if rate > 0:
                bill_item["item_rate"] = rate
            
            qty = self._parse_qty(item.get("item_quantity"))
            if qty > 0:
                bill_item["item_quantity"] = qty
            
            result["bill_items"].append(bill_item)
        
        return result
    
    def _parse_num(self, val) -> float:
        if val is None:
            return 0.0
        if isinstance(val, (int, float)):
            return float(val)
        try:
            s = str(val).replace(',', '').replace('₹', '').replace('Rs.', '').strip()
            m = re.search(r'[\d.]+', s)
            return float(m.group()) if m else 0.0
        except:
            return 0.0
    
    def _parse_qty(self, val) -> float:
        if val is None:
            return 0.0
        if isinstance(val, (int, float)):
            return float(val)
        try:
            s = re.sub(r'\s*(No|Nos|Units?)\.?\s*', '', str(val), flags=re.IGNORECASE)
            m = re.search(r'[\d.]+', s)
            return float(m.group()) if m else 0.0
        except:
            return 0.0
    
    def extract_from_image(self, image_content: bytes, timings: dict = None) -> Dict:
        """Extract from single image"""
        if timings is None:
            timings = {}
        
        try:
            t0 = time.time()
            img = Image.open(BytesIO(image_content))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Standard resolution
            if max(img.size) > DEFAULT_MAX_DIM:
                ratio = DEFAULT_MAX_DIM / max(img.size)
                img = img.resize((int(img.size[0] * ratio), int(img.size[1] * ratio)), Image.LANCZOS)
            
            timings['conversion'] = time.time() - t0
            logger.info(f"[IMAGE] Size: {img.size[0]}x{img.size[1]}")
            
            t0 = time.time()
            result = self._extract_single_page_safe(img, 1, None, DEFAULT_MAX_DIM)
            timings['extraction'] = time.time() - t0
            
            return {
                "pagewise_line_items": [result] if result and result.get('bill_items') else [],
                "total_item_count": len(result.get('bill_items', [])) if result else 0
            }
        except Exception as e:
            logger.error(f"[IMAGE] Error: {str(e)}")
            raise