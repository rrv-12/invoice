"""
Invoice Extractor - Robust Version for HackRx Datathon
Key improvements:
- Higher image resolution (1500px) for better text readability
- Retry mechanism with exponential backoff
- Better error logging to diagnose failures
- Fallback text extraction for digital PDFs
- Improved prompt engineering
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

logger = logging.getLogger(__name__)

class InvoiceExtractor:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        
        # Disable all safety filters
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        
        # Improved prompt with clearer instructions
        self.prompt = """You are a medical bill/invoice data extractor. Extract ALL line items from this bill page.

OUTPUT FORMAT (JSON only, no other text):
{
  "page_type": "Bill Detail",
  "bill_items": [
    {
      "item_name": "Item description here",
      "item_amount": 123.45,
      "item_rate": 123.45,
      "item_quantity": 1
    }
  ]
}

EXTRACTION RULES:
1. Extract EVERY line item with a price/amount
2. item_amount = The net/total amount for that line (rightmost amount column)
3. item_rate = Unit price/rate per item
4. item_quantity = Just the number (ignore "No", "Nos", "Units")
5. page_type: Use "Pharmacy" for medicine pages, "Final Bill" for summary pages, "Bill Detail" for detailed item pages
6. SKIP: Headers, footers, page totals, subtotals, grand totals, tax lines
7. If no line items found, return: {"page_type": "Bill Detail", "bill_items": []}

IMPORTANT: Return ONLY valid JSON. No markdown, no explanations."""

    def reset_token_count(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
    
    def get_token_usage(self) -> Dict[str, int]:
        return {
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens
        }
    
    def extract_from_url(self, url: str) -> Dict:
        """Main entry point: Download and extract from URL"""
        self.reset_token_count()
        
        try:
            logger.info(f"Downloading document from URL...")
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            content = response.content
            content_type = response.headers.get('Content-Type', '').lower()
            
            logger.info(f"Downloaded: {len(content)} bytes, Content-Type: {content_type}")
            
            # Detect file type
            is_pdf = (
                'application/pdf' in content_type or 
                url.lower().split('?')[0].endswith('.pdf') or
                content[:4] == b'%PDF'
            )
            
            if is_pdf:
                logger.info("Detected PDF document")
                return self.extract_from_pdf(content)
            else:
                logger.info("Detected image document")
                return self.extract_from_image(content)
                
        except requests.exceptions.Timeout:
            logger.error("Download timeout after 120s")
            raise Exception("Document download timeout")
        except requests.exceptions.RequestException as e:
            logger.error(f"Download error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Extraction error: {str(e)}")
            raise
    
    def extract_from_pdf(self, pdf_content: bytes) -> Dict:
        """Extract from PDF - try text first, then vision"""
        try:
            import fitz
            
            pdf = fitz.open(stream=pdf_content, filetype="pdf")
            num_pages = len(pdf)
            logger.info(f"Processing PDF: {num_pages} pages")
            
            all_pages = []
            total_items = 0
            
            # Adaptive delay based on page count
            if num_pages <= 5:
                delay = 2.0
            elif num_pages <= 10:
                delay = 1.5
            else:
                delay = 1.2
            
            for page_num in range(num_pages):
                if page_num > 0:
                    time.sleep(delay)
                
                logger.info(f"Processing page {page_num + 1}/{num_pages}")
                
                page = pdf[page_num]
                
                # First, try to extract text (for digital PDFs)
                page_text = page.get_text("text").strip()
                has_selectable_text = len(page_text) > 100
                
                if has_selectable_text:
                    logger.info(f"Page {page_num + 1}: Found {len(page_text)} chars of selectable text")
                
                # Convert to image for vision extraction
                # KEY FIX: Higher resolution - use 2.0x zoom for better text readability
                pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                # KEY FIX: Higher max resolution - 1500px instead of 800px
                max_dim = 1500
                if max(img.size) > max_dim:
                    ratio = max_dim / max(img.size)
                    new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                    img = img.resize(new_size, Image.LANCZOS)
                
                logger.info(f"Page {page_num + 1}: Image size {img.size[0]}x{img.size[1]}")
                
                # Extract with retry logic
                result = self.extract_page_with_retry(img, page_num + 1, page_text if has_selectable_text else None)
                
                if result and result.get('bill_items'):
                    all_pages.append(result)
                    items_count = len(result['bill_items'])
                    total_items += items_count
                    logger.info(f"Page {page_num + 1}: Extracted {items_count} items")
                else:
                    logger.info(f"Page {page_num + 1}: No items found (might be blank/header page)")
            
            pdf.close()
            
            logger.info(f"PDF extraction complete: {total_items} items across {len(all_pages)} pages")
            
            return {
                "pagewise_line_items": all_pages,
                "total_item_count": total_items
            }
            
        except ImportError:
            logger.error("PyMuPDF (fitz) not installed")
            raise Exception("PDF processing library not available")
        except Exception as e:
            logger.error(f"PDF processing error: {str(e)}")
            raise
    
    def extract_from_image(self, image_content: bytes) -> Dict:
        """Extract from single image"""
        try:
            img = Image.open(BytesIO(image_content))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Higher resolution for images too
            max_dim = 1500
            if max(img.size) > max_dim:
                ratio = max_dim / max(img.size)
                new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                img = img.resize(new_size, Image.LANCZOS)
            
            logger.info(f"Image size: {img.size[0]}x{img.size[1]}")
            
            result = self.extract_page_with_retry(img, 1, None)
            
            return {
                "pagewise_line_items": [result] if result and result.get('bill_items') else [],
                "total_item_count": len(result.get('bill_items', [])) if result else 0
            }
        except Exception as e:
            logger.error(f"Image processing error: {str(e)}")
            raise
    
    def extract_page_with_retry(self, image: Image.Image, page_num: int, 
                                 extracted_text: Optional[str] = None,
                                 max_retries: int = 3) -> Dict:
        """Extract from page with retry logic and exponential backoff"""
        
        empty_result = {"page_no": str(page_num), "page_type": "Bill Detail", "bill_items": []}
        
        for attempt in range(max_retries):
            try:
                result = self.extract_page(image, page_num, extracted_text)
                
                if result and (result.get('bill_items') or attempt == max_retries - 1):
                    return result
                
                # If no items found, might be a blank page or extraction issue
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2  # 2s, 4s, 6s
                    logger.warning(f"Page {page_num}: Empty result, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 3  # 3s, 6s, 9s for errors
                    logger.warning(f"Page {page_num}: Error '{str(e)}', retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Page {page_num}: All retries failed - {str(e)}")
                    return empty_result
        
        return empty_result
    
    def extract_page(self, image: Image.Image, page_num: int, 
                     extracted_text: Optional[str] = None) -> Dict:
        """Extract line items from a single page image"""
        
        empty_result = {"page_no": str(page_num), "page_type": "Bill Detail", "bill_items": []}
        
        try:
            # Build content - image + optional text context
            content = [self.prompt, image]
            
            # If we have extracted text, include it as additional context
            if extracted_text and len(extracted_text) > 50:
                text_hint = f"\n\nAdditional text context from this page:\n{extracted_text[:2000]}"
                content = [self.prompt + text_hint, image]
            
            response = self.model.generate_content(
                content,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=4096,  # Increased for pages with many items
                ),
                safety_settings=self.safety_settings
            )
            
            # Track tokens
            if hasattr(response, 'usage_metadata'):
                self.total_input_tokens += getattr(response.usage_metadata, 'prompt_token_count', 0)
                self.total_output_tokens += getattr(response.usage_metadata, 'candidates_token_count', 0)
            else:
                # Estimate if metadata not available
                self.total_input_tokens += 500
                self.total_output_tokens += 200
            
            # Get text with detailed error logging
            text, error_reason = self.safe_get_text_with_reason(response)
            
            if not text:
                logger.warning(f"Page {page_num}: No response - {error_reason}")
                return empty_result
            
            logger.debug(f"Page {page_num}: Got {len(text)} chars response")
            
            # Parse JSON with robust handling
            result = self.parse_response(text, page_num)
            return result if result else empty_result
            
        except Exception as e:
            logger.error(f"Page {page_num} extraction error: {str(e)}")
            raise
    
    def safe_get_text_with_reason(self, response) -> Tuple[Optional[str], str]:
        """Safely extract text from Gemini response with detailed error reason"""
        try:
            if not response:
                return None, "Empty response object"
            
            if not hasattr(response, 'candidates') or not response.candidates:
                # Try to get error info
                if hasattr(response, 'prompt_feedback'):
                    feedback = response.prompt_feedback
                    if hasattr(feedback, 'block_reason'):
                        return None, f"Blocked: {feedback.block_reason}"
                return None, "No candidates in response"
            
            candidate = response.candidates[0]
            
            # Check finish reason
            if hasattr(candidate, 'finish_reason'):
                reason = candidate.finish_reason
                reason_value = getattr(reason, 'value', reason) if hasattr(reason, 'value') else reason
                
                # Map reason codes
                reason_map = {
                    1: "STOP (normal)",
                    2: "MAX_TOKENS (truncated)",
                    3: "SAFETY (blocked)",
                    4: "RECITATION (blocked)",
                    5: "OTHER"
                }
                
                if reason_value in [3, 4]:
                    reason_name = reason_map.get(reason_value, f"Code {reason_value}")
                    return None, f"Finish reason: {reason_name}"
            
            # Check for content
            if not hasattr(candidate, 'content'):
                return None, "Candidate has no content attribute"
            
            if not candidate.content:
                return None, "Candidate content is empty"
            
            if not hasattr(candidate.content, 'parts') or not candidate.content.parts:
                return None, "Content has no parts"
            
            text = candidate.content.parts[0].text
            if not text or not text.strip():
                return None, "Extracted text is empty"
            
            return text.strip(), "OK"
            
        except Exception as e:
            return None, f"Exception: {str(e)}"
    
    def parse_response(self, text: str, page_num: int) -> Optional[Dict]:
        """Parse JSON with robust truncation handling"""
        try:
            text = text.strip()
            
            # Remove markdown code blocks
            text = re.sub(r'^```json?\s*', '', text, flags=re.IGNORECASE)
            text = re.sub(r'\s*```$', '', text)
            text = text.strip()
            
            # Try direct parse first
            try:
                match = re.search(r'\{[\s\S]*\}', text)
                if match:
                    json_text = match.group()
                    json_text = self.fix_json(json_text)
                    data = json.loads(json_text)
                    return self.build_result(data, page_num)
            except json.JSONDecodeError as e:
                logger.debug(f"Page {page_num}: Direct JSON parse failed: {e}")
            
            # Fallback: extract items with regex
            return self.extract_items_regex(text, page_num)
            
        except Exception as e:
            logger.warning(f"Page {page_num}: Parse error: {e}")
            return self.extract_items_regex(text, page_num)
    
    def fix_json(self, text: str) -> str:
        """Fix common JSON issues including truncation"""
        # Remove newlines inside strings
        result = []
        in_string = False
        escape_next = False
        
        for i, char in enumerate(text):
            if escape_next:
                result.append(char)
                escape_next = False
                continue
            
            if char == '\\':
                result.append(char)
                escape_next = True
                continue
            
            if char == '"':
                in_string = not in_string
            
            if char in '\n\r' and in_string:
                result.append(' ')
            else:
                result.append(char)
        
        text = ''.join(result)
        
        # Remove trailing commas before } or ]
        text = re.sub(r',\s*}', '}', text)
        text = re.sub(r',\s*]', ']', text)
        
        # Fix truncated JSON
        open_braces = text.count('{') - text.count('}')
        open_brackets = text.count('[') - text.count(']')
        
        if open_braces > 0 or open_brackets > 0:
            # Remove incomplete last item if truncated mid-item
            last_comma = text.rfind(',')
            last_open_brace = text.rfind('{"item')
            
            if last_open_brace > last_comma and last_open_brace > 0:
                text = text[:last_open_brace].rstrip(',').rstrip()
            
            # Recount and close
            open_braces = text.count('{') - text.count('}')
            open_brackets = text.count('[') - text.count(']')
            
            text = text + ']' * max(0, open_brackets) + '}' * max(0, open_braces)
        
        return text
    
    def extract_items_regex(self, text: str, page_num: int) -> Optional[Dict]:
        """Extract items using regex - handles truncated responses"""
        try:
            items = []
            
            # Pattern 1: Standard order
            pattern1 = r'"item_name"\s*:\s*"([^"]+)"[^}]*?"item_amount"\s*:\s*([\d.]+)'
            matches1 = re.findall(pattern1, text, re.DOTALL)
            
            for name, amount in matches1:
                if not name.strip() or float(amount) <= 0:
                    continue
                    
                item = {"item_name": name.strip(), "item_amount": float(amount)}
                
                # Find rate and quantity in nearby context
                item_section = text[text.find(f'"{name}"'):text.find(f'"{name}"') + 300]
                
                rate_match = re.search(r'"item_rate"\s*:\s*([\d.]+)', item_section)
                if rate_match:
                    item["item_rate"] = float(rate_match.group(1))
                
                qty_match = re.search(r'"item_quantity"\s*:\s*([\d.]+)', item_section)
                if qty_match:
                    item["item_quantity"] = float(qty_match.group(1))
                
                items.append(item)
            
            # Pattern 2: Reverse order (amount before name)
            if not items:
                pattern2 = r'"item_amount"\s*:\s*([\d.]+)[^}]*?"item_name"\s*:\s*"([^"]+)"'
                matches2 = re.findall(pattern2, text, re.DOTALL)
                
                for amount, name in matches2:
                    if float(amount) > 0 and name.strip():
                        items.append({
                            "item_name": name.strip(),
                            "item_amount": float(amount)
                        })
            
            if items:
                # Deduplicate
                seen = set()
                unique_items = []
                for item in items:
                    key = (item["item_name"].lower(), item["item_amount"])
                    if key not in seen:
                        seen.add(key)
                        unique_items.append(item)
                
                # Detect page type
                page_type = "Bill Detail"
                type_match = re.search(r'"page_type"\s*:\s*"([^"]+)"', text)
                if type_match:
                    detected = type_match.group(1)
                    if detected in ["Pharmacy", "Final Bill", "Bill Detail"]:
                        page_type = detected
                
                # Also infer from content
                text_lower = text.lower()
                if any(kw in text_lower for kw in ['pharmacy', 'medicine', 'tablet', 'capsule', 'syrup']):
                    page_type = "Pharmacy"
                elif any(kw in text_lower for kw in ['final bill', 'grand total', 'total payable']):
                    page_type = "Final Bill"
                
                logger.info(f"Page {page_num}: Regex extracted {len(unique_items)} items")
                
                return {
                    "page_no": str(page_num),
                    "page_type": page_type,
                    "bill_items": unique_items
                }
            
            return None
            
        except Exception as e:
            logger.warning(f"Page {page_num}: Regex extraction failed: {e}")
            return None
    
    def build_result(self, data: dict, page_num: int) -> Dict:
        """Build result from parsed JSON"""
        result = {
            "page_no": str(page_num),
            "page_type": data.get("page_type", "Bill Detail"),
            "bill_items": []
        }
        
        # Validate page type
        if result["page_type"] not in ["Bill Detail", "Final Bill", "Pharmacy"]:
            result["page_type"] = "Bill Detail"
        
        for item in data.get("bill_items", []):
            name = str(item.get("item_name", "")).strip()
            amount = self.parse_num(item.get("item_amount"))
            
            if not name or amount <= 0:
                continue
            
            bill_item = {
                "item_name": name,
                "item_amount": amount
            }
            
            rate = self.parse_num(item.get("item_rate"))
            if rate > 0:
                bill_item["item_rate"] = rate
            
            qty = self.parse_qty(item.get("item_quantity"))
            if qty > 0:
                bill_item["item_quantity"] = qty
            
            result["bill_items"].append(bill_item)
        
        return result
    
    def parse_num(self, val) -> float:
        """Parse numeric value robustly"""
        if val is None:
            return 0.0
        if isinstance(val, (int, float)):
            return float(val)
        try:
            s = str(val).replace(',', '').replace('₹', '').replace('Rs', '').replace('Rs.', '').strip()
            m = re.search(r'[\d.]+', s)
            return float(m.group()) if m else 0.0
        except:
            return 0.0
    
    def parse_qty(self, val) -> float:
        """Parse quantity value robustly"""
        if val is None:
            return 0.0
        if isinstance(val, (int, float)):
            return float(val)
        try:
            s = re.sub(r'\s*(No|Nos|Units?|Pcs?)\.?\s*', '', str(val), flags=re.IGNORECASE)
            m = re.search(r'[\d.]+', s)
            return float(m.group()) if m else 0.0
        except:
            return 0.0