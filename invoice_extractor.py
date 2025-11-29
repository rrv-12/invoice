"""
Invoice Extractor - Final Optimized Version
- Handles truncated JSON responses
- Fast processing under 150 seconds
- Robust error handling
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
from typing import Dict, Optional, List

logger = logging.getLogger(__name__)

class InvoiceExtractor:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        
        # Disable all safety filters
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        
        # Ultra-compact prompt for faster response
        self.prompt = """Extract bill items as JSON. Format:
{"page_type":"Bill Detail","bill_items":[{"item_name":"X","item_amount":100.0,"item_rate":100.0,"item_quantity":1}]}

Rules:
- Extract ALL items with prices
- item_amount = Total/Net amount (last column)
- item_rate = Rate per unit
- item_quantity = just number (ignore "No")
- page_type: Pharmacy/Bill Detail/Final Bill
- Skip totals, headers, taxes
Return ONLY JSON."""

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
        self.reset_token_count()
        
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            content = response.content
            content_type = response.headers.get('Content-Type', '').lower()
            
            is_pdf = (
                'application/pdf' in content_type or 
                url.lower().split('?')[0].endswith('.pdf') or
                content[:4] == b'%PDF'
            )
            
            if is_pdf:
                return self.extract_from_pdf(content)
            else:
                return self.extract_from_image(content)
                
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            raise
    
    def extract_from_pdf(self, pdf_content: bytes) -> Dict:
        try:
            import fitz
            
            pdf = fitz.open(stream=pdf_content, filetype="pdf")
            num_pages = len(pdf)
            logger.info(f"Processing PDF: {num_pages} pages")
            
            all_pages = []
            total_items = 0
            
            # Fast delay: aim for ~10s per page max
            delay = 1.5 if num_pages <= 10 else 1.0
            
            for page_num in range(num_pages):
                if page_num > 0:
                    time.sleep(delay)
                
                logger.info(f"Page {page_num + 1}/{num_pages}")
                
                # Convert to image - lower quality for speed
                page = pdf[page_num]
                pix = page.get_pixmap(matrix=fitz.Matrix(1.0, 1.0))  # 1x zoom
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                # Resize to max 800px for faster processing
                if max(img.size) > 800:
                    ratio = 800 / max(img.size)
                    img = img.resize((int(img.size[0]*ratio), int(img.size[1]*ratio)), Image.LANCZOS)
                
                # Extract
                result = self.extract_page(img, page_num + 1)
                if result and result.get('bill_items'):
                    all_pages.append(result)
                    items_count = len(result['bill_items'])
                    total_items += items_count
                    logger.info(f"Page {page_num + 1}: Extracted {items_count} items")
            
            pdf.close()
            
            return {
                "pagewise_line_items": all_pages,
                "total_item_count": total_items
            }
            
        except Exception as e:
            logger.error(f"PDF error: {str(e)}")
            raise
    
    def extract_from_image(self, image_content: bytes) -> Dict:
        try:
            img = Image.open(BytesIO(image_content))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            if max(img.size) > 800:
                ratio = 800 / max(img.size)
                img = img.resize((int(img.size[0]*ratio), int(img.size[1]*ratio)), Image.LANCZOS)
            
            result = self.extract_page(img, 1)
            
            return {
                "pagewise_line_items": [result] if result and result.get('bill_items') else [],
                "total_item_count": len(result.get('bill_items', [])) if result else 0
            }
        except Exception as e:
            logger.error(f"Image error: {str(e)}")
            raise
    
    def extract_page(self, image: Image.Image, page_num: int) -> Dict:
        empty_result = {"page_no": str(page_num), "page_type": "Bill Detail", "bill_items": []}
        
        try:
            response = self.model.generate_content(
                [self.prompt, image],
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=2048,  # Reduced to prevent truncation
                ),
                safety_settings=self.safety_settings
            )
            
            # Track tokens
            if hasattr(response, 'usage_metadata'):
                self.total_input_tokens += getattr(response.usage_metadata, 'prompt_token_count', 0)
                self.total_output_tokens += getattr(response.usage_metadata, 'candidates_token_count', 0)
            else:
                self.total_input_tokens += 300
                self.total_output_tokens += 150
            
            # Get text safely
            text = self.safe_get_text(response)
            if not text:
                logger.warning(f"Page {page_num}: No response text")
                return empty_result
            
            # Parse JSON with robust handling
            result = self.parse_response(text, page_num)
            return result if result else empty_result
            
        except Exception as e:
            logger.error(f"Page {page_num}: {str(e)}")
            return empty_result
    
    def safe_get_text(self, response) -> Optional[str]:
        """Safely extract text from Gemini response"""
        try:
            if not response.candidates:
                return None
            
            candidate = response.candidates[0]
            
            # Check if blocked
            if hasattr(candidate, 'finish_reason'):
                reason = candidate.finish_reason
                # 1=STOP (good), 2=MAX_TOKENS (ok, truncated), 3=SAFETY (bad), 4=RECITATION (bad)
                if hasattr(reason, 'value') and reason.value in [3, 4]:
                    return None
            
            if not hasattr(candidate, 'content') or not candidate.content.parts:
                return None
            
            return candidate.content.parts[0].text
        except:
            return None
    
    def parse_response(self, text: str, page_num: int) -> Optional[Dict]:
        """Parse JSON with robust truncation handling"""
        try:
            text = text.strip()
            
            # Remove markdown
            text = re.sub(r'^```json?\s*', '', text)
            text = re.sub(r'\s*```$', '', text)
            text = text.strip()
            
            # Try direct parse first
            try:
                match = re.search(r'\{[\s\S]*\}', text)
                if match:
                    json_text = match.group()
                    # Fix common issues
                    json_text = self.fix_json(json_text)
                    data = json.loads(json_text)
                    return self.build_result(data, page_num)
            except json.JSONDecodeError:
                pass
            
            # If direct parse fails, extract items with regex
            return self.extract_items_regex(text, page_num)
            
        except Exception as e:
            logger.warning(f"Parse error: {e}")
            return self.extract_items_regex(text, page_num)
    
    def fix_json(self, text: str) -> str:
        """Fix common JSON issues including truncation"""
        # Remove newlines inside strings
        result = []
        in_string = False
        for i, char in enumerate(text):
            if char == '"' and (i == 0 or text[i-1] != '\\'):
                in_string = not in_string
            if char in '\n\r' and in_string:
                result.append(' ')
            else:
                result.append(char)
        text = ''.join(result)
        
        # Remove trailing commas
        text = re.sub(r',\s*}', '}', text)
        text = re.sub(r',\s*]', ']', text)
        
        # Try to fix truncated JSON by closing brackets
        open_braces = text.count('{') - text.count('}')
        open_brackets = text.count('[') - text.count(']')
        
        # If truncated mid-item, try to close it
        if open_braces > 0 or open_brackets > 0:
            # Remove incomplete last item
            last_comma = text.rfind(',')
            last_open_brace = text.rfind('{"item')
            
            if last_open_brace > last_comma:
                # Truncated in middle of item, remove it
                text = text[:last_open_brace].rstrip(',').rstrip()
            
            # Close brackets
            text = text + ']' * open_brackets + '}' * open_braces
        
        return text
    
    def extract_items_regex(self, text: str, page_num: int) -> Optional[Dict]:
        """Extract items using regex - handles truncated responses"""
        try:
            items = []
            
            # Pattern 1: Complete items
            pattern = r'"item_name"\s*:\s*"([^"]+)"[^}]*?"item_amount"\s*:\s*([\d.]+)'
            matches = re.findall(pattern, text, re.DOTALL)
            
            for name, amount in matches:
                item = {"item_name": name.strip(), "item_amount": float(amount)}
                
                # Try to find rate and quantity for this item
                item_section = text[text.find(name):text.find(name)+200]
                
                rate_match = re.search(r'"item_rate"\s*:\s*([\d.]+)', item_section)
                if rate_match:
                    item["item_rate"] = float(rate_match.group(1))
                
                qty_match = re.search(r'"item_quantity"\s*:\s*([\d.]+)', item_section)
                if qty_match:
                    item["item_quantity"] = float(qty_match.group(1))
                
                if item["item_amount"] > 0:
                    items.append(item)
            
            # Pattern 2: Reverse order (amount before name)
            if not items:
                pattern2 = r'"item_amount"\s*:\s*([\d.]+)[^}]*?"item_name"\s*:\s*"([^"]+)"'
                matches2 = re.findall(pattern2, text, re.DOTALL)
                for amount, name in matches2:
                    if float(amount) > 0:
                        items.append({"item_name": name.strip(), "item_amount": float(amount)})
            
            if items:
                # Deduplicate
                seen = set()
                unique_items = []
                for item in items:
                    key = (item["item_name"], item["item_amount"])
                    if key not in seen:
                        seen.add(key)
                        unique_items.append(item)
                
                page_type = "Bill Detail"
                type_match = re.search(r'"page_type"\s*:\s*"([^"]+)"', text)
                if type_match and type_match.group(1) in ["Pharmacy", "Final Bill", "Bill Detail"]:
                    page_type = type_match.group(1)
                
                logger.info(f"Page {page_num}: Regex extracted {len(unique_items)} items")
                return {
                    "page_no": str(page_num),
                    "page_type": page_type,
                    "bill_items": unique_items
                }
            
            return None
        except Exception as e:
            logger.warning(f"Regex extraction failed: {e}")
            return None
    
    def build_result(self, data: dict, page_num: int) -> Dict:
        """Build result from parsed JSON"""
        result = {
            "page_no": str(page_num),
            "page_type": data.get("page_type", "Bill Detail"),
            "bill_items": []
        }
        
        if result["page_type"] not in ["Bill Detail", "Final Bill", "Pharmacy"]:
            result["page_type"] = "Bill Detail"
        
        for item in data.get("bill_items", []):
            bill_item = {
                "item_name": str(item.get("item_name", "")).strip(),
                "item_amount": self.parse_num(item.get("item_amount"))
            }
            
            rate = self.parse_num(item.get("item_rate"))
            if rate > 0:
                bill_item["item_rate"] = rate
            
            qty = self.parse_qty(item.get("item_quantity"))
            if qty > 0:
                bill_item["item_quantity"] = qty
            
            if bill_item["item_name"] and bill_item["item_amount"] > 0:
                result["bill_items"].append(bill_item)
        
        return result
    
    def parse_num(self, val) -> float:
        if val is None:
            return 0.0
        if isinstance(val, (int, float)):
            return float(val)
        try:
            s = str(val).replace(',', '').replace('₹', '').strip()
            m = re.search(r'[\d.]+', s)
            return float(m.group()) if m else 0.0
        except:
            return 0.0
    
    def parse_qty(self, val) -> float:
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