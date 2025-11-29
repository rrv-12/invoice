"""
Invoice Extractor using Google Gemini Vision API
- Simple, robust extraction
- Handles Gemini blocked responses
- Processes within 150 seconds
"""

import google.generativeai as genai
import requests
from io import BytesIO
from PIL import Image
import json
import re
import logging
import time
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class InvoiceExtractor:
    def __init__(self, api_key: str):
        """Initialize Gemini with API key"""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Token tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        
        # Safety settings to prevent blocked responses
        self.safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        
        # Simple extraction prompt
        self.prompt = """Extract ALL line items from this invoice. Return JSON only.

Rules:
- Extract every row with amounts as separate item
- Skip headers, totals, patient info
- item_quantity: just the number (ignore "No", "Nos")

Format:
{"page_type":"Bill Detail","bill_items":[{"item_name":"Name","item_amount":100.0,"item_rate":100.0,"item_quantity":1}]}

page_type: "Pharmacy" for medicines, "Final Bill" for summary, "Bill Detail" otherwise.
Return ONLY valid JSON."""

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
        """Extract invoice data from image or PDF URL"""
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
        """Extract from PDF - one page at a time"""
        try:
            import fitz
            
            pdf = fitz.open(stream=pdf_content, filetype="pdf")
            num_pages = len(pdf)
            logger.info(f"Processing PDF: {num_pages} pages")
            
            all_pages = []
            total_items = 0
            
            # Calculate delay to fit within 150 seconds
            # Reserve 30 sec for overhead, 120 sec for pages
            delay = min(3, 120 / max(num_pages, 1))
            
            for page_num in range(num_pages):
                if page_num > 0:
                    time.sleep(delay)
                
                logger.info(f"Page {page_num + 1}/{num_pages}")
                
                # Convert to image
                page = pdf[page_num]
                pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                # Resize if large
                if max(img.size) > 1200:
                    ratio = 1200 / max(img.size)
                    img = img.resize((int(img.size[0]*ratio), int(img.size[1]*ratio)), Image.LANCZOS)
                
                # Extract
                result = self.extract_page(img, page_num + 1)
                if result and result.get('bill_items'):
                    all_pages.append(result)
                    total_items += len(result['bill_items'])
            
            pdf.close()
            
            return {
                "pagewise_line_items": all_pages,
                "total_item_count": total_items
            }
            
        except Exception as e:
            logger.error(f"PDF error: {str(e)}")
            raise
    
    def extract_from_image(self, image_content: bytes) -> Dict:
        """Extract from single image"""
        try:
            img = Image.open(BytesIO(image_content))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            result = self.extract_page(img, 1)
            
            return {
                "pagewise_line_items": [result] if result else [],
                "total_item_count": len(result.get('bill_items', [])) if result else 0
            }
        except Exception as e:
            logger.error(f"Image error: {str(e)}")
            raise
    
    def extract_page(self, image: Image.Image, page_num: int) -> Dict:
        """Extract from single page with proper error handling"""
        empty_result = {"page_no": str(page_num), "page_type": "Bill Detail", "bill_items": []}
        
        try:
            response = self.model.generate_content(
                [self.prompt, image],
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=4096,
                ),
                safety_settings=self.safety_settings
            )
            
            # Track tokens
            if hasattr(response, 'usage_metadata'):
                self.total_input_tokens += getattr(response.usage_metadata, 'prompt_token_count', 0)
                self.total_output_tokens += getattr(response.usage_metadata, 'candidates_token_count', 0)
            else:
                self.total_input_tokens += 500
                self.total_output_tokens += 200
            
            # SAFE text extraction - handles blocked responses
            text = self.get_response_text(response)
            if not text:
                logger.warning(f"Page {page_num}: No response text")
                return empty_result
            
            # Parse
            result = self.parse_json(text)
            if result:
                result['page_no'] = str(page_num)
                return result
            
            return empty_result
            
        except Exception as e:
            logger.error(f"Page {page_num} error: {str(e)}")
            return empty_result
    
    def get_response_text(self, response) -> Optional[str]:
        """Safely extract text from Gemini response"""
        try:
            # Check if response has candidates
            if not response.candidates:
                return None
            
            candidate = response.candidates[0]
            
            # Check finish reason (2 = blocked)
            if hasattr(candidate, 'finish_reason') and candidate.finish_reason == 2:
                logger.warning("Response blocked by safety filter")
                return None
            
            # Check if has content parts
            if not hasattr(candidate, 'content') or not candidate.content.parts:
                return None
            
            # Get text from parts
            return candidate.content.parts[0].text
            
        except Exception as e:
            logger.warning(f"Could not get response text: {e}")
            return None
    
    def parse_json(self, text: str) -> Optional[Dict]:
        """Parse JSON from response"""
        try:
            text = text.strip()
            
            # Remove markdown
            if text.startswith('```'):
                text = re.sub(r'^```json?\n?', '', text)
                text = re.sub(r'\n?```$', '', text)
            
            # Find JSON object
            match = re.search(r'\{[\s\S]*\}', text)
            if match:
                text = match.group()
            
            data = json.loads(text)
            
            # Build result
            result = {
                "page_no": "1",
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
            
        except json.JSONDecodeError:
            return self.regex_extract(text)
        except:
            return None
    
    def regex_extract(self, text: str) -> Optional[Dict]:
        """Fallback: extract items with regex"""
        try:
            names = re.findall(r'"item_name"\s*:\s*"([^"]+)"', text)
            amounts = re.findall(r'"item_amount"\s*:\s*([\d.]+)', text)
            rates = re.findall(r'"item_rate"\s*:\s*([\d.]+)', text)
            qtys = re.findall(r'"item_quantity"\s*:\s*([\d.]+)', text)
            
            if not names or not amounts:
                return None
            
            items = []
            for i in range(min(len(names), len(amounts))):
                item = {"item_name": names[i], "item_amount": float(amounts[i])}
                if i < len(rates):
                    item["item_rate"] = float(rates[i])
                if i < len(qtys):
                    item["item_quantity"] = float(qtys[i])
                if item["item_amount"] > 0:
                    items.append(item)
            
            if items:
                page_type = "Bill Detail"
                m = re.search(r'"page_type"\s*:\s*"([^"]+)"', text)
                if m:
                    page_type = m.group(1)
                return {"page_no": "1", "page_type": page_type, "bill_items": items}
            
            return None
        except:
            return None
    
    def parse_num(self, val) -> float:
        """Parse number"""
        if val is None:
            return 0.0
        if isinstance(val, (int, float)):
            return float(val)
        try:
            return float(str(val).replace(',', '').replace('₹', '').strip())
        except:
            return 0.0
    
    def parse_qty(self, val) -> float:
        """Parse quantity (handles '1 No' format)"""
        if val is None:
            return 0.0
        if isinstance(val, (int, float)):
            return float(val)
        try:
            s = str(val)
            for suffix in [' No', ' Nos', 'No', 'Nos', ' Units']:
                s = s.replace(suffix, '')
            m = re.search(r'([\d.]+)', s)
            return float(m.group(1)) if m else 0.0
        except:
            return 0.0