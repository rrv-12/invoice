"""
Invoice Extractor using Google Gemini Vision API
- Proper safety settings to avoid blocks
- Retry mechanism for blocked responses
- Optimized for < 150 seconds
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
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class InvoiceExtractor:
    def __init__(self, api_key: str):
        """Initialize Gemini"""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        
        # CORRECT safety settings format
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        
        # Primary prompt
        self.prompt = """You are an invoice data extractor. Extract ALL line items from this medical bill image.

Return a JSON object with this exact structure:
{"page_type":"Bill Detail","bill_items":[{"item_name":"Item Name","item_amount":100.00,"item_rate":100.00,"item_quantity":1}]}

Rules:
- Extract EVERY row that has a price/amount
- item_name: exactly as shown in bill
- item_amount: the final/net amount (rightmost number column)
- item_rate: the rate/gross amount if shown
- item_quantity: just the number (if it says "1 No" or "2 Nos", extract 1 or 2)
- Skip totals, subtotals, headers
- page_type: "Pharmacy" for medicines, "Final Bill" for summary page, "Bill Detail" for detailed items

Return ONLY the JSON object, no explanation."""

        # Simpler backup prompt for retry
        self.simple_prompt = """Extract bill items as JSON:
{"page_type":"Bill Detail","bill_items":[{"item_name":"X","item_amount":0.0}]}
List all items with prices. Return only JSON."""

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
        """Main entry point"""
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
        """Extract from PDF with optimized timing"""
        try:
            import fitz
            
            pdf = fitz.open(stream=pdf_content, filetype="pdf")
            num_pages = len(pdf)
            logger.info(f"Processing PDF: {num_pages} pages")
            
            all_pages = []
            total_items = 0
            
            # Optimized delay: target 120 seconds total for API calls
            # Leave 30 sec buffer for download/processing
            delay = max(1.5, min(3, 100 / max(num_pages, 1)))
            
            for page_num in range(num_pages):
                if page_num > 0:
                    time.sleep(delay)
                
                logger.info(f"Page {page_num + 1}/{num_pages}")
                
                # Convert to image with moderate quality
                page = pdf[page_num]
                # Use 1.2x zoom instead of 1.5x for faster processing
                pix = page.get_pixmap(matrix=fitz.Matrix(1.2, 1.2))
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                # Resize to max 1000px (smaller = faster + less likely to be blocked)
                if max(img.size) > 1000:
                    ratio = 1000 / max(img.size)
                    img = img.resize((int(img.size[0]*ratio), int(img.size[1]*ratio)), Image.LANCZOS)
                
                # Extract with retry
                result = self.extract_page_with_retry(img, page_num + 1)
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
            
            # Resize if needed
            if max(img.size) > 1000:
                ratio = 1000 / max(img.size)
                img = img.resize((int(img.size[0]*ratio), int(img.size[1]*ratio)), Image.LANCZOS)
            
            result = self.extract_page_with_retry(img, 1)
            
            return {
                "pagewise_line_items": [result] if result else [],
                "total_item_count": len(result.get('bill_items', [])) if result else 0
            }
        except Exception as e:
            logger.error(f"Image error: {str(e)}")
            raise
    
    def extract_page_with_retry(self, image: Image.Image, page_num: int) -> Dict:
        """Extract with retry on failure"""
        # First attempt with main prompt
        result = self.call_gemini(image, page_num, self.prompt)
        
        if result and result.get('bill_items'):
            return result
        
        # Retry with simpler prompt
        logger.info(f"Page {page_num}: Retrying with simple prompt")
        time.sleep(1)
        result = self.call_gemini(image, page_num, self.simple_prompt)
        
        if result and result.get('bill_items'):
            return result
        
        # Return empty if both fail
        return {"page_no": str(page_num), "page_type": "Bill Detail", "bill_items": []}
    
    def call_gemini(self, image: Image.Image, page_num: int, prompt: str) -> Optional[Dict]:
        """Call Gemini API with proper error handling"""
        try:
            response = self.model.generate_content(
                [prompt, image],
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
                self.total_input_tokens += 400
                self.total_output_tokens += 200
            
            # Check for blocked response
            if not response.candidates:
                logger.warning(f"Page {page_num}: No candidates in response")
                return None
            
            candidate = response.candidates[0]
            
            # Check finish reason
            # 0=UNSPECIFIED, 1=STOP (good), 2=MAX_TOKENS, 3=SAFETY, 4=RECITATION, 5=OTHER
            finish_reason = getattr(candidate, 'finish_reason', None)
            if finish_reason and finish_reason.value in [3, 4]:  # SAFETY or RECITATION
                logger.warning(f"Page {page_num}: Response blocked (reason: {finish_reason})")
                return None
            
            # Get text
            if not hasattr(candidate, 'content') or not candidate.content.parts:
                logger.warning(f"Page {page_num}: No content in response")
                return None
            
            text = candidate.content.parts[0].text
            if not text:
                return None
            
            # Parse JSON
            result = self.parse_json(text)
            if result:
                result['page_no'] = str(page_num)
            return result
            
        except Exception as e:
            logger.error(f"Page {page_num}: API error - {str(e)}")
            return None
    
    def parse_json(self, text: str) -> Optional[Dict]:
        """Parse JSON from response"""
        try:
            text = text.strip()
            
            # Remove markdown code blocks
            text = re.sub(r'^```json?\s*\n?', '', text)
            text = re.sub(r'\n?```$', '', text)
            text = text.strip()
            
            # Find JSON object
            match = re.search(r'\{[\s\S]*\}', text)
            if match:
                text = match.group()
            
            # Fix common JSON issues
            text = text.replace('\n', ' ')
            text = re.sub(r',\s*}', '}', text)  # Remove trailing commas
            text = re.sub(r',\s*]', ']', text)
            
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
            
            return result if result["bill_items"] else None
            
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}")
            return self.regex_extract(text)
        except Exception as e:
            logger.warning(f"Parse error: {e}")
            return None
    
    def regex_extract(self, text: str) -> Optional[Dict]:
        """Fallback regex extraction"""
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
                if m and m.group(1) in ["Bill Detail", "Final Bill", "Pharmacy"]:
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
            s = str(val).replace(',', '').replace('₹', '').replace('Rs.', '').replace('Rs', '').strip()
            m = re.search(r'[\d.]+', s)
            return float(m.group()) if m else 0.0
        except:
            return 0.0
    
    def parse_qty(self, val) -> float:
        """Parse quantity"""
        if val is None:
            return 0.0
        if isinstance(val, (int, float)):
            return float(val)
        try:
            s = str(val)
            # Remove unit suffixes
            s = re.sub(r'\s*(No|Nos|Units?|Pcs?|Qty)\.?\s*', '', s, flags=re.IGNORECASE)
            m = re.search(r'[\d.]+', s)
            return float(m.group()) if m else 0.0
        except:
            return 0.0