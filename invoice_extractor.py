"""
Invoice Extractor using Google Gemini Vision API (FREE)
- Gemini 1.5 Flash: FREE tier with 15 RPM, 1M tokens/day
- Supports multi-page PDFs
- Returns structured JSON with token usage
"""

import google.generativeai as genai
import requests
from io import BytesIO
from PIL import Image
import base64
import json
import re
import logging
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

class InvoiceExtractor:
    def __init__(self, api_key: str):
        """Initialize Gemini with API key"""
        genai.configure(api_key=api_key)
        
        # Use Gemini 1.5 Flash (FREE and fast)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Token tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        
        # Extraction prompt - optimized for concise output
        self.extraction_prompt = """Extract ALL line items from this invoice. Return compact JSON only.

RULES:
1. Extract EVERY billable line item with actual charges
2. SKIP section headers that are just titles (like "Consultation", "Drugs" without amounts on same line)
3. SKIP subtotals, grand totals, discounts, deposits, refunds
4. item_name: Short name from bill (exclude batch numbers, lot numbers, DOE dates)
5. item_amount: Net amount (rightmost amount column, after any discounts)
6. item_rate: Unit price/rate (if shown)
7. item_quantity: Qty (if shown)

PAGE TYPE:
- "Final Bill" if this is a SUMMARY page with category totals (e.g., "Consultation: 2650", "Drugs: 1702")
- "Pharmacy" if items are medicines/drugs with batch numbers
- "Bill Detail" for itemized services/procedures

Return compact JSON:
{"page_type":"Bill Detail","bill_items":[{"item_name":"Service Name","item_amount":100,"item_rate":100,"item_quantity":1}]}

CRITICAL: Return ONLY valid JSON. No markdown. Keep names SHORT."""

    def reset_token_count(self):
        """Reset token counters for new request"""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
    
    def get_token_usage(self) -> Dict[str, int]:
        """Get cumulative token usage"""
        return {
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens
        }
    
    def extract_from_url(self, url: str) -> Dict:
        """Extract invoice data from image or PDF URL"""
        self.reset_token_count()
        
        try:
            # Download content
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            content = response.content
            content_type = response.headers.get('Content-Type', '').lower()
            
            # Check if PDF
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
            logger.error(f"Error extracting from URL: {str(e)}")
            raise
    
    def extract_from_pdf(self, pdf_content: bytes) -> Dict:
        """Extract from multi-page PDF using batched processing for efficiency"""
        try:
            import fitz  # PyMuPDF
            import time
            
            pdf_document = fitz.open(stream=pdf_content, filetype="pdf")
            num_pages = len(pdf_document)
            
            logger.info(f"Processing PDF with {num_pages} pages")
            
            # Convert all pages to images first
            page_images = []
            for page_num in range(num_pages):
                page = pdf_document[page_num]
                
                # Use 1.5x zoom (good balance of quality and size)
                mat = fitz.Matrix(1.5, 1.5)
                pix = page.get_pixmap(matrix=mat)
                
                # Convert to PIL Image
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                # Resize if too large (max 1500px for batching)
                max_dimension = 1500
                if max(img.size) > max_dimension:
                    ratio = max_dimension / max(img.size)
                    new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                    img = img.resize(new_size, Image.LANCZOS)
                
                # Skip likely blank pages (very low content)
                if not self.is_blank_page(img):
                    page_images.append((page_num + 1, img))
                else:
                    logger.info(f"Skipping blank page {page_num + 1}")
            
            pdf_document.close()
            
            if not page_images:
                logger.warning("No valid pages found in PDF")
                return {"pagewise_line_items": [], "total_item_count": 0}
            
            logger.info(f"Processing {len(page_images)} non-blank pages")
            
            # Process in batches of 3-4 pages per API call
            batch_size = 3
            all_page_items = []
            
            for i in range(0, len(page_images), batch_size):
                batch = page_images[i:i + batch_size]
                batch_page_nums = [p[0] for p in batch]
                batch_images = [p[1] for p in batch]
                
                logger.info(f"Processing batch: pages {batch_page_nums}")
                
                # Rate limiting between batches
                if i > 0:
                    time.sleep(4.5)
                
                # Extract from batch
                batch_results = self.extract_batch(batch_images, batch_page_nums)
                
                if batch_results:
                    all_page_items.extend(batch_results)
            
            # Deduplicate across pages
            all_page_items = self.deduplicate_across_pages(all_page_items)
            
            # Calculate total
            total_items = sum(len(page.get('bill_items', [])) for page in all_page_items)
            
            return {
                "pagewise_line_items": all_page_items,
                "total_item_count": total_items
            }
            
        except ImportError:
            logger.error("PyMuPDF not installed. Install with: pip install PyMuPDF")
            raise
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise
    
    def is_blank_page(self, img: Image.Image) -> bool:
        """Check if page is mostly blank (skip header-only pages)"""
        try:
            import numpy as np
            
            # Convert to grayscale and check variance
            gray = img.convert('L')
            arr = np.array(gray)
            
            # If very low variance, page is likely blank
            variance = np.var(arr)
            
            # Also check if mostly white (>95% pixels are light)
            white_ratio = np.sum(arr > 240) / arr.size
            
            is_blank = variance < 500 or white_ratio > 0.95
            
            return is_blank
        except:
            return False  # If check fails, process the page anyway
    
    def extract_batch(self, images: List[Image.Image], page_nums: List[int]) -> List[Dict]:
        """Extract from multiple pages in a single API call"""
        try:
            # Build prompt for multiple pages
            if len(images) == 1:
                batch_prompt = self.extraction_prompt
            else:
                batch_prompt = f"""Extract ALL line items from these {len(images)} invoice pages. Return JSON for EACH page.

RULES:
1. Extract EVERY billable line item from EACH page
2. SKIP section headers, subtotals, grand totals, discounts
3. item_name: Short name (exclude batch/lot numbers)
4. item_amount: Net amount (rightmost column)
5. item_rate: Unit price (if shown)
6. item_quantity: Qty (if shown)

PAGE TYPES:
- "Final Bill" for summary pages with category totals
- "Pharmacy" for medicine/drug pages
- "Bill Detail" for itemized services

Return as JSON array with one object per page:
[{{"page_no":"{page_nums[0]}","page_type":"Bill Detail","bill_items":[{{"item_name":"X","item_amount":100}}]}},{{"page_no":"{page_nums[1]}","page_type":"Pharmacy","bill_items":[...]}}]

CRITICAL: Return ONLY valid JSON array. No markdown."""
            
            # Build content list with prompt and all images
            content = [batch_prompt] + list(images)
            
            # Call Gemini with all images
            response = self.model.generate_content(
                content,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=8192,
                )
            )
            
            # Track tokens
            if hasattr(response, 'usage_metadata'):
                self.total_input_tokens += getattr(response.usage_metadata, 'prompt_token_count', 0)
                self.total_output_tokens += getattr(response.usage_metadata, 'candidates_token_count', 0)
            else:
                self.total_input_tokens += 1500 * len(images)
                self.total_output_tokens += len(response.text) // 4
            
            # Parse response
            return self.parse_batch_response(response.text, page_nums)
            
        except Exception as e:
            logger.error(f"Batch extraction failed: {str(e)}")
            # Fallback: process pages individually
            logger.info("Falling back to individual page processing")
            results = []
            for img, page_num in zip(images, page_nums):
                result = self.extract_single_page(img, page_num)
                if result:
                    results.append(result)
            return results
    
    def parse_batch_response(self, response_text: str, page_nums: List[int]) -> List[Dict]:
        """Parse response containing multiple pages"""
        try:
            text = response_text.strip()
            
            # Remove markdown
            if text.startswith('```json'):
                text = text[7:]
            elif text.startswith('```'):
                text = text[3:]
            if text.endswith('```'):
                text = text[:-3]
            text = text.strip()
            
            # Fix JSON issues
            text = self.fix_json_string(text)
            
            # Try to parse as array
            if text.startswith('['):
                data = json.loads(text)
                if isinstance(data, list):
                    results = []
                    for i, page_data in enumerate(data):
                        page_num = page_data.get('page_no', str(page_nums[i] if i < len(page_nums) else i + 1))
                        formatted = self.format_result(page_data)
                        if formatted:
                            formatted['page_no'] = str(page_num)
                            results.append(formatted)
                    return results
            
            # Try as single object (for single page batch)
            if text.startswith('{'):
                data = json.loads(text)
                formatted = self.format_result(data)
                if formatted:
                    formatted['page_no'] = str(page_nums[0])
                    return [formatted]
            
            return []
            
        except json.JSONDecodeError as e:
            logger.warning(f"Batch JSON parse error: {e}")
            # Try to salvage
            return self.salvage_batch_response(response_text, page_nums)
        except Exception as e:
            logger.error(f"Error parsing batch response: {e}")
            return []
    
    def salvage_batch_response(self, response_text: str, page_nums: List[int]) -> List[Dict]:
        """Salvage items from partial batch response"""
        try:
            # Find all page objects
            page_pattern = r'\{\s*"page_no"\s*:\s*"?(\d+)"?\s*,\s*"page_type"\s*:\s*"([^"]+)"'
            page_matches = list(re.finditer(page_pattern, response_text))
            
            if not page_matches:
                # No page structure, try to extract as single page
                result = self.salvage_partial_json(response_text)
                if result:
                    result['page_no'] = str(page_nums[0])
                    return [result]
                return []
            
            results = []
            for i, match in enumerate(page_matches):
                # Find the content for this page (until next page or end)
                start = match.start()
                end = page_matches[i + 1].start() if i + 1 < len(page_matches) else len(response_text)
                page_content = response_text[start:end]
                
                # Extract items from this page section
                result = self.salvage_partial_json(page_content)
                if result:
                    result['page_no'] = match.group(1)
                    result['page_type'] = match.group(2)
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Salvage batch failed: {e}")
            return []
    
    def deduplicate_across_pages(self, all_page_items: List[Dict]) -> List[Dict]:
        """Remove duplicate items that appear on both summary and detail pages"""
        if len(all_page_items) <= 1:
            return all_page_items
        
        # Find summary pages (pages with generic names like "Consultation", "Drugs", "Room Rent")
        summary_keywords = ['consultation', 'drugs', 'investigations', 'medical consumable', 
                          'other charges', 'procedures', 'room rent', 'surgery', 'pharmacy']
        
        detail_pages = []
        summary_pages = []
        
        for page in all_page_items:
            items = page.get('bill_items', [])
            if not items:
                continue
            
            # Check if this looks like a summary page
            generic_count = 0
            for item in items:
                name_lower = item.get('item_name', '').lower().strip()
                if name_lower in summary_keywords or len(name_lower) < 15:
                    generic_count += 1
            
            # If more than 50% of items are generic, it's likely a summary page
            if generic_count / len(items) > 0.5:
                summary_pages.append(page)
                page['page_type'] = 'Final Bill'  # Mark as summary
            else:
                detail_pages.append(page)
        
        # If we have both summary and detail pages, prefer detail pages
        if detail_pages and summary_pages:
            logger.info(f"Found {len(summary_pages)} summary pages and {len(detail_pages)} detail pages. Using detail pages.")
            return detail_pages
        
        # Otherwise return all pages
        return all_page_items
    
    def extract_from_image(self, image_content: bytes) -> Dict:
        """Extract from single image"""
        try:
            img = Image.open(BytesIO(image_content))
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            page_result = self.extract_single_page(img, 1)
            
            return {
                "pagewise_line_items": [page_result] if page_result else [],
                "total_item_count": len(page_result.get('bill_items', [])) if page_result else 0
            }
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise
    
    def extract_single_page(self, image: Image.Image, page_num: int) -> Optional[Dict]:
        """Extract line items from a single page using Gemini Vision"""
        try:
            # Call Gemini Vision API with higher token limit
            response = self.model.generate_content(
                [self.extraction_prompt, image],
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,  # Low temperature for consistent extraction
                    max_output_tokens=8192,  # Increased for large invoices
                )
            )
            
            # Track token usage
            if hasattr(response, 'usage_metadata'):
                self.total_input_tokens += getattr(response.usage_metadata, 'prompt_token_count', 0)
                self.total_output_tokens += getattr(response.usage_metadata, 'candidates_token_count', 0)
            else:
                # Estimate if not available
                self.total_input_tokens += 1500  # Approximate for image + prompt
                self.total_output_tokens += len(response.text) // 4
            
            # Parse response
            result = self.parse_llm_response(response.text)
            
            if result:
                result['page_no'] = str(page_num)
                return result
            
            # If parsing failed, try with a retry prompt
            logger.warning(f"First parse failed for page {page_num}, retrying...")
            return self.retry_extraction(image, page_num)
            
        except Exception as e:
            logger.error(f"Error in Gemini extraction: {str(e)}")
            return None
    
    def retry_extraction(self, image: Image.Image, page_num: int) -> Optional[Dict]:
        """Retry extraction with a simpler, more compact prompt"""
        try:
            retry_prompt = """List ALL billable items as compact JSON. Skip headers/totals.
Format: {"page_type":"Bill Detail","bill_items":[{"item_name":"X","item_amount":0.0,"item_rate":0.0,"item_quantity":1}]}
Keep item names SHORT. Return ONLY JSON."""
            
            response = self.model.generate_content(
                [retry_prompt, image],
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=8192,
                )
            )
            
            # Track tokens
            if hasattr(response, 'usage_metadata'):
                self.total_input_tokens += getattr(response.usage_metadata, 'prompt_token_count', 0)
                self.total_output_tokens += getattr(response.usage_metadata, 'candidates_token_count', 0)
            
            result = self.parse_llm_response(response.text)
            if result:
                result['page_no'] = str(page_num)
                return result
            
            return None
            
        except Exception as e:
            logger.error(f"Retry extraction failed: {str(e)}")
            return None
    
    def parse_llm_response(self, response_text: str) -> Optional[Dict]:
        """Parse JSON response from LLM with robust error handling"""
        try:
            # Clean up response - remove markdown code blocks if present
            text = response_text.strip()
            
            # Remove ```json and ``` markers
            if text.startswith('```json'):
                text = text[7:]
            elif text.startswith('```'):
                text = text[3:]
            
            if text.endswith('```'):
                text = text[:-3]
            
            text = text.strip()
            
            # Try to find JSON object in response
            json_match = re.search(r'\{[\s\S]*\}', text)
            if json_match:
                text = json_match.group()
            
            # Fix common JSON issues
            text = self.fix_json_string(text)
            
            # Parse JSON
            data = json.loads(text)
            
            return self.format_result(data)
            
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}")
            # Try to salvage partial JSON
            return self.salvage_partial_json(response_text)
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            return None
    
    def fix_json_string(self, text: str) -> str:
        """Fix common JSON formatting issues"""
        # First, try to remove any newlines inside string values
        # This handles cases like "item_name": "Doctor\nVisiting Fee..."
        
        # Replace literal newlines inside strings with spaces
        result = []
        in_string = False
        i = 0
        while i < len(text):
            char = text[i]
            
            if char == '"' and (i == 0 or text[i-1] != '\\'):
                in_string = not in_string
                result.append(char)
            elif char == '\n' and in_string:
                # Replace newline inside string with space
                result.append(' ')
            elif char == '\r' and in_string:
                # Skip carriage return inside string
                pass
            else:
                result.append(char)
            
            i += 1
        
        return ''.join(result)
    
    def salvage_partial_json(self, response_text: str) -> Optional[Dict]:
        """Try to extract items from partial/truncated JSON"""
        try:
            logger.info("Attempting to salvage partial JSON...")
            
            bill_items = []
            
            # Strategy 1: Find complete item objects (any field order)
            # Match objects that have at least item_name and item_amount
            complete_items = re.findall(
                r'\{[^{}]*"item_name"\s*:\s*"([^"]+)"[^{}]*"item_amount"\s*:\s*([\d.]+)[^{}]*\}',
                response_text,
                re.DOTALL
            )
            
            for match in complete_items:
                item = {
                    "item_name": match[0].strip(),
                    "item_amount": float(match[1])
                }
                if item["item_name"] and item["item_amount"] > 0:
                    bill_items.append(item)
            
            # Strategy 2: Also try reverse order (item_amount before item_name)
            reverse_items = re.findall(
                r'\{[^{}]*"item_amount"\s*:\s*([\d.]+)[^{}]*"item_name"\s*:\s*"([^"]+)"[^{}]*\}',
                response_text,
                re.DOTALL
            )
            
            for match in reverse_items:
                item = {
                    "item_name": match[1].strip(),
                    "item_amount": float(match[0])
                }
                # Avoid duplicates
                if item["item_name"] and item["item_amount"] > 0:
                    if not any(i["item_name"] == item["item_name"] and i["item_amount"] == item["item_amount"] for i in bill_items):
                        bill_items.append(item)
            
            # Strategy 3: Line-by-line extraction for truncated responses
            if not bill_items:
                logger.info("Trying line-by-line extraction...")
                
                # Find all item_name and item_amount pairs even if object is incomplete
                names = re.findall(r'"item_name"\s*:\s*"([^"]+)"', response_text)
                amounts = re.findall(r'"item_amount"\s*:\s*([\d.]+)', response_text)
                rates = re.findall(r'"item_rate"\s*:\s*([\d.]+)', response_text)
                quantities = re.findall(r'"item_quantity"\s*:\s*([\d.]+)', response_text)
                
                # Pair them up (assumes they appear in order)
                for i in range(min(len(names), len(amounts))):
                    item = {
                        "item_name": names[i].strip(),
                        "item_amount": float(amounts[i])
                    }
                    if i < len(rates):
                        item["item_rate"] = float(rates[i])
                    if i < len(quantities):
                        item["item_quantity"] = float(quantities[i])
                    
                    if item["item_name"] and item["item_amount"] > 0:
                        bill_items.append(item)
            
            # Now add rate and quantity to items from Strategy 1 & 2 if available
            if bill_items:
                # Extract rates and quantities separately
                for item in bill_items:
                    if "item_rate" not in item:
                        # Try to find rate for this item
                        pattern = re.escape(item["item_name"]) + r'[^}]*"item_rate"\s*:\s*([\d.]+)'
                        rate_match = re.search(pattern, response_text)
                        if rate_match:
                            item["item_rate"] = float(rate_match.group(1))
                    
                    if "item_quantity" not in item:
                        pattern = re.escape(item["item_name"]) + r'[^}]*"item_quantity"\s*:\s*([\d.]+)'
                        qty_match = re.search(pattern, response_text)
                        if qty_match:
                            item["item_quantity"] = float(qty_match.group(1))
            
            if bill_items:
                logger.info(f"Salvaged {len(bill_items)} items from partial JSON")
                
                # Try to get page_type
                page_type = "Bill Detail"
                type_match = re.search(r'"page_type"\s*:\s*"([^"]+)"', response_text)
                if type_match:
                    page_type = type_match.group(1)
                
                return {
                    "page_no": "1",
                    "page_type": page_type,
                    "bill_items": bill_items
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Salvage failed: {e}")
            return None
    
    def format_result(self, data: dict) -> Optional[Dict]:
        """Format and validate the parsed JSON data"""
        result = {
            "page_no": "1",
            "page_type": data.get("page_type", "Bill Detail"),
            "bill_items": []
        }
        
        # Validate page_type
        valid_types = ["Bill Detail", "Final Bill", "Pharmacy"]
        if result["page_type"] not in valid_types:
            result["page_type"] = "Bill Detail"
        
        # Process bill items
        for item in data.get("bill_items", []):
            bill_item = {
                "item_name": str(item.get("item_name", "")).strip(),
                "item_amount": float(item.get("item_amount", 0))
            }
            
            # Add optional fields if present and valid
            if "item_rate" in item and item["item_rate"] is not None:
                try:
                    bill_item["item_rate"] = float(item["item_rate"])
                except (ValueError, TypeError):
                    pass
            
            if "item_quantity" in item and item["item_quantity"] is not None:
                try:
                    bill_item["item_quantity"] = float(item["item_quantity"])
                except (ValueError, TypeError):
                    pass
            
            # Only add if valid (has name and positive amount)
            if bill_item["item_name"] and bill_item["item_amount"] > 0:
                result["bill_items"].append(bill_item)
        
        return result if result["bill_items"] else None