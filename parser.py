"""
parser.py - Robust JSON parsing with multiple fallback strategies
"""

import json
import re
import logging
from typing import Optional, List, Dict, Any, Tuple

logger = logging.getLogger(__name__)


class JSONParser:
    """
    Robust JSON parser with multiple fallback strategies for handling
    malformed, truncated, or hallucinated LLM outputs.
    """
    
    def __init__(self):
        # Patterns for extracting JSON from various formats
        self.json_block_pattern = re.compile(
            r'```(?:json)?\s*([\s\S]*?)```',
            re.IGNORECASE
        )
        self.json_object_pattern = re.compile(
            r'\{[\s\S]*\}',
            re.DOTALL
        )
    
    def parse(self, text: str, page_num: int = 1) -> Optional[Dict]:
        """
        Parse JSON from LLM response with multiple fallback strategies.
        
        Strategy order:
        1. Direct JSON parse
        2. Extract from markdown code blocks
        3. Extract JSON object via regex
        4. Fix common issues and retry
        5. Regex-based item extraction (last resort)
        
        Args:
            text: Raw text from LLM
            page_num: Page number for logging
            
        Returns:
            Parsed dict or None if all strategies fail
        """
        if not text or not text.strip():
            logger.warning(f"[Page {page_num}] Empty response")
            return None
        
        text = text.strip()
        
        # Strategy 1: Direct parse
        result = self._try_direct_parse(text, page_num)
        if result:
            return result
        
        # Strategy 2: Extract from code blocks
        result = self._try_code_block_parse(text, page_num)
        if result:
            return result
        
        # Strategy 3: Extract JSON object
        result = self._try_json_object_parse(text, page_num)
        if result:
            return result
        
        # Strategy 4: Fix and retry
        result = self._try_fixed_parse(text, page_num)
        if result:
            return result
        
        # Strategy 5: Regex extraction (last resort)
        result = self._try_regex_extraction(text, page_num)
        if result:
            return result
        
        logger.warning(f"[Page {page_num}] All parsing strategies failed")
        return None
    
    def _try_direct_parse(self, text: str, page_num: int) -> Optional[Dict]:
        """Try to parse text directly as JSON."""
        try:
            data = json.loads(text)
            if self._validate_structure(data):
                logger.debug(f"[Page {page_num}] Direct parse successful")
                return data
        except json.JSONDecodeError:
            pass
        return None
    
    def _try_code_block_parse(self, text: str, page_num: int) -> Optional[Dict]:
        """Extract JSON from markdown code blocks."""
        match = self.json_block_pattern.search(text)
        if match:
            json_text = match.group(1).strip()
            try:
                data = json.loads(json_text)
                if self._validate_structure(data):
                    logger.debug(f"[Page {page_num}] Code block parse successful")
                    return data
            except json.JSONDecodeError:
                pass
        return None
    
    def _try_json_object_parse(self, text: str, page_num: int) -> Optional[Dict]:
        """Extract JSON object using regex."""
        match = self.json_object_pattern.search(text)
        if match:
            json_text = match.group()
            try:
                data = json.loads(json_text)
                if self._validate_structure(data):
                    logger.debug(f"[Page {page_num}] JSON object parse successful")
                    return data
            except json.JSONDecodeError:
                pass
        return None
    
    def _try_fixed_parse(self, text: str, page_num: int) -> Optional[Dict]:
        """Apply fixes and try to parse."""
        # Extract potential JSON portion
        match = self.json_object_pattern.search(text)
        if not match:
            return None
        
        json_text = match.group()
        
        # Apply progressive fixes
        fixed_text = self._fix_json_issues(json_text)
        
        try:
            data = json.loads(fixed_text)
            if self._validate_structure(data):
                logger.debug(f"[Page {page_num}] Fixed parse successful")
                return data
        except json.JSONDecodeError as e:
            logger.debug(f"[Page {page_num}] Fixed parse failed: {e}")
        
        return None
    
    def _fix_json_issues(self, text: str) -> str:
        """
        Fix common JSON issues in LLM outputs.
        """
        # Step 1: Remove any BOM or weird unicode
        text = text.encode('utf-8', errors='ignore').decode('utf-8')
        
        # Step 2: Fix newlines inside strings
        text = self._fix_string_newlines(text)
        
        # Step 3: Fix trailing commas
        text = re.sub(r',\s*}', '}', text)
        text = re.sub(r',\s*]', ']', text)
        
        # Step 4: Fix missing commas between items
        text = re.sub(r'}\s*{', '},{', text)
        
        # Step 5: Fix unquoted keys (rare but possible)
        text = re.sub(r'(\s*)(\w+)(\s*):(\s*)', r'\1"\2"\3:\4', text)
        # But this might double-quote already quoted keys, so fix that
        text = re.sub(r'""(\w+)""', r'"\1"', text)
        
        # Step 6: Fix truncated JSON
        text = self._fix_truncation(text)
        
        # Step 7: Fix single quotes (Python-style) to double quotes
        # Only for simple cases - this is tricky
        # text = text.replace("'", '"')  # Too aggressive
        
        return text
    
    def _fix_string_newlines(self, text: str) -> str:
        """Remove newlines from inside JSON strings."""
        result = []
        in_string = False
        escape_next = False
        
        for char in text:
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
        
        return ''.join(result)
    
    def _fix_truncation(self, text: str) -> str:
        """Fix truncated JSON by closing open brackets."""
        # Count brackets
        open_braces = text.count('{') - text.count('}')
        open_brackets = text.count('[') - text.count(']')
        
        if open_braces > 0 or open_brackets > 0:
            # Try to remove incomplete last item
            # Look for last complete item
            last_complete = self._find_last_complete_item(text)
            if last_complete > 0:
                text = text[:last_complete + 1]
            
            # Recount and close
            open_braces = text.count('{') - text.count('}')
            open_brackets = text.count('[') - text.count(']')
            
            # Add closing brackets
            text = text.rstrip(',').rstrip()
            text += ']' * max(0, open_brackets)
            text += '}' * max(0, open_braces)
        
        return text
    
    def _find_last_complete_item(self, text: str) -> int:
        """Find the position of the last complete JSON item."""
        # Look for the last properly closed item in bill_items
        # Pattern: complete object ending with }
        
        # Find all complete objects
        positions = []
        depth = 0
        start = -1
        
        for i, char in enumerate(text):
            if char == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0 and start >= 0:
                    # Check if this looks like a bill item
                    snippet = text[start:i+1]
                    if '"item_name"' in snippet or '"item_amount"' in snippet:
                        positions.append(i)
        
        return positions[-1] if positions else -1
    
    def _try_regex_extraction(self, text: str, page_num: int) -> Optional[Dict]:
        """
        Last resort: Extract items using regex patterns.
        Handles severely malformed JSON.
        """
        items = []
        
        # Pattern 1: item_name followed by item_amount
        pattern1 = re.compile(
            r'"item_name"\s*:\s*"([^"]+)"[^}]*?'
            r'"item_amount"\s*:\s*([\d,]+\.?\d*)',
            re.DOTALL | re.IGNORECASE
        )
        
        # Pattern 2: item_amount followed by item_name
        pattern2 = re.compile(
            r'"item_amount"\s*:\s*([\d,]+\.?\d*)[^}]*?'
            r'"item_name"\s*:\s*"([^"]+)"',
            re.DOTALL | re.IGNORECASE
        )
        
        # Pattern 3: Looser pattern for edge cases
        pattern3 = re.compile(
            r'item_name["\s:]+([^"]+)["\s,]+.*?'
            r'item_amount["\s:]+(\d+\.?\d*)',
            re.DOTALL | re.IGNORECASE
        )
        
        # Try patterns in order
        for match in pattern1.finditer(text):
            name = match.group(1).strip()
            amount = self._parse_number(match.group(2))
            if name and amount > 0:
                item = self._extract_full_item(text, match.start(), match.end(), name, amount)
                items.append(item)
        
        if not items:
            for match in pattern2.finditer(text):
                amount = self._parse_number(match.group(1))
                name = match.group(2).strip()
                if name and amount > 0:
                    item = self._extract_full_item(text, match.start(), match.end(), name, amount)
                    items.append(item)
        
        if not items:
            for match in pattern3.finditer(text):
                name = match.group(1).strip().strip('"')
                amount = self._parse_number(match.group(2))
                if name and amount > 0:
                    items.append({
                        "item_name": name,
                        "item_amount": amount
                    })
        
        if items:
            # Deduplicate
            items = self._deduplicate_items(items)
            
            # Detect page type
            page_type = self._detect_page_type(text)
            
            logger.info(f"[Page {page_num}] Regex extracted {len(items)} items")
            return {
                "page_type": page_type,
                "bill_items": items
            }
        
        return None
    
    def _extract_full_item(self, text: str, start: int, end: int, 
                          name: str, amount: float) -> Dict:
        """Extract full item including rate and quantity from context."""
        # Look for rate and quantity in surrounding context
        context_start = max(0, start - 50)
        context_end = min(len(text), end + 200)
        context = text[context_start:context_end]
        
        item = {
            "item_name": name,
            "item_amount": amount
        }
        
        # Extract rate
        rate_match = re.search(
            r'"item_rate"\s*:\s*([\d,]+\.?\d*)',
            context,
            re.IGNORECASE
        )
        if rate_match:
            rate = self._parse_number(rate_match.group(1))
            if rate > 0:
                item["item_rate"] = rate
        
        # Extract quantity
        qty_match = re.search(
            r'"item_quantity"\s*:\s*([\d,]+\.?\d*)',
            context,
            re.IGNORECASE
        )
        if qty_match:
            qty = self._parse_number(qty_match.group(1))
            if qty > 0:
                item["item_quantity"] = qty
        
        return item
    
    def _parse_number(self, s: str) -> float:
        """Parse a number string, handling commas and currency symbols."""
        if not s:
            return 0.0
        try:
            # Remove commas and currency symbols
            s = s.replace(',', '').replace('₹', '').replace('Rs', '').strip()
            return float(s)
        except (ValueError, TypeError):
            return 0.0
    
    def _deduplicate_items(self, items: List[Dict]) -> List[Dict]:
        """Remove duplicate items based on name and amount."""
        seen = set()
        unique = []
        
        for item in items:
            # Create a key (lowercase name + amount)
            key = (
                item.get('item_name', '').lower().strip(),
                round(item.get('item_amount', 0), 2)
            )
            
            if key not in seen and key[0] and key[1] > 0:
                seen.add(key)
                unique.append(item)
        
        return unique
    
    def _detect_page_type(self, text: str) -> str:
        """Detect page type from text content."""
        text_lower = text.lower()
        
        # Check for explicit page_type in response
        type_match = re.search(
            r'"page_type"\s*:\s*"([^"]+)"',
            text,
            re.IGNORECASE
        )
        if type_match:
            explicit_type = type_match.group(1)
            if explicit_type in ['Pharmacy', 'Final Bill', 'Bill Detail', 
                                'Investigation', 'Consultation', 'Room Charges']:
                return explicit_type
        
        # Infer from content
        if any(kw in text_lower for kw in ['pharmacy', 'medicine', 'tablet', 
                                           'capsule', 'syrup', 'injection', 'mg', 'ml']):
            return 'Pharmacy'
        elif any(kw in text_lower for kw in ['final bill', 'grand total', 
                                              'total payable', 'net amount']):
            return 'Final Bill'
        elif any(kw in text_lower for kw in ['investigation', 'lab', 'test', 
                                              'pathology', 'radiology']):
            return 'Investigation'
        elif any(kw in text_lower for kw in ['consultation', 'doctor', 'visit']):
            return 'Consultation'
        elif any(kw in text_lower for kw in ['room', 'bed', 'accommodation']):
            return 'Room Charges'
        
        return 'Bill Detail'
    
    def _validate_structure(self, data: Any) -> bool:
        """
        Validate that parsed data has expected structure.
        """
        if not isinstance(data, dict):
            return False
        
        # Must have bill_items (even if empty)
        if 'bill_items' not in data:
            return False
        
        if not isinstance(data['bill_items'], list):
            return False
        
        # Validate items have required fields
        for item in data['bill_items']:
            if not isinstance(item, dict):
                return False
            # Must have at least item_name and item_amount
            if 'item_name' not in item or 'item_amount' not in item:
                # Try alternative field names
                if not ('name' in item or 'description' in item):
                    continue  # Skip invalid items
        
        return True


class ResponseValidator:
    """
    Validates and cleans extracted data for consistency and accuracy.
    """
    
    def __init__(self):
        # Keywords that indicate non-item rows
        self.skip_keywords = [
            'total', 'subtotal', 'sub-total', 'grand total',
            'net amount', 'net payable', 'amount payable',
            'discount', 'tax', 'gst', 'cgst', 'sgst', 'igst',
            'advance', 'deposit', 'adjustment', 'balance',
            'page', 'continued', 'header', 'footer'
        ]
        
        # Minimum item name length
        self.min_name_length = 3
    
    def validate_and_clean(self, data: Dict, page_num: int = 1) -> Dict:
        """
        Validate and clean extraction results.
        
        Args:
            data: Parsed extraction data
            page_num: Page number for logging
            
        Returns:
            Cleaned and validated data
        """
        if not data or 'bill_items' not in data:
            return {
                "page_type": "Bill Detail",
                "bill_items": []
            }
        
        cleaned_items = []
        warnings = []
        
        for item in data.get('bill_items', []):
            cleaned_item, item_warnings = self._validate_item(item)
            if cleaned_item:
                cleaned_items.append(cleaned_item)
            warnings.extend(item_warnings)
        
        result = {
            "page_type": data.get('page_type', 'Bill Detail'),
            "bill_items": cleaned_items
        }
        
        if warnings:
            logger.debug(f"[Page {page_num}] Validation warnings: {warnings}")
        
        return result
    
    def _validate_item(self, item: Dict) -> Tuple[Optional[Dict], List[str]]:
        """
        Validate and clean a single item.
        
        Returns:
            Tuple of (cleaned item or None, list of warnings)
        """
        warnings = []
        
        # Get item name
        name = item.get('item_name', item.get('name', item.get('description', '')))
        name = str(name).strip() if name else ''
        
        # Skip if name is empty or too short
        if len(name) < self.min_name_length:
            return None, [f"Skipped item with short name: '{name}'"]
        
        # Skip if name matches skip keywords
        name_lower = name.lower()
        if any(kw in name_lower for kw in self.skip_keywords):
            return None, [f"Skipped total/header row: '{name}'"]
        
        # Get amount
        amount = self._parse_amount(item.get('item_amount', item.get('amount', 0)))
        
        # Skip zero or negative amounts
        if amount <= 0:
            return None, [f"Skipped item with invalid amount: '{name}' = {amount}"]
        
        # Build cleaned item
        cleaned = {
            "item_name": self._clean_name(name),
            "item_amount": round(amount, 2)
        }
        
        # Add optional fields if valid
        rate = self._parse_amount(item.get('item_rate', item.get('rate', 0)))
        if rate > 0:
            cleaned["item_rate"] = round(rate, 2)
        
        quantity = self._parse_quantity(item.get('item_quantity', item.get('quantity', 0)))
        if quantity > 0:
            cleaned["item_quantity"] = quantity
        
        # Cross-validation
        if rate > 0 and quantity > 0:
            expected_amount = rate * quantity
            if abs(expected_amount - amount) > max(1.0, amount * 0.1):
                warnings.append(
                    f"Amount mismatch for '{name}': "
                    f"{rate} × {quantity} = {expected_amount} ≠ {amount}"
                )
        
        return cleaned, warnings
    
    def _clean_name(self, name: str) -> str:
        """Clean item name."""
        # Remove leading numbers/symbols
        name = re.sub(r'^[\d\.\-\)\]\s]+', '', name)
        # Remove trailing punctuation
        name = name.strip('.,;:-() ')
        # Normalize whitespace
        name = ' '.join(name.split())
        return name
    
    def _parse_amount(self, value) -> float:
        """Parse monetary amount."""
        if value is None:
            return 0.0
        if isinstance(value, (int, float)):
            return float(value)
        try:
            s = str(value).replace(',', '').replace('₹', '').replace('Rs.', '').replace('Rs', '').strip()
            match = re.search(r'[\d.]+', s)
            return float(match.group()) if match else 0.0
        except (ValueError, TypeError):
            return 0.0
    
    def _parse_quantity(self, value) -> float:
        """Parse quantity."""
        if value is None:
            return 0.0
        if isinstance(value, (int, float)):
            return float(value)
        try:
            s = re.sub(r'\s*(No|Nos|Units?|Pcs?|Qty)\.?\s*', '', str(value), flags=re.IGNORECASE)
            match = re.search(r'[\d.]+', s)
            return float(match.group()) if match else 0.0
        except (ValueError, TypeError):
            return 0.0