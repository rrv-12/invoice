"""
prompts.py - Optimized prompts for Gemini 2.5 Flash medical invoice extraction
"""

# Main extraction prompt - optimized for accuracy
EXTRACTION_PROMPT_V1 = """You are a precise medical bill data extractor. Your task is to extract ALL line items from this hospital/medical bill image.

## OUTPUT FORMAT
Return ONLY a JSON object in this exact format:
{
  "page_type": "Bill Detail",
  "bill_items": [
    {
      "item_name": "Full item description",
      "item_amount": 123.45,
      "item_rate": 123.45,
      "item_quantity": 1
    }
  ]
}

## FIELD DEFINITIONS
- page_type: One of "Bill Detail", "Pharmacy", "Final Bill", "Investigation", "Consultation", "Room Charges", "Services"
- item_name: Complete description of the item/service (preserve full text)
- item_amount: The NET/TOTAL amount for this line item (usually rightmost column). This is the final amount after any per-line discounts.
- item_rate: Unit price/rate per item (if shown)
- item_quantity: Numeric quantity only (1, 2, 3...), ignore "No", "Nos", "Units"

## EXTRACTION RULES
1. Extract EVERY line item that has a price/amount
2. For item_amount, use the RIGHTMOST amount column (net amount, not gross)
3. If multiple amount columns exist (Gross, Discount, Net), use the NET amount
4. Preserve FULL item descriptions - do not truncate
5. Include medicines, procedures, consultations, room charges, tests, etc.
6. For pharmacy items, include drug strength (mg, ml) if shown

## WHAT TO SKIP
- Page headers/footers
- Column headers (Sr No, Description, Rate, Qty, Amount)
- Section totals (Sub Total, Grand Total, Total Amount)
- Tax lines (GST, CGST, SGST)
- Discount summary lines
- Empty rows

## PAGE TYPE DETECTION
- "Pharmacy": Contains medicines, tablets, capsules, syrups, injections
- "Investigation": Lab tests, pathology, radiology, X-ray, MRI, CT
- "Consultation": Doctor visits, consultation fees
- "Room Charges": Bed charges, room rent, accommodation
- "Final Bill": Summary page with grand totals
- "Bill Detail": General itemized charges

## EXAMPLES

Example 1 - Pharmacy items:
{
  "page_type": "Pharmacy",
  "bill_items": [
    {"item_name": "TAB PARACETAMOL 500MG", "item_amount": 45.00, "item_rate": 4.50, "item_quantity": 10},
    {"item_name": "INJ CEFTRIAXONE 1GM", "item_amount": 280.00, "item_rate": 140.00, "item_quantity": 2}
  ]
}

Example 2 - Investigation items:
{
  "page_type": "Investigation",
  "bill_items": [
    {"item_name": "CBC (Complete Blood Count)", "item_amount": 450.00, "item_rate": 450.00, "item_quantity": 1},
    {"item_name": "LIPID PROFILE", "item_amount": 800.00, "item_rate": 800.00, "item_quantity": 1}
  ]
}

## CRITICAL INSTRUCTIONS
- Return ONLY valid JSON - no markdown, no explanations
- If no items found, return: {"page_type": "Bill Detail", "bill_items": []}
- Numbers must be numeric (123.45), not strings ("123.45")
- Ensure all JSON brackets are properly closed

Extract all line items from this bill image now:"""


# Alternative prompt with more structure
EXTRACTION_PROMPT_V2 = """TASK: Extract line items from medical bill image

OUTPUT: JSON only, exact format:
{"page_type":"TYPE","bill_items":[{"item_name":"NAME","item_amount":AMT,"item_rate":RATE,"item_quantity":QTY}]}

RULES:
1. item_amount = Net amount (rightmost column)
2. item_rate = Unit price
3. item_quantity = Number only
4. SKIP totals, taxes, headers
5. page_type: Pharmacy|Investigation|Consultation|Room Charges|Bill Detail|Final Bill

EXTRACT NOW:"""


# Prompt for retries with additional context
RETRY_PROMPT = """Previous extraction may have missed items. Please carefully re-examine this medical bill image.

Focus on:
1. Items in table rows with amounts
2. Small or faded text
3. Multi-line item descriptions
4. Items near page edges

OUTPUT FORMAT (JSON only):
{
  "page_type": "Bill Detail",
  "bill_items": [
    {"item_name": "description", "item_amount": 0.00, "item_rate": 0.00, "item_quantity": 1}
  ]
}

IMPORTANT:
- Extract ALL items with prices
- Use NET amount (rightmost column)
- No markdown, only JSON
- Empty result if no items: {"page_type": "Bill Detail", "bill_items": []}

Extract all line items:"""


# Prompt with text context (for digital PDFs)
def get_text_enhanced_prompt(extracted_text: str) -> str:
    """Generate prompt with text context for digital PDFs."""
    # Truncate text if too long
    if len(extracted_text) > 3000:
        extracted_text = extracted_text[:3000] + "..."
    
    return f"""You are extracting line items from a medical bill. The page contains the following text:

---TEXT START---
{extracted_text}
---TEXT END---

Using BOTH the image AND the text above, extract ALL line items.

OUTPUT FORMAT (JSON only):
{{
  "page_type": "Bill Detail",
  "bill_items": [
    {{"item_name": "Full description", "item_amount": 123.45, "item_rate": 123.45, "item_quantity": 1}}
  ]
}}

RULES:
1. item_amount = Net/Total amount for the line (rightmost amount column)
2. item_rate = Unit price/rate (if available)
3. item_quantity = Numeric quantity only
4. SKIP: Headers, totals, subtotals, tax lines
5. Include ALL items with prices

page_type options: Pharmacy, Investigation, Consultation, Room Charges, Bill Detail, Final Bill

Return ONLY valid JSON. No explanations."""


# Section-specific prompts
PHARMACY_PROMPT = """Extract PHARMACY/MEDICINE items from this bill image.

Look for:
- Tablet names (TAB, TABLET)
- Capsules (CAP, CAPSULE)
- Syrups (SYR, SYRUP)
- Injections (INJ, INJECTION)
- Drug strengths (MG, ML, MCG)

OUTPUT (JSON only):
{
  "page_type": "Pharmacy",
  "bill_items": [
    {"item_name": "MEDICINE NAME WITH STRENGTH", "item_amount": 0.00, "item_rate": 0.00, "item_quantity": 1}
  ]
}

Extract pharmacy items:"""


INVESTIGATION_PROMPT = """Extract INVESTIGATION/LAB TEST items from this bill image.

Look for:
- Blood tests (CBC, Hemoglobin, etc.)
- Urine tests
- Pathology reports
- Radiology (X-Ray, CT, MRI, USG)
- ECG, Echo, etc.

OUTPUT (JSON only):
{
  "page_type": "Investigation",
  "bill_items": [
    {"item_name": "TEST NAME", "item_amount": 0.00, "item_rate": 0.00, "item_quantity": 1}
  ]
}

Extract investigation items:"""


# Prompt selector based on context
def select_prompt(page_text: str = "", attempt: int = 1, detected_type: str = None) -> str:
    """
    Select the most appropriate prompt based on context.
    
    Args:
        page_text: Extracted text from the page (if available)
        attempt: Retry attempt number (1, 2, 3...)
        detected_type: Pre-detected page type
        
    Returns:
        Selected prompt string
    """
    # Use retry prompt for subsequent attempts
    if attempt > 1:
        return RETRY_PROMPT
    
    # If we have significant text, use text-enhanced prompt
    if page_text and len(page_text) > 200:
        return get_text_enhanced_prompt(page_text)
    
    # Use section-specific prompts if type is detected
    if detected_type:
        type_lower = detected_type.lower()
        if 'pharmacy' in type_lower or 'medicine' in type_lower:
            return PHARMACY_PROMPT
        elif 'investigation' in type_lower or 'lab' in type_lower:
            return INVESTIGATION_PROMPT
    
    # Default to main prompt
    return EXTRACTION_PROMPT_V1


# Generation config for deterministic extraction
GENERATION_CONFIG = {
    "temperature": 0,  # Deterministic output
    "max_output_tokens": 4096,  # Allow for large responses
    "top_p": 1,
    "top_k": 1
}


# Alternative config for retries (slightly more creative)
RETRY_GENERATION_CONFIG = {
    "temperature": 0.1,
    "max_output_tokens": 4096,
    "top_p": 0.95,
    "top_k": 40
}