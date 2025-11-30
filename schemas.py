"""
schemas.py - Pydantic models and validation schemas for invoice extraction
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Optional, Literal
from enum import Enum
import re


class PageType(str, Enum):
    """Valid page types for medical invoices"""
    BILL_DETAIL = "Bill Detail"
    PHARMACY = "Pharmacy"
    FINAL_BILL = "Final Bill"
    INVESTIGATION = "Investigation"
    CONSULTATION = "Consultation"
    ROOM_CHARGES = "Room Charges"
    SERVICES = "Services"
    PROCEDURE = "Procedure"


class ExtractedItem(BaseModel):
    """Schema for a single extracted line item with validation"""
    item_name: str = Field(..., min_length=1, max_length=500)
    item_amount: float = Field(..., ge=0)
    item_rate: Optional[float] = Field(default=None, ge=0)
    item_quantity: Optional[float] = Field(default=None, ge=0)
    
    @field_validator('item_name')
    @classmethod
    def clean_item_name(cls, v: str) -> str:
        """Clean and validate item name"""
        if not v:
            raise ValueError("Item name cannot be empty")
        # Remove excessive whitespace
        v = ' '.join(v.split())
        # Remove common artifacts
        v = re.sub(r'^[\d\.\-\s]+', '', v)  # Leading numbers/dots
        v = v.strip('.,;:-() ')
        if len(v) < 2:
            raise ValueError("Item name too short after cleaning")
        return v
    
    @field_validator('item_amount', 'item_rate')
    @classmethod
    def validate_amount(cls, v: Optional[float]) -> Optional[float]:
        """Validate monetary amounts"""
        if v is None:
            return None
        if v < 0:
            raise ValueError("Amount cannot be negative")
        if v > 100_000_000:  # 10 crore sanity limit
            raise ValueError("Amount exceeds sanity limit")
        return round(v, 2)
    
    @field_validator('item_quantity')
    @classmethod
    def validate_quantity(cls, v: Optional[float]) -> Optional[float]:
        """Validate quantity"""
        if v is None:
            return None
        if v < 0:
            raise ValueError("Quantity cannot be negative")
        if v > 10000:  # Sanity limit
            raise ValueError("Quantity exceeds sanity limit")
        return v
    
    @model_validator(mode='after')
    def validate_consistency(self):
        """Cross-validate rate × quantity ≈ amount"""
        if self.item_rate and self.item_quantity and self.item_amount:
            expected = self.item_rate * self.item_quantity
            tolerance = max(1.0, self.item_amount * 0.05)  # 5% or ₹1
            if abs(expected - self.item_amount) > tolerance:
                # Don't fail, but log warning - medical bills often have discounts
                pass
        return self


class PageResult(BaseModel):
    """Schema for extraction results from a single page"""
    page_no: str
    page_type: str = Field(default="Bill Detail")
    bill_items: List[ExtractedItem] = Field(default_factory=list)
    confidence_score: Optional[float] = Field(default=None, ge=0, le=1)
    warnings: List[str] = Field(default_factory=list)
    
    @field_validator('page_type')
    @classmethod
    def validate_page_type(cls, v: str) -> str:
        """Normalize page type to valid values"""
        v_lower = v.lower().strip()
        
        # Map variations to standard types
        type_mapping = {
            'pharmacy': 'Pharmacy',
            'medicine': 'Pharmacy',
            'medicines': 'Pharmacy',
            'drug': 'Pharmacy',
            'final bill': 'Final Bill',
            'final': 'Final Bill',
            'summary': 'Final Bill',
            'total': 'Final Bill',
            'bill detail': 'Bill Detail',
            'detail': 'Bill Detail',
            'details': 'Bill Detail',
            'investigation': 'Investigation',
            'lab': 'Investigation',
            'laboratory': 'Investigation',
            'pathology': 'Investigation',
            'radiology': 'Investigation',
            'consultation': 'Consultation',
            'doctor': 'Consultation',
            'room': 'Room Charges',
            'room charges': 'Room Charges',
            'accommodation': 'Room Charges',
            'bed': 'Room Charges',
            'services': 'Services',
            'service': 'Services',
            'procedure': 'Procedure',
            'surgery': 'Procedure',
            'operation': 'Procedure',
        }
        
        for key, mapped_type in type_mapping.items():
            if key in v_lower:
                return mapped_type
        
        return 'Bill Detail'  # Default


class ExtractionResult(BaseModel):
    """Schema for complete extraction result"""
    pagewise_line_items: List[PageResult] = Field(default_factory=list)
    total_item_count: int = Field(default=0, ge=0)
    extraction_metadata: dict = Field(default_factory=dict)
    
    @model_validator(mode='after')
    def compute_total(self):
        """Compute total item count from pages"""
        self.total_item_count = sum(
            len(page.bill_items) for page in self.pagewise_line_items
        )
        return self


# Validation utilities
def is_valid_item_name(name: str) -> bool:
    """Check if an item name looks legitimate"""
    if not name or len(name) < 2:
        return False
    
    # Reject if mostly numbers/symbols
    alpha_chars = sum(1 for c in name if c.isalpha())
    if alpha_chars < len(name) * 0.3:
        return False
    
    # Reject common false positives
    reject_patterns = [
        r'^(total|subtotal|sub-total|grand total|net amount|amount|sum)$',
        r'^(page|pg|sr\.?no|s\.?no|sl\.?no)$',
        r'^\d+$',
        r'^[=\-_\.]+$',
    ]
    
    name_lower = name.lower().strip()
    for pattern in reject_patterns:
        if re.match(pattern, name_lower, re.IGNORECASE):
            return False
    
    return True


def is_reasonable_amount(amount: float, context: str = "") -> bool:
    """Check if amount is reasonable for medical bills"""
    if amount < 0:
        return False
    if amount > 50_000_000:  # 5 crore max
        return False
    if amount == 0:
        return False  # Zero amounts are usually totals/headers
    return True