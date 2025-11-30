"""
preprocessor.py - Image preprocessing for optimal OCR/Vision extraction
"""

import logging
from io import BytesIO
from typing import Tuple, Optional
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """
    Preprocesses images for optimal extraction quality.
    Handles deskewing, contrast enhancement, noise reduction, and resizing.
    """
    
    # Optimal dimensions for Gemini Vision
    TARGET_MAX_DIM = 1600  # Higher res for better text recognition
    TARGET_MIN_DIM = 800   # Minimum for readable text
    
    # Quality thresholds
    MIN_CONTRAST_RATIO = 0.3
    OPTIMAL_DPI = 150
    
    def __init__(self, 
                 target_max_dim: int = 1600,
                 enhance_contrast: bool = True,
                 denoise: bool = True,
                 auto_orient: bool = True):
        """
        Initialize preprocessor with configuration.
        
        Args:
            target_max_dim: Maximum dimension for output images
            enhance_contrast: Whether to apply contrast enhancement
            denoise: Whether to apply noise reduction
            auto_orient: Whether to auto-orient based on EXIF
        """
        self.target_max_dim = target_max_dim
        self.enhance_contrast = enhance_contrast
        self.denoise = denoise
        self.auto_orient = auto_orient
    
    def process(self, image: Image.Image, page_num: int = 1) -> Image.Image:
        """
        Main preprocessing pipeline.
        
        Args:
            image: PIL Image to process
            page_num: Page number for logging
            
        Returns:
            Processed PIL Image
        """
        original_size = image.size
        logger.debug(f"[Page {page_num}] Original size: {original_size}")
        
        # Step 1: Convert to RGB if needed
        image = self._ensure_rgb(image)
        
        # Step 2: Auto-orient based on EXIF
        if self.auto_orient:
            image = self._auto_orient(image)
        
        # Step 3: Analyze image quality
        quality_info = self._analyze_quality(image)
        logger.debug(f"[Page {page_num}] Quality: {quality_info}")
        
        # Step 4: Resize to optimal dimensions
        image = self._smart_resize(image, page_num)
        
        # Step 5: Enhance contrast if needed
        if self.enhance_contrast and quality_info.get('low_contrast', False):
            image = self._enhance_contrast(image)
        
        # Step 6: Denoise if needed
        if self.denoise and quality_info.get('noisy', False):
            image = self._reduce_noise(image)
        
        # Step 7: Sharpen for text clarity
        image = self._sharpen_text(image)
        
        logger.debug(f"[Page {page_num}] Final size: {image.size}")
        return image
    
    def _ensure_rgb(self, image: Image.Image) -> Image.Image:
        """Convert image to RGB mode if needed."""
        if image.mode == 'RGBA':
            # Create white background for transparency
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[-1])
            return background
        elif image.mode != 'RGB':
            return image.convert('RGB')
        return image
    
    def _auto_orient(self, image: Image.Image) -> Image.Image:
        """Auto-orient image based on EXIF data."""
        try:
            return ImageOps.exif_transpose(image)
        except Exception:
            return image
    
    def _analyze_quality(self, image: Image.Image) -> dict:
        """
        Analyze image quality to determine preprocessing needs.
        
        Returns:
            Dict with quality indicators
        """
        quality = {
            'low_contrast': False,
            'noisy': False,
            'blurry': False,
            'small': False
        }
        
        try:
            # Convert to grayscale for analysis
            gray = image.convert('L')
            pixels = list(gray.getdata())
            
            # Check contrast (standard deviation of pixel values)
            mean_val = sum(pixels) / len(pixels)
            variance = sum((p - mean_val) ** 2 for p in pixels) / len(pixels)
            std_dev = variance ** 0.5
            
            if std_dev < 40:  # Low contrast threshold
                quality['low_contrast'] = True
            
            # Check if image is small
            if max(image.size) < self.TARGET_MIN_DIM:
                quality['small'] = True
            
            # Simple noise detection via edge density
            # High edge density in smooth areas indicates noise
            # This is a simplified heuristic
            
        except Exception as e:
            logger.warning(f"Quality analysis failed: {e}")
        
        return quality
    
    def _smart_resize(self, image: Image.Image, page_num: int = 1) -> Image.Image:
        """
        Intelligently resize image to optimal dimensions.
        
        - Maintains aspect ratio
        - Uses high-quality resampling
        - Avoids upscaling small images too much
        """
        width, height = image.size
        max_dim = max(width, height)
        
        if max_dim <= self.target_max_dim:
            # Only upscale if image is very small
            if max_dim < self.TARGET_MIN_DIM:
                scale = self.TARGET_MIN_DIM / max_dim
                scale = min(scale, 2.0)  # Don't upscale more than 2x
                new_size = (int(width * scale), int(height * scale))
                logger.debug(f"[Page {page_num}] Upscaling from {image.size} to {new_size}")
                return image.resize(new_size, Image.LANCZOS)
            return image
        
        # Downscale large images
        scale = self.target_max_dim / max_dim
        new_size = (int(width * scale), int(height * scale))
        logger.debug(f"[Page {page_num}] Downscaling from {image.size} to {new_size}")
        return image.resize(new_size, Image.LANCZOS)
    
    def _enhance_contrast(self, image: Image.Image) -> Image.Image:
        """Apply adaptive contrast enhancement."""
        try:
            # Auto-contrast first
            image = ImageOps.autocontrast(image, cutoff=0.5)
            
            # Then apply moderate contrast boost
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.2)
            
            return image
        except Exception as e:
            logger.warning(f"Contrast enhancement failed: {e}")
            return image
    
    def _reduce_noise(self, image: Image.Image) -> Image.Image:
        """Apply noise reduction filter."""
        try:
            # Median filter for salt-and-pepper noise
            return image.filter(ImageFilter.MedianFilter(size=3))
        except Exception as e:
            logger.warning(f"Noise reduction failed: {e}")
            return image
    
    def _sharpen_text(self, image: Image.Image) -> Image.Image:
        """Apply mild sharpening to improve text clarity."""
        try:
            enhancer = ImageEnhance.Sharpness(image)
            return enhancer.enhance(1.3)
        except Exception as e:
            logger.warning(f"Sharpening failed: {e}")
            return image
    
    def process_for_ocr(self, image: Image.Image) -> Image.Image:
        """
        Process image specifically for OCR (grayscale + binarization).
        Use this for text extraction fallback.
        """
        # Convert to grayscale
        gray = image.convert('L')
        
        # Apply adaptive thresholding (simple version)
        # For true adaptive thresholding, would need OpenCV
        threshold = 128
        return gray.point(lambda p: 255 if p > threshold else 0)


class PDFPageConverter:
    """Converts PDF pages to images with optimal settings."""
    
    def __init__(self, 
                 zoom: float = 2.0,
                 max_dim: int = 1600):
        """
        Initialize PDF converter.
        
        Args:
            zoom: Zoom factor for PDF to image conversion
            max_dim: Maximum dimension for output images
        """
        self.zoom = zoom
        self.max_dim = max_dim
        self.preprocessor = ImagePreprocessor(target_max_dim=max_dim)
    
    def convert_page(self, pdf_page, page_num: int = 1) -> Tuple[Image.Image, str]:
        """
        Convert a PDF page to a processed image.
        
        Args:
            pdf_page: PyMuPDF page object
            page_num: Page number for logging
            
        Returns:
            Tuple of (processed image, extracted text)
        """
        import fitz
        
        # Extract text first (for digital PDFs)
        text = pdf_page.get_text("text").strip()
        
        # Convert to image with zoom
        matrix = fitz.Matrix(self.zoom, self.zoom)
        pix = pdf_page.get_pixmap(matrix=matrix)
        
        # Convert to PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Apply preprocessing
        img = self.preprocessor.process(img, page_num)
        
        return img, text
    
    def is_digital_pdf(self, pdf_page) -> bool:
        """
        Check if a PDF page has selectable text (digital vs scanned).
        
        Returns:
            True if page appears to be digital (has text layer)
        """
        text = pdf_page.get_text("text").strip()
        # Consider digital if more than 100 characters of text
        return len(text) > 100