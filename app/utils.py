import logging
import io
import pdfplumber

logger = logging.getLogger(__name__)

def extract_text_from_bytes(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes"""
    try:
        pdf_stream = io.BytesIO(pdf_bytes)
        
        text = ""
        
        with pdfplumber.open(pdf_stream) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        
        logger.debug(f"Extracted {len(text)} characters from PDF")
        return text.strip()
        
    except Exception as e:
        logger.error(f"PDF extraction error: {str(e)}")
        return ""

def extract_text_from_upload_file(upload_file) -> str:
    """Extract text from FastAPI UploadFile object"""
    try:
        # Read file content
        pdf_bytes = upload_file.file.read()
        
        # Extract text from bytes
        text = extract_text_from_bytes(pdf_bytes)
        
        logger.info(f"Successfully extracted {len(text)} characters from uploaded PDF")
        return text
        
    except Exception as e:
        logger.error(f"Upload file extraction error: {str(e)}")
        return ""

def validate_pdf_file(upload_file) -> bool:
    """Validate that uploaded file is a PDF"""
    if not upload_file.filename:
        return False
    
    return upload_file.filename.lower().endswith('.pdf')

def sanitize_text(text: str) -> str:
    """Sanitize extracted text by removing excessive whitespace"""
    if not text:
        return ""
    
    # Remove multiple consecutive whitespace
    import re
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text
