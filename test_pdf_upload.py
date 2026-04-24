import requests
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io

# Create a test PDF
pdf_buffer = io.BytesIO()
c = canvas.Canvas(pdf_buffer, pagesize=letter)
c.drawString(100, 750, "Test Resume")
c.drawString(100, 730, "Name: John Doe")
c.drawString(100, 710, "Skills: Python, JavaScript, React")
c.drawString(100, 690, "Experience: 5 years in software development")
c.save()
pdf_buffer.seek(0)

# Test PDF upload
files = {'file': ('test_resume.pdf', pdf_buffer, 'application/pdf')}
response = requests.post('http://localhost:10000/extract-pdf', files=files)

print(f"Status Code: {response.status_code}")
print(f"Response: {response.json()}")

if response.status_code == 200:
    data = response.json()
    if 'extracted_text' in data:
        print("\n✓ PDF extraction successful!")
        print(f"Extracted text length: {len(data['extracted_text'])} characters")
        print(f"First 300 chars:\n{data['extracted_text'][:300]}")
    else:
        print("✗ No extracted_text in response")
else:
    print("✗ Upload failed")
