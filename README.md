# 🏥 Invoice Data Extraction API

**HackRx Datathon Submission** - AI-powered invoice/bill data extraction using Google Gemini Vision

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![Gemini](https://img.shields.io/badge/Google-Gemini%201.5-orange.svg)](https://ai.google.dev)

## 🎯 Problem Statement

Build an API that extracts line items from multi-page hospital bills/invoices with:
- Accurate item name, quantity, rate, and amount extraction
- Multi-page PDF support
- Bill total reconciliation
- Minimal double-counting or missing items

## 🚀 Solution Overview

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Invoice/PDF URL                          │
│                          │                                  │
│                          ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              FastAPI Endpoint                        │   │
│  │            POST /extract-bill-data                   │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│            ┌─────────────┴─────────────┐                   │
│            ▼                           ▼                    │
│       Single Image                Multi-Page PDF            │
│            │                           │                    │
│            │              ┌────────────┼────────────┐       │
│            │              ▼            ▼            ▼       │
│            │          Page 1-3    Page 4-6    Page 7-9      │
│            │         (Batch 1)   (Batch 2)   (Batch 3)      │
│            │              │            │            │       │
│            └──────────────┴────────────┴────────────┘       │
│                          │                                  │
│                          ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           Google Gemini 1.5 Flash Vision            │   │
│  │              (FREE API - 15 RPM)                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│                          ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              JSON Response Parser                    │   │
│  │     • Fix truncated JSON                            │   │
│  │     • Salvage partial responses                     │   │
│  │     • Deduplicate across pages                      │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│                          ▼                                  │
│               Structured API Response                       │
└─────────────────────────────────────────────────────────────┘
```

### Key Features

| Feature | Description |
|---------|-------------|
| **LLM-Powered** | Uses Google Gemini 1.5 Flash Vision for intelligent extraction |
| **Multi-Page PDF** | Supports PDFs with 10+ pages using batched processing |
| **Smart Batching** | Sends 3 pages per API call for 3x faster processing |
| **Blank Page Detection** | Automatically skips empty/header-only pages |
| **JSON Recovery** | Salvages data from truncated/malformed LLM responses |
| **Deduplication** | Removes duplicate items between summary and detail pages |
| **Token Tracking** | Reports actual LLM token usage in response |

## 📦 Installation

### Prerequisites
- Python 3.10+
- Google Gemini API Key (FREE)

### Setup

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/invoice-extraction.git
cd invoice-extraction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Set API key
export GEMINI_API_KEY="your-api-key-here"  # Linux/Mac
# OR
set GEMINI_API_KEY=your-api-key-here       # Windows CMD
# OR
$env:GEMINI_API_KEY="your-api-key-here"    # Windows PowerShell
```

### Get FREE Gemini API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with Google account
3. Click "Create API Key"
4. Copy and set as environment variable

## 🏃 Running the Server

```bash
python main.py
```

Server starts at `http://localhost:8000`

## 📡 API Endpoint

### POST /extract-bill-data

Extract line items from invoice image or PDF.

**Request:**
```json
{
    "document": "https://example.com/invoice.pdf"
}
```

**Response:**
```json
{
    "is_success": true,
    "token_usage": {
        "total_tokens": 2500,
        "input_tokens": 1800,
        "output_tokens": 700
    },
    "data": {
        "pagewise_line_items": [
            {
                "page_no": "1",
                "page_type": "Bill Detail",
                "bill_items": [
                    {
                        "item_name": "Consultation - Dr. Smith",
                        "item_amount": 500.00,
                        "item_rate": 500.00,
                        "item_quantity": 1
                    },
                    {
                        "item_name": "Blood Test - CBC",
                        "item_amount": 250.00,
                        "item_rate": 250.00,
                        "item_quantity": 1
                    }
                ]
            }
        ],
        "total_item_count": 2
    }
}
```

### Supported Formats
- **Images**: PNG, JPG, JPEG, WEBP
- **Documents**: PDF (multi-page supported)

## 🧪 Testing

### Using cURL

```bash
curl -X POST "http://localhost:8000/extract-bill-data" \
  -H "Content-Type: application/json" \
  -d '{"document": "YOUR_INVOICE_URL"}'
```

### Using Python

```python
import requests

response = requests.post(
    "http://localhost:8000/extract-bill-data",
    json={"document": "YOUR_INVOICE_URL"}
)
print(response.json())
```

### Health Check

```bash
curl http://localhost:8000/health
```

## 🚀 Deployment

### Deploy to Render (Recommended - FREE)

1. Fork this repository
2. Go to [Render Dashboard](https://dashboard.render.com)
3. Click "New" → "Web Service"
4. Connect your GitHub repo
5. Set environment variable: `GEMINI_API_KEY`
6. Deploy!

### Deploy with Docker

```bash
docker build -t invoice-extraction .
docker run -p 8000:8000 -e GEMINI_API_KEY="your-key" invoice-extraction
```

## 📁 Project Structure

```
├── main.py                 # FastAPI application & endpoints
├── invoice_extractor.py    # Core extraction logic with Gemini
├── requirements.txt        # Python dependencies
├── Dockerfile             # Container configuration
├── README.md              # This file
└── .env.example           # Environment variable template
```

## 🔧 Technical Approach

### 1. Document Processing
- Downloads document from URL
- Detects format (PDF vs Image)
- For PDFs: Converts pages to images using PyMuPDF

### 2. Smart Batching (for PDFs)
- Groups 3 pages per API call
- Reduces API calls by 3x
- Includes rate limiting (4.5s between batches)

### 3. LLM Extraction
- Sends image(s) to Gemini 1.5 Flash Vision
- Optimized prompt for structured JSON output
- Handles both summary and detail pages

### 4. Response Parsing
- Robust JSON parsing with error recovery
- Salvages partial data from truncated responses
- Deduplicates items across pages

### 5. Post-Processing
- Validates page types (Bill Detail/Pharmacy/Final Bill)
- Removes duplicate entries
- Calculates total item count

## ⚡ Performance

| Document Type | Processing Time | API Calls |
|--------------|-----------------|-----------|
| Single Image | 3-5 seconds | 1 |
| 3-page PDF | 8-12 seconds | 1 |
| 10-page PDF | 20-30 seconds | 4 |
| 15-page PDF | 30-45 seconds | 5 |

## 🎯 Accuracy Optimizations

1. **Precise Prompting**: Instructs LLM to skip headers, totals, and subtotals
2. **Page Type Detection**: Distinguishes summary pages from detailed pages
3. **Deduplication**: Prefers detailed pages over summary when both exist
4. **JSON Recovery**: Multiple strategies to extract data from malformed responses

## 📄 License

MIT License

## 👤 Author

[Your Name]

---

**Built for HackRx Datathon 2025** 🏆
