# Medical Report Explainer

An LLM-powered patient empowerment application that translates complex diagnostic imaging reports, pathology results, and laboratory summaries into plain, empathetic, and easily understandable language.

---

## Overview

Medical reports are filled with dense clinical jargon that causes patient anxiety and confusion. **Medical Report Explainer** parses technical diagnostic documents (e.g., radiology reports, blood work panels) and generates clear, structured layman explanations, vocabulary breakdowns, and recommended follow-up questions for patients to ask their physicians.

## Key Features

- **Jargon Translation Engine**: Converts complex clinical terminology into accessible reading levels without losing clinical meaning.
- **Structured Explanations**: Produces a structured summary: Key Findings, What This Means, Glossary of Terms, Questions for Your Doctor.
- **Safety & Disclaimer Guardrails**: Enforces clear non-diagnostic disclaimers and safety guidance encouraging physician consultation.

## Tech Stack

- **LLM Pipeline**: Python / LangChain / OpenAI API (GPT-4 / GPT-3.5) / Anthropic Claude
- **Document Ingestion**: PyPDF2 / pdfplumber / Tesseract OCR
- **Web Interface**: Streamlit / React frontend

## Installation & Setup

```bash
git clone https://github.com/junxiant/Med-Report-Explainer.git
cd Med-Report-Explainer

pip install -r requirements.txt

# Set up API Key
export OPENAI_API_KEY="your-api-key"

# Run application
python app.py
```

Author Jun Tan

Linkedin https://www.linkedin.com/in/junxiant/
