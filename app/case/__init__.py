"""PDF case input mode — user-supplied PDF with questions and ship parameters.

Distinct from `app/ingestion/` which builds the BKI knowledge base.
This module reads short PDFs the user submits at chat time, containing:
- technical questions about ship structure
- principal particulars (L, B, T, displacement, etc.)
"""
