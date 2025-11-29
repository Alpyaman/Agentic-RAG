"""
Financial Document Ingestion Script

This script ingests financial documents (10-Ks, 10-Qs, earnings transcripts)
into ChromaDB for the Financial Analyst node to query.

Supports: PDF, HTML, DOCX, PPTX, TXT
(SEC EDGAR filings are typically downloaded as HTML)

Usage:
    python src/ingest.py data/TSLA_2023_10K.pdf TSLA 2023
    python src/ingest.py data/TSLA_2023_10K.html TSLA 2023
    python src/ingest.py data/AAPL_Q3_2023.pdf AAPL 2023 --quarter Q3 --type 10-Q

Requirements:
    - LLAMA_CLOUD_API_KEY environment variable (for LlamaParse)
    - GOOGLE_API_KEY environment variable (for embeddings)
"""

import os
import sys
import argparse
from pathlib import Path
from financial_analyst import ingest_financial_document
import traceback

def check_environment():
    """Check if required API keys are set"""
    llama_key = os.getenv("LLAMA_CLOUD_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")

    missing = []
    if not llama_key:
        missing.append("LLAMA_CLOUD_API_KEY")
    if not google_key:
        missing.append("GOOGLE_API_KEY")

    if missing:
        print("Missing required API keys:")
        for key in missing:
            print(f"   • {key}")
        print("\nSet them with:")
        if "LLAMA_CLOUD_API_KEY" in missing:
            print("  export LLAMA_CLOUD_API_KEY='your-llama-key'")
            print("  Get from: https://cloud.llamaindex.ai")
        if "GOOGLE_API_KEY" in missing:
            print("  export GOOGLE_API_KEY='your-google-key'")
            print("  Get from: https://aistudio.google.com/apikey")
        return False

    return True

def validate_file(file_path: str) -> Path:
    """Validate that the file exists and is a supported format"""

    path = Path(file_path)

    if not path.exists():
        print(f"Error: File not found at {file_path}")
        sys.exit(1)

    # LlamaParse supports PDF, HTML, DOCX, PPTX, and more
    supported_formats = ['.pdf', '.html', '.htm', '.docx', '.pptx', '.txt']
    if path.suffix.lower() not in supported_formats:
        print(f"Error: Unsupported file format: {path.suffix}")
        print(f"   Supported formats: {', '.join(supported_formats)}")
        sys.exit(1)

    return path

def main():
    """Main ingestion entry point"""
    parser = argparse.ArgumentParser(
        description="Ingest financial PDFs into the Vector Database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        # Ingest Tesla 10-K for 2023
        python src/ingest.py data/TSLA_2023_10K.pdf TSLA 2023

        # Ingest Apple 10-Q for Q3 2023
        python src/ingest.py data/AAPL_Q3_2023.pdf AAPL 2023 --quarter Q3 --type 10-Q

        # Ingest earnings transcript
        python src/ingest.py data/NVDA_Q4_2023_Earnings.pdf NVDA 2023 --quarter Q4 --type earnings

        Environment Variables:
        LLAMA_CLOUD_API_KEY    - LlamaParse API key (required)
        GOOGLE_API_KEY         - Google AI API key (required)

        Get API keys:
        LlamaParse: https://cloud.llamaindex.ai
        Google AI:  https://aistudio.google.com/apikey
        """
        )

    parser.add_argument(
        "file",
        help="Path to the PDF file (e.g., data/TSLA_10K.pdf)"
    )

    parser.add_argument(
        "ticker",
        help="Stock ticker symbol (e.g., TSLA, AAPL, NVDA)"
    )

    parser.add_argument(
        "year",
        type=int,
        help="Fiscal year (e.g., 2023)"
    )

    parser.add_argument(
        "--quarter",
        "-q",
        help="Quarter for 10-Q reports (e.g., Q1, Q2, Q3, Q4)",
        default=None
    )

    parser.add_argument(
        "--type",
        "-t",
        help="Document type (default: 10-K)",
        choices=["10-K", "10-Q", "earnings", "8-K", "proxy"],
        default="10-K"
    )
 
    args = parser.parse_args()

    # Banner
    print("\n" + "=" * 80)
    print("Financial Document Ingestion")
    print("=" * 80)

    # Check environment
    if not check_environment():
        sys.exit(1)

    # Validate file
    file_path = validate_file(args.file)

    # Display ingestion plan
    print("\nIngestion Plan:")
    print(f"   File:         {file_path}")
    print(f"   Ticker:       {args.ticker}")
    print(f"   Year:         {args.year}")
    if args.quarter:
        print(f"   Quarter:      {args.quarter}")
    print(f"   Type:         {args.type}")
    print("   Target DB:    ./chroma_db (default)")

    # Confirm
    confirm = input("\nProceed with ingestion? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Cancelled.")
        sys.exit(0)

    # Run ingestion
    print("\n" + "=" * 80)
    print("Starting Ingestion...")
    print("=" * 80 + "\n")

    try:
        num_chunks = ingest_financial_document(
            file_path=str(file_path),
            ticker=args.ticker,
            year=args.year,
            quarter=args.quarter,
            doc_type=args.type
        )

        # Success
        print("\n" + "=" * 80)
        print("Ingestion Complete!")
        print("=" * 80)

        print("\nResults:")
        print(f"   Chunks created:  {num_chunks}")
        print("    Vector DB:       ./chroma_db")
        print(f"   Ticker:          {args.ticker}")
        print(f"   Year:            {args.year}")

        print("\nNext Steps:")
        print("   1. Test retrieval:")
        print("      python src/test_financial_analyst.py")
        print("   2. Run full analysis:")
        print("      python src/main.py '{args.ticker.upper()}' {args.ticker}")
        print("   3. Ingest more documents:")
        print("      python src/ingest.py <another-file.pdf> <ticker> <year>")

        print("=" * 80 + "\n")

    except Exception as e:
        print("\n" + "=" * 80)
        print("Ingestion Failed!")
        print("=" * 80)
        print(f"\nError: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()