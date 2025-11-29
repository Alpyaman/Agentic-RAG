"""
Batch Financial Document Ingestion

This script ingests multiple financial documents from a directory or manifest file.
Useful for building a comprehensive financial document database.

Usage:
    # Ingest all PDFs in a directory
    python src/batch_ingest.py data/financial_reports/

    # Ingest from a manifest file
    python src/batch_ingest.py --manifest data/manifest.csv

Manifest CSV format:
    file_path,ticker,year,quarter,doc_type
    data/TSLA_2023_10K.pdf,TSLA,2023,,10-K
    data/AAPL_Q3_2023.pdf,AAPL,2023,Q3,10-Q
    data/NVDA_2023_10K.pdf,NVDA,2023,,10-K
"""

import os
import sys
import csv
import argparse
from pathlib import Path
from typing import List, Dict
from financial_analyst import ingest_financial_document

def parse_manifest(manifest_path: str) -> List[Dict]:
    """Parse a CSV manifest file"""
    documents = []

    with open(manifest_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Validate required fields
            if not all(k in row for k in ['file_path', 'ticker', 'year']):
                print(f"Skipping invalid row: {row}")
                continue

            doc = {
                'file_path': row['file_path'],
                'ticker': row['ticker'],
                'year': int(row['year']),
                'quarter': row.get('quarter') or None,
                'doc_type': row.get('doc_type', '10-K')
            }
            documents.append(doc)

    return documents

def discover_pdfs(directory: str) -> List[Dict]:
    """
    Auto-discover PDFs in a directory.

    Attempts to parse metadata from filenames.
    Expected format: {TICKER}_{YEAR}_{TYPE}.pdf
    Example: TSLA_2023_10K.pdf, AAPL_Q3_2023_10Q.pdf
    """
    documents = []
    path = Path(directory)

    for pdf_file in path.glob("**/*.pdf"):
        # Try to parse filename
        stem = pdf_file.stem  # e.g., "TSLA_2023_10K"
        parts = stem.split('_')

        if len(parts) >= 3:
            ticker = parts[0]
            year_str = parts[1]
            doc_type_str = parts[2]

            # Parse year
            try:
                year = int(year_str)
            except ValueError:
                # Maybe format is TSLA_Q3_2023_10Q
                if len(parts) >= 4 and parts[1].startswith('Q'):
                    quarter = parts[1]
                    year = int(parts[2])
                    doc_type = parts[3].replace('Q', '-Q').replace('K', '-K')
                else:
                    print(f"Cannot parse year from: {pdf_file.name}")
                    continue
            else:
                quarter = None
                doc_type = doc_type_str.replace('Q', '-Q').replace('K', '-K')

            doc = {
                'file_path': str(pdf_file),
                'ticker': ticker,
                'year': year,
                'quarter': quarter,
                'doc_type': doc_type
            }
            documents.append(doc)
        else:
            print(f"  Skipping (unknown format): {pdf_file.name}")
            print("    Expected: TICKER_YEAR_TYPE.pdf (e.g., TSLA_2023_10K.pdf)")

    return documents

def ingest_batch(documents: List[Dict], persist_dir: str):
    """Ingest a batch of documents"""
    total = len(documents)
    success = 0
    failed = 0

    print(f"\nBatch Ingestion: {total} documents")
    print("=" * 80)

    for i, doc in enumerate(documents, 1):
        print(f"\n[{i}/{total}] Processing: {Path(doc['file_path']).name}")
        print(f"   Ticker: {doc['ticker']}, Year: {doc['year']}, Type: {doc['doc_type']}")

        try:
            num_chunks = ingest_financial_document(
                file_path=doc['file_path'],
                ticker=doc['ticker'],
                year=doc['year'],
                quarter=doc['quarter'],
                doc_type=doc['doc_type'],
                persist_directory=persist_dir
            )

            print(f"   ✓ Success: {num_chunks} chunks")
            success += 1

        except Exception as e:
            print(f"   ✗ Failed: {str(e)}")
            failed += 1

    # Summary
    print("\n" + "=" * 80)
    print("Batch Ingestion Summary")
    print("=" * 80)
    print(f"   Total:    {total}")
    print(f"   Success:  {success}")
    print(f"   Failed:   {failed}")
    print(f"   Database: {persist_dir}")
    print("=" * 80 + "\n")

def main():
    """Main batch ingestion entry point"""
    parser = argparse.ArgumentParser(
        description="Batch ingest financial documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        # Auto-discover PDFs in directory
        python src/batch_ingest.py data/financial_reports/

        # Use manifest file
            python src/batch_ingest.py --manifest data/manifest.csv

        Manifest CSV Format:
        file_path,ticker,year,quarter,doc_type
        data/TSLA_2023_10K.pdf,TSLA,2023,,10-K
        data/AAPL_Q3_2023.pdf,AAPL,2023,Q3,10-Q

        Filename Auto-Discovery:
        Expected format: TICKER_YEAR_TYPE.pdf
        Examples:
            - TSLA_2023_10K.pdf → TSLA, 2023, 10-K
            - AAPL_Q3_2023_10Q.pdf → AAPL, 2023 Q3, 10-Q
        """
        )

    parser.add_argument(
        "directory",
        nargs="?",
        help="Directory containing PDF files"
    )

    parser.add_argument(
        "--manifest",
        "-m",
        help="CSV manifest file with document metadata"
    )

    parser.add_argument(
        "--persist-dir",
        help="ChromaDB persist directory (default: ./chroma_db)",
        default="./chroma_db"
    )

    args = parser.parse_args()

    # Check arguments
    if not args.directory and not args.manifest:
        parser.print_help()
        sys.exit(1)

    # Banner
    print("\n" + "=" * 80)
    print("Batch Financial Document Ingestion")
    print("=" * 80)

    # Check environment
    llama_key = os.getenv("LLAMA_CLOUD_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")

    if not llama_key or not google_key:
        print("\nMissing API keys")
        if not llama_key:
            print("   LLAMA_CLOUD_API_KEY not set")
        if not google_key:
            print("   GOOGLE_API_KEY not set")
        sys.exit(1)

    # Discover documents
    if args.manifest:
        print(f"\nLoading manifest: {args.manifest}")
        documents = parse_manifest(args.manifest)
    else:
        print(f"\nAuto-discovering PDFs in: {args.directory}")
        documents = discover_pdfs(args.directory)

    if not documents:
        print("\nNo documents found!")
        sys.exit(1)

    print(f"\n✓ Found {len(documents)} documents")

    # Show preview
    print("\nDocuments to ingest:")
    for i, doc in enumerate(documents[:5], 1):
        filename = Path(doc['file_path']).name
        print(f"   {i}. {filename} ({doc['ticker']}, {doc['year']}, {doc['doc_type']})")

    if len(documents) > 5:
        print(f"   ... and {len(documents) - 5} more")

    # Confirm
    confirm = input("\nProceed with batch ingestion? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Cancelled.")
        sys.exit(0)

    # Ingest
    ingest_batch(documents, args.persist_dir)

if __name__ == "__main__":
    main()