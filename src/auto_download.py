"""
Automated SEC EDGAR Filing downloader.

This script automatically downloads 10-K and 10-Q filings from SEC EDGAR and optionally
ingests them into the vector database.

Features:
- Downloads filings for any ticker and year from SEC EDGAR
- Respects SEC rate limits (10 requested/seconds)
- Converts HTML filing to a format suitable for LlamaParse
- Integrates with the ingestion pipeline
- Batch download support via CSV manifest

Usage:
    # Download single filing
    python src/auto_download.py TSLA 2023 --filing-type 10-K

    # Download and auto-ingest
    python src/auto_download.py TSLA 2023 --filing-type 10-K --ingest

    # Batch download from manifest
    python src/auto_download.py --manifest data/manifest.csv --download

    # Download multiple years
    python src/auto_download.py TSLA --years 2021 2022 2023 --filing-type 10-K

Requirements:
    pip install sec-edgar-downloader
"""

import argparse
import sys
import csv
from pathlib import Path
from typing import List, Optional
import time
import traceback
import shutil
from datetime import datetime
from sec_edgar_downloader import Downloader
from .ingest import ingest_filing


    
def download_filing(ticker: str, filing_type: str = "10-K", after_date: Optional[str] = None,
                    before_date: Optional[str] = None, download_details: bool = True,
                    company_name: Optional[str] = None) -> List[Path]:
    """
    Download SEC filings for a given ticker.

    Args:
        ticker: Stock ticker symbol (e.g., 'TSLA')
        filing_type: Type of filing ('10-K', '10-Q', '8-K', etc.)
        after_date: Download filings after this date (format: 'YYYY-MM-DD')
        before_date: Download filings before this date (format: 'YYYY-MM-DD')
        download_details: Whether to download full filing details
        company_name: Optional company name for display

    Returns:
        List of paths to downloaded filing directories
    """

    # Initialize downloader
    # You should set a User-Agent with your contact info (SEC requirement)
    # Format: "YourName/Version (email@example.com)"
    dl = Downloader(company_name or "AgenticRAG", "info@example.com")

    print(f"\n{'='*80}")
    print(f"Downloading {filing_type} filings for {ticker}")
    if company_name:
        print(f"   Company: {company_name}")
    print(f"{'='*80}\n")

    # Download the filings
    # They will be saved to: ./sec-edgar-filings/{ticker}/{filing_type}/
    try:
        num_downloaded = dl.get(filing_type, ticker, after=after_date, before=before_date,
            download_details=download_details)
        
        print(f"\nDownloaded {num_downloaded} {filing_type} filing(s)")

        # Find the downloaded files
        base_path = Path("sec-edgar-filings") / ticker / filing_type

        if base_path.exists():
            filing_dirs = [d for d in base_path.iterdir() if d.is_dir()]
            print(f"Location: {base_path}")
            return filing_dirs
        else:
            print(f"Warning: Expected path {base_path} not found")
            return []
        
    except Exception as e:
        print(f"Download failed: {str(e)}")
        traceback.print_exc()
        return []
    
def convert_filing_to_ingestible_format(filing_dir: Path, ticker: str,
                                        output_dir: Path = Path("data")) -> Optional[Path]:
    """
    Convert downloaded SEC filing to a format ready for ingestion.

    SEC filings are downloaded as HTML. We'll extract the main filing and save it in a
    format LlamaParse can handle.

    Args:
        filing_dir: Path to the filing directory
        ticker: Stock ticker
        output_dir: Where to save the processed file

    Returns:
        Path to the converted file, or None if conversion failed
    """
    try:
        # Find the primary document (usually full-submission.txt or primary-document.html)
        primary_doc = None

        # Look for common filing document names
        candidates = [
            filing_dir / "primary-document.html",
            filing_dir / "full-submission.txt",
        ]

        # Also check for numbered documents (e.g., 0000000001.html)
        for file in filing_dir.glob("*.html"):
            if file.name not in ["index.json", "index.html"]:
                candidates.append(file)

        for candidate in candidates:
            if candidate.exists():
                primary_doc = candidate
                break

        if not primary_doc:
            print(f"Warning: Could not find primary document in {filing_dir}")
            return None

        # Extract filing metadata from directory structure
        # Path format: sec-edgar-filings/{ticker}/{filing_type}/{accession_number}/
        parts = filing_dir.parts
        filing_type = parts[-2] if len(parts) >= 2 else "unknown"
        accession = parts[-1] if len(parts) >= 1 else "unknown"

        # Try to extract year from filing directory name or content
        # Accession numbers contain the year: 0001318605-24-000004 -> 2024
        year = None
        if len(accession) > 10:
            year_str = accession.split('-')[1]
            if year_str.isdigit():
                year = 2000 + int(year_str) if len(year_str) == 2 else int(year_str)

        if not year:
            # Fallback: use current year
            year = datetime.now().year

        # Create output filename
        # Format: {TICKER}_{YEAR}_{TYPE}.html
        filing_type_clean = filing_type.replace('-', '')  # 10-K -> 10K
        output_filename = f"{ticker.upper()}_{year}_{filing_type_clean}.html"
        output_path = output_dir / output_filename

        # Copy the file
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(primary_doc, output_path)

        print(f"Converted: {output_path.name}")

        return output_path

    except Exception as e:
        print(f"Conversion failed: {str(e)}")
        traceback.print_exc()
        return None

def download_and_prepare(ticker: str, years: List[int], filing_type: str = "10-K",
                         output_dir: Path = Path("data"), auto_ingest: bool = False) -> List[Path]:
    """
    Download filings and prepare them for ingestion.

    Args:
        ticker: Stock ticker
        years: List of years to download
        filing_type: Type of filing
        output_dir: Where to save prepared files
        auto_ingest: Whether to automatically ingest after download

    Returns:
        List of paths to prepared files
    """

    prepared_files = []

    for year in years:
        # Set date range for the fiscal year
        after_date = f"{year}-01-01"
        before_date = f"{year}-12-31"

        print(f"\n📅 Year: {year}")

        # Download the filing

        filing_dirs = download_filing(ticker=ticker, filing_type=filing_type,
                                      after_date=after_date, before_date=before_date)

        if not filing_dirs:
            print(f"No {filing_type} found for {ticker} in {year}")
            continue

        # Convert each filing
        for filing_dir in filing_dirs:
            prepared_file = convert_filing_to_ingestible_format(filing_dir, ticker, output_dir)
            if prepared_file:
                prepared_files.append(prepared_file)

                # Auto-ingest if requested
                if auto_ingest:
                    print("Auto-ingesting...")
                    try:
                        num_chunks = ingest_filing(str(prepared_file), ticker, year,
                                                   doc_type=filing_type)
                        print(f"Ingested {num_chunks} chunks")
                    except Exception as e:
                        print(f"Ingestion failed: {str(e)}")

        # Respect SEC rate limits (max 10 requests/second)
        time.sleep(0.2)

    return prepared_files

def batch_download_from_manifest(manifest_path: Path, auto_ingest: bool = False):
    """
    Download filings for all tickers in a manifest CSV.

    Expected CSV format:
    ticker,year,filing_type
    TSLA,2023,10-K
    AAPL,2023,10-K
    """

    print(f"\n{'='*80}")
    print("Batch Download from Manifest")
    print(f"{'='*80}")
    print(f"Manifest: {manifest_path}\n")

    if not manifest_path.exists():
        print(f"Error: Manifest file not found: {manifest_path}")
        return

    with open(manifest_path, 'r') as f:
        reader = csv.DictReader(f)

        for row in reader:
            ticker = row['ticker']
            year = int(row['year'])
            filing_type = row.get('filing_type', '10-K')

            download_and_prepare(ticker=ticker, years=[year], filing_type=filing_type,
                                 auto_ingest=auto_ingest)

def main():
    """Main entry point for the auto-downloader."""
    parser = argparse.ArgumentParser(
        description="Automatically download SEC filings from EDGAR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download Tesla's 2023 10-K
  python src/auto_download.py TSLA --years 2023 --filing-type 10-K

  # Download and auto-ingest
  python src/auto_download.py TSLA --years 2023 --filing-type 10-K --ingest

  # Download multiple years
  python src/auto_download.py AAPL --years 2021 2022 2023 --filing-type 10-K

  # Batch download from manifest
  python src/auto_download.py --manifest data/manifest.csv --download
        """
    )

    parser.add_argument("ticker", nargs="?", help="Stock ticker (e.g., TSLA)")
    parser.add_argument("--years", nargs="+", type=int, help="Years to download (e.g., 2021 2022 2023)")
    parser.add_argument("--filing-type", "-t", default="10-K",
                       choices=["10-K", "10-Q", "8-K", "DEF 14A"],
                       help="Type of filing to download")
    parser.add_argument("--output-dir", "-o", default="data",
                       help="Output directory for downloaded files")
    parser.add_argument("--ingest", action="store_true",
                       help="Automatically ingest files after downloading")
    parser.add_argument("--manifest", "-m", help="Path to CSV manifest for batch download")

    args = parser.parse_args()

    # Batch download from manifest
    if args.manifest:
        batch_download_from_manifest(Path(args.manifest), auto_ingest=args.ingest)
        return

    # Single ticker download
    if not args.ticker:
        parser.print_help()
        print("\nError: Either provide a ticker or use --manifest for batch download")
        sys.exit(1)

    if not args.years:
        print("Error: Please specify --years")
        sys.exit(1)

    prepared_files = download_and_prepare(ticker=args.ticker, years=args.years,
                                          filing_type=args.filing_type,
                                          output_dir=Path(args.output_dir),
                                          auto_ingest=args.ingest)

    print(f"\n{'='*80}")
    print("Download Complete!")
    print(f"{'='*80}")
    print(f"Downloaded and prepared {len(prepared_files)} filing(s)")
    print(f"Location: {args.output_dir}/")

    if not args.ingest:
        print("\nTo ingest these files into the vector database, run:")
        for file in prepared_files:
            print(f"  python src/ingest.py {file} {args.ticker} <year>")

if __name__ == "__main__":
    main()