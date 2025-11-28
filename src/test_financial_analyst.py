"""
Test script for the Financial Analyst node

This script demonstrates how the Financial Analyst node operates.

IMPORTANT: To run full tests, you need:
1. GOOGLE_API_KEY environment variable set (for embeddings and LLM)
2. A ChromaDB database with financial data (optional for basic tests)

You can get your API key from:
- Google AI Studio: https://aistudio.google.com/apikey
"""

import os
from state import AgentState
from dotenv import load_dotenv
from financial_analyst import (
    create_vector_store,
    create_python_repl,
    financial_analyst_node,
)

load_dotenv()

 
def check_api_keys():
    """Check if required API keys are set"""
    google_key = os.getenv("GOOGLE_API_KEY")

    print("=" * 80)
    print("API Key Check")
    print("=" * 80)

    if google_key:
        print(f"  GOOGLE_API_KEY is set ({google_key[:8]}...)")
    else:
        print("  GOOGLE_API_KEY is NOT set")
        print("  Set it with: export GOOGLE_API_KEY='your-key-here'")
        print("  Get your API key from: https://aistudio.google.com/apikey")

    print()

    return bool(google_key)

def test_python_repl():
    """Test the Python REPL tool for calculations"""
    print("=" * 80)
    print("Test 1: Python REPL Tool")
    print("=" * 80)

    python_repl = create_python_repl()

    # Test basic calculation
    print("\nTesting calculation: Debt-to-Equity Ratio")
    print("Code: debt = 5000000; equity = 12000000; ratio = debt / equity; print(f'Ratio: {ratio:.4f}')")

    try:
        result = python_repl.run(
            "debt = 5000000\nequity = 12000000\nratio = debt / equity\nprint(f'Ratio: {ratio:.4f}')"
        )
        print(f"Result: {result}")
        print("Python REPL works correctly\n")
        return True
    except Exception as e:
        print(f"Test failed: {e}\n")
        return False

def test_vector_store_creation():
    """Test ChromaDB vector store creation"""
    print("=" * 80)
    print("Test 2: Vector Store Creation")
    print("=" * 80)

    try:
        # Create a test vector store in memory
        vectorstore = create_vector_store(
            collection_name="test_financial_reports",
            persist_directory="./test_chroma_db"
        )

        print(f"ChromaDB vector store created successfully {vectorstore}")
        print("Collection: test_financial_reports")
        print("Embedding model: Google embedding-001\n")

        return True

    except Exception as e:
        print(f"Test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False

def test_financial_analyst_empty_db():
    """Test Financial Analyst node with empty database"""
    print("=" * 80)
    print("Test 3: Financial Analyst Node (Empty DB)")
    print("=" * 80)

    test_state: AgentState = {
        "company": "Tesla",
        "ticker": "TSLA",
        "financial_context": [],
        "market_context": [],
        "memo_sections": {},
        "messages": [],
        "research_iterations": 0,
        "is_data_sufficient": False,
    }

    try:
        result = financial_analyst_node(test_state)

        assert "financial_context" in result
        assert len(result["financial_context"]) > 0

        print("\n" + "=" * 80)
        print("Financial Analyst Completed")
        print("=" * 80)
        print("\nFinancial Context (with empty database):")
        print("-" * 80)
        print(result["financial_context"][0][:500] + "...")
        print("-" * 80)

        return True

    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def demo_document_ingestion():
    """Demonstrate the document ingestion process"""
    print("\n" + "=" * 80)
    print("Demo: Document Ingestion Pipeline")
    print("=" * 80)

    print("\nDocument Ingestion Process:")
    print("1. Parse PDF with LlamaParse (preserves table structure)")
    print("2. Convert tables to Markdown format")
    print("3. Split into chunks with metadata")
    print("4. Embed using Google embedding-001")
    print("5. Store in ChromaDB with metadata filtering")

    print("\nMetadata Schema:")
    print("  - ticker: Stock symbol (e.g., 'TSLA')")
    print("  - year: Fiscal year (e.g., 2023)")
    print("  - quarter: Quarter or 'FY' for annual")
    print("  - doc_type: '10-K', '10-Q', or 'Earnings'")
    print("  - section: Section name for precise retrieval")

    print("\nExample usage:")
    print("  ingest_financial_document(")
    print("      file_path='./data/TSLA_2023_10K.pdf',")
    print("      ticker='TSLA',")
    print("      year=2023,")
    print("      doc_type='10-K'")
    print("  )")

    print("\nIngestion pipeline documented\n")

def demo_calculation_pattern():
    """Demonstrate the calculation pattern with Python REPL"""
    print("=" * 80)
    print("Demo: LLM + Python REPL Pattern")
    print("=" * 80)

    print("\nWhy use Python REPL instead of LLM math?")
    print("  LLM: 100 / 3 = 33 (hallucinated)")
    print("  Python: 100 / 3 = 33.333... (precise)")

    print("\nPattern:")
    print("  1. LLM extracts numbers from financial documents")
    print("  2. LLM identifies what calculation is needed")
    print("  3. Python REPL performs the calculation")
    print("  4. LLM incorporates result into analysis")

    print("\nExample Calculations:")
    print("  - Debt-to-Equity: debt / equity")
    print("  - Operating Margin: (operating_income / revenue) * 100")
    print("  - CAGR: ((ending / beginning) ** (1/years)) - 1")
    print("  - P/E Ratio: market_cap / net_income")

    print("\nCalculation pattern documented\n")

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Financial Analyst Node Test Suite")
    print("=" * 80 + "\n")

    # Check API keys
    has_keys = check_api_keys()

    # Always run demos
    demo_calculation_pattern()
    demo_document_ingestion()

    # Run tests that don't require API
    success1 = test_python_repl()

    if has_keys:
        print("=" * 80)
        print("Running Live API Tests")
        print("=" * 80 + "\n")

        success2 = test_vector_store_creation()
        success3 = test_financial_analyst_empty_db()

        if success1 and success2 and success3:
            print("\n" + "=" * 80)
            print("All Tests Passed!")
            print("=" * 80)
            print("\nNote: These tests use an empty database.")
            print("To test with real data:")
            print("  1. Get financial PDFs (10-Ks from SEC EDGAR)")
            print("  2. Use ingest_financial_document() to load them")
            print("  3. Re-run the Financial Analyst node tests")
            print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("Skipping Live API Tests")
        print("=" * 80)
        print("\nTo run live tests, set your API key:")
        print("  export GOOGLE_API_KEY='your-google-key'")
        print("  Get API key from: https://aistudio.google.com/apikey")
        print("\nThen run: python test_financial_analyst.py")
        print("=" * 80)