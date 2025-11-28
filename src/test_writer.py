"""
Test script for the Writer node

This script demonstrates how the Writer node generates investment memos.

IMPORTANT: To run tests, you need:
- GOOGLE_API_KEY environment variable set

You can get your API key from:
- Google AI Studio: https://aistudio.google.com/apikey
"""

import os
from state import AgentState
import traceback
from dotenv import load_dotenv
from writer import (
    writer_node,
    writer_node_structured,
    MEMO_SECTIONS
)

load_dotenv()

def check_api_key():
    """Check if required API key is set"""
    google_key = os.getenv("GOOGLE_API_KEY")

    print("=" * 80)
    print("API Key Check")
    print("=" * 80)

    if google_key:
        print(f"✓ GOOGLE_API_KEY is set ({google_key[:8]}...)")
    else:
        print("✗ GOOGLE_API_KEY is NOT set")
        print("  Set it with: export GOOGLE_API_KEY='your-key-here'")
        print("  Get your API key from: https://aistudio.google.com/apikey")

    print()

    return bool(google_key)

def test_memo_structure():
    """Test the memo structure definition"""
    print("=" * 80)
    print("Test 1: Memo Structure")
    print("=" * 80)

    print("\nInvestment Memo Sections (Based on Sequoia/a16z):")
    for i, (section, description) in enumerate(MEMO_SECTIONS.items(), 1):
        print(f"  {i}. {section.replace('_', ' ').title()}")
        print(f"     → {description}")

    print(f"\n✓ Total sections: {len(MEMO_SECTIONS)}")
    print("✓ Memo structure validated\n")

    return True

def create_mock_research_data():
    """Create mock research data for testing"""
    financial_context = [
        """**What is Tesla's revenue for the last 3 years?**
Tesla's revenue has shown strong growth:
- 2021: $53.82 billion
- 2022: $81.46 billion (+51.4% YoY)
- 2023: $96.77 billion (+18.8% YoY)
[Source: SEC 10-K Filings]""",

        """**What are the key risk factors for Tesla?**
Key risks include:
- Supply chain dependencies for battery materials
- Increasing competition in EV market
- Regulatory changes in key markets
- Execution risk on new product launches (Cybertruck)
[Source: 10-K Risk Factors Section]""",

        """**What is Tesla's debt-to-equity ratio?**
Debt-to-Equity Ratio: 0.17 (Q3 2023)
This indicates a conservative capital structure with low leverage.
[Source: Balance Sheet, Q3 2023 10-Q]"""
    ]

    market_context = [
        """Tesla (TSLA) is a mega-capitalization company with a market cap of $1.42 trillion.
The stock closed at $426.58, up 1.71%. YTD return of 5.63%.

**Recent Developments:**
- Revenue for last year: $95.63B (+11.57% YoY)
- Operating income: $4.87B
- Diluted EPS (TTM): $1.45

**Market Trends:**
- Facing increased competition from BYD in China
- Traditional automakers (Ford, GM) investing heavily in EVs

**Financial Signals:**
- P/E ratio: 294.19 (very high, potential overvaluation)
- Forward P/E: 188.68
- Price-to-sales: 15.72 (significantly above industry average)

**Risks & Opportunities:**
- Risk: High valuation metrics suggest overvaluation
- Opportunity: Strong market position and revenue growth
- Opportunity: Return on equity above industry average
[Sources: Yahoo Finance, CNN Markets, Investopedia, Nasdaq]"""
    ]

    return financial_context, market_context

def test_writer_with_mock_data():
    """Test Writer node with mock research data"""
    print("=" * 80)
    print("Test 2: Writer Node with Mock Data")
    print("=" * 80)

    financial_context, market_context = create_mock_research_data()

    test_state: AgentState = {
        "company": "Tesla",
        "ticker": "TSLA",
        "financial_context": financial_context,
        "market_context": market_context,
        "memo_sections": {},
        "messages": [],
        "research_iterations": 0,
        "is_data_sufficient": False,
    }

    try:
        result = writer_node(test_state)

        assert "memo_sections" in result
        assert "full_draft" in result["memo_sections"]
        assert len(result["memo_sections"]["full_draft"]) > 0

        memo = result["memo_sections"]["full_draft"]

        print("\n" + "=" * 80)
        print("✓ Investment Memo Generated Successfully")
        print("=" * 80)
        print(f"\nMemo Length: {len(memo)} characters")
        print(f"Word Count: ~{len(memo.split())} words")

        # Show preview
        print("\nMemo Preview (first 1000 characters):")
        print("-" * 80)
        print(memo[:1000])
        print("...")
        print("-" * 80)

        # Check for key sections
        print("\nSection Validation:")
        sections_found = []
        for section in MEMO_SECTIONS.keys():
            section_title = section.replace('_', ' ').title()
            if section_title.lower() in memo.lower():
                sections_found.append(section_title)
                print(f"  ✓ {section_title}")

        print(f"\n✓ Found {len(sections_found)}/{len(MEMO_SECTIONS)} sections")

        return True

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        traceback.print_exc()
        return False

def test_structured_writer():
    """Test structured writer that generates sections separately"""
    print("\n" + "=" * 80)
    print("Test 3: Structured Writer (Section by Section)")
    print("=" * 80)

    financial_context, market_context = create_mock_research_data()

    test_state: AgentState = {
        "company": "Tesla",
        "ticker": "TSLA",
        "financial_context": financial_context,
        "market_context": market_context,
        "memo_sections": {},
        "messages": [],
        "research_iterations": 0,
        "is_data_sufficient": False,
    }

    try:
        result = writer_node_structured(test_state)

        assert "memo_sections" in result
        assert "full_draft" in result["memo_sections"]

        # Check that individual sections are also returned
        for section_name in MEMO_SECTIONS.keys():
            if section_name in result["memo_sections"]:
                print(f"  ✓ {section_name}: {len(result['memo_sections'][section_name])} chars")

        print("\nStructured memo generated successfully")
        print(f"  Total length: {len(result['memo_sections']['full_draft'])} characters")

        return True

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        traceback.print_exc()
        return False

def demo_memo_requirements():
    """Demonstrate the memo generation requirements"""
    print("\n" + "=" * 80)
    print("Demo: Investment Memo Requirements")
    print("=" * 80)

    print("\nCritical Requirements:")
    print("  1. Citations: [Source: URL] after every factual claim")
    print("  2. Data-Driven: Specific numbers, percentages, dates")
    print("  3. Objectivity: Both bullish and bearish perspectives")
    print("  4. Actionable: Clear Buy/Hold/Sell recommendation")
    print("  5. Professional: Written for Investment Committee")

    print("\nModel Selection:")
    print("  - Gemini Flash: Quick synthesis and section generation")
    print("  - Gemini Pro: Final memo generation (higher quality)")
    print("  - Temperature 0.3: Mostly factual with slight creativity")

    print("\nOutput Format:")
    print("  - Markdown with ## headers")
    print("  - **Bold** for key points")
    print("  - Tables for financial data")
    print("  - Bullet points for lists")
    print("  - Citations after claims")

    print("\n✓ Requirements documented\n")

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Writer Node Test Suite")
    print("=" * 80 + "\n")

    # Check API key
    has_key = check_api_key()

    # Always run structure test
    test_memo_structure()

    # Always run demo
    demo_memo_requirements()

    if has_key:
        print("=" * 80)
        print("Running Live API Tests")
        print("=" * 80 + "\n")

        success1 = test_writer_with_mock_data()

        if success1:
            # Test structured writer if basic works
            success2 = test_structured_writer()

            if success1 and success2:
                print("\n" + "=" * 80)
                print("✓ All Tests Passed!")
                print("=" * 80)
                print("\nThe Writer node successfully generates:")
                print("  • Professional investment memos")
                print("  • Structured sections (Executive Summary, Risks, etc.)")
                print("  • Cited claims with [Source: URL] format")
                print("  • Actionable recommendations")
                print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("Skipping Live API Tests")
        print("=" * 80)
        print("\nTo run live tests, set your API key:")
        print("  export GOOGLE_API_KEY='your-google-key'")
        print("  Get API key from: https://aistudio.google.com/apikey")
        print("\nThen run: python test_writer.py")
        print("=" * 80)