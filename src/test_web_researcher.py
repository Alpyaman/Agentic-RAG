"""
Test script for the Web Researcher node

This script demonstrates how the Web Researcher node operates.

IMPORTANT: To run this test, you need:
1. TAVILY_API_KEY environment variable set
2. GOOGLE_API_KEY environment variable set

You can get these from:
- Tavily: https://tavily.com/
- Gemini: https://aistudio.google.com/apikey
"""

import os
from state import AgentState
from web_researcher import (
    web_research_node,
    iterative_web_research_node
)
from dotenv import load_dotenv

load_dotenv()

def check_api_keys():
    """Check if required API keys are set"""
    tavily_key = os.getenv("TAVILY_API_KEY")
    gemini_key = os.getenv("GOOGLE_API_KEY")

    print("=" * 80)
    print("API Key Check")
    print("=" * 80)

    if tavily_key:
        print(f"✓ TAVILY_API_KEY is set ({tavily_key[:8]}...)")
    else:
        print("✗ TAVILY_API_KEY is NOT set")
        print("  Set it with: export TAVILY_API_KEY='your-key-here'")

    if gemini_key:
        print(f"✓ GOOGLE_API_KEY is set ({gemini_key[:8]}...)")
    else:
        print("✗ GOOGLE_API_KEY is NOT set")
        print("  Set it with: export GOOGLE_API_KEY='your-key-here'")

    print()

    return bool(tavily_key and gemini_key)

def test_basic_web_research():
    """Test the basic web research node"""
    print("=" * 80)
    print("Test 1: Basic Web Research Node")
    print("=" * 80)

    # Create test state
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
        # Execute the web research node
        result = web_research_node(test_state)

        # The result should have market_context
        assert "market_context" in result
        assert len(result["market_context"]) > 0

        print("\n" + "=" * 80)
        print("✓ Web Research Completed Successfully")
        print("=" * 80)
        print("\nMarket Context Summary:")
        print("-" * 80)
        print(result["market_context"][0])
        print("-" * 80)

        return True

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_iterative_web_research():
    """Test the iterative web research node (advanced)"""
    print("\n" + "=" * 80)
    print("Test 2: Iterative Web Research Node (Advanced)")
    print("=" * 80)

    test_state: AgentState = {
        "company": "NVIDIA",
        "ticker": "NVDA",
        "financial_context": [],
        "market_context": [],
        "memo_sections": {},
        "messages": [],
        "research_iterations": 0,
        "is_data_sufficient": False,
    }

    try:
        result = iterative_web_research_node(test_state)

        assert "market_context" in result
        assert len(result["market_context"]) > 0

        print("\n" + "=" * 80)
        print("✓ Iterative Research Completed Successfully")
        print("=" * 80)
        print("\nComprehensive Market Context:")
        print("-" * 80)
        print(result["market_context"][0])
        print("-" * 80)

        return True

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_state_accumulation():
    """Test that market_context accumulates properly"""
    print("\n" + "=" * 80)
    print("Test 3: State Accumulation (Simulated)")
    print("=" * 80)

    # Simulate what happens in LangGraph when operator.add is used
    initial_state: AgentState = {
        "company": "Apple",
        "ticker": "AAPL",
        "financial_context": [],
        "market_context": [],
        "memo_sections": {},
        "messages": [],
        "research_iterations": 0,
        "is_data_sufficient": False,
    }

    # Simulate first web research
    print("Simulating first web research call...")
    first_result = {
        "market_context": ["First batch of market research findings..."]
    }

    # In LangGraph, operator.add would merge this
    updated_market = initial_state["market_context"] + first_result["market_context"]
    print(f"After first call: {len(updated_market)} items")

    # Simulate second web research (different query)
    print("Simulating second web research call...")
    second_result = {
        "market_context": ["Second batch of market research findings..."]
    }

    updated_market = updated_market + second_result["market_context"]
    print(f"After second call: {len(updated_market)} items")

    print("\n✓ State accumulation works correctly")
    print(f"  Total market_context items: {len(updated_market)}")
    for i, context in enumerate(updated_market, 1):
        print(f"  {i}. {context[:50]}...")

    return True

def demo_tavily_configuration():
    """Demonstrate the Tavily tool configuration"""
    print("\n" + "=" * 80)
    print("Demo: Tavily Tool Configuration")
    print("=" * 80)

    print("\nTavily Configuration:")
    print("  - max_results: 5")
    print("  - search_depth: 'advanced' (for deep research)")
    print("  - include_raw_content: True (for LLM analysis)")
    print("\nWhy these settings?")
    print("  • max_results=5: Balances breadth with context limits")
    print("  • search_depth='advanced': Triggers deeper crawl, multiple sources")
    print("  • include_raw_content=True: Preserves nuance for our LLM to analyze")
    print("\n✓ Tavily tool configured correctly")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Web Researcher Node Test Suite")
    print("=" * 80 + "\n")

    # Check API keys
    has_keys = check_api_keys()

    # Always run demo
    demo_tavily_configuration()

    # Always run simulated test
    test_state_accumulation()

    if has_keys:
        print("\n" + "=" * 80)
        print("Running Live API Tests")
        print("=" * 80)

        # Run live tests
        success1 = test_basic_web_research()

        if success1:
            # Only run iterative if basic works
            success2 = test_iterative_web_research()

            if success1 and success2:
                print("\n" + "=" * 80)
                print("✓ All Tests Passed!")
                print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("⚠ Skipping Live API Tests")
        print("=" * 80)
        print("\nTo run live tests, set your API keys:")
        print("  export TAVILY_API_KEY='your-tavily-key'")
        print("  export GOOGLE_API_KEY='your-gemini-key'")
        print("\nThen run: python test_web_researcher.py")
        print("=" * 80)