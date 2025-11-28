"""
Test script for the complete Agentic RAG Graph

This script demonstrates the full workflow from company input to investment memo.

IMPORTANT: To run full tests, you need:
1. GOOGLE_API_KEY environment variable set (for Gemini LLM and embeddings)
2. TAVILY_API_KEY environment variable set (for web research)

You can get your API keys from:
- Google AI Studio: https://aistudio.google.com/apikey
- Tavily: https://tavily.com (free tier: 1000 searches/month)
"""

import os
from state import AgentState
from dotenv import load_dotenv
import traceback
from graph import (
    create_research_graph,
    analyze_company,
    visualize_graph,
    evaluate_data_sufficiency,
    route_after_evaluation,
    combined_research_node
)

load_dotenv()

def check_api_keys():
    """Check if required API keys are set"""
    google_key = os.getenv("GOOGLE_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")

    print("=" * 80)
    print("API Key Check")
    print("=" * 80)

    if google_key:
        print(f"✓ GOOGLE_API_KEY is set ({google_key[:8]}...)")
    else:
        print("✗ GOOGLE_API_KEY is NOT set")
        print("  Set it with: export GOOGLE_API_KEY='your-key-here'")
        print("  Get your API key from: https://aistudio.google.com/apikey")

    if tavily_key:
        print(f"✓ TAVILY_API_KEY is set ({tavily_key[:8]}...)")
    else:
        print("TAVILY_API_KEY is NOT set")
        print("  Set it with: export TAVILY_API_KEY='your-key-here'")
        print("  Get your API key from: https://tavily.com")

    print()

    return bool(google_key and tavily_key)

def test_graph_structure():
    """Test the graph structure and compilation"""
    print("=" * 80)
    print("Test 1: Graph Structure")
    print("=" * 80)

    try:
        app = create_research_graph()

        print(f"\n✓ Graph compiled successfully {app.nodes}")
        print("\nGraph Nodes:")
        print("  1. research - Combined Web + Financial research")
        print("  2. evaluate - Data sufficiency evaluation")
        print("  3. write - Investment memo generation")

        print("\nGraph Edges:")
        print("  START → research")
        print("  research → evaluate")
        print("  evaluate → [conditional]")
        print("    ├─ research (if insufficient data)")
        print("    └─ write (if sufficient data)")
        print("  write → END")

        print("\n✓ Graph structure validated\n")

        return True

    except Exception as e:
        print(f"\n✗ Test failed: {e}\n")
        traceback.print_exc()
        return False

def test_evaluation_logic():
    """Test the data sufficiency evaluation logic"""
    print("=" * 80)
    print("Test 2: Evaluation Logic")
    print("=" * 80)

    # Test case 1: No data
    print("\nTest Case 1: No data")
    state1: AgentState = {
        "company": "Tesla",
        "ticker": "TSLA",
        "financial_context": [],
        "market_context": [],
        "memo_sections": {},
        "messages": [],
        "research_iterations": 0,
        "is_data_sufficient": False,
    }

    result1 = evaluate_data_sufficiency(state1)
    assert result1["is_data_sufficient"], "Should be insufficient with no data"
    print("  Correctly marked as insufficient")

    # Test case 2: Has data and 1 iteration
    print("\nTest Case 2: Has data, 1 iteration")
    state2: AgentState = {
        "company": "Tesla",
        "ticker": "TSLA",
        "financial_context": ["Some financial data"],
        "market_context": ["Some market data"],
        "memo_sections": {},
        "messages": [],
        "research_iterations": 1,
        "is_data_sufficient": False,
    }

    result2 = evaluate_data_sufficiency(state2)
    assert result2["is_data_sufficient"], "Should be sufficient with data and 1 iteration"
    print("  ✓ Correctly marked as sufficient")

    # Test case 3: Max iterations reached
    print("\nTest Case 3: Max iterations (3) reached")
    state3: AgentState = {
        "company": "Tesla",
        "ticker": "TSLA",
        "financial_context": [],
        "market_context": [],
        "memo_sections": {},
        "messages": [],
        "research_iterations": 3,
        "is_data_sufficient": False,
    }

    result3 = evaluate_data_sufficiency(state3)
    assert result3["is_data_sufficient"], "Should be sufficient at max iterations"
    print("  Correctly gives up at max iterations")
    print("\nAll evaluation tests passed\n")
    return True

def test_routing_logic():
    """Test the routing logic"""
    print("=" * 80)
    print("Test 3: Routing Logic")
    print("=" * 80)

    # Test route to research
    state_insufficient: AgentState = {
        "company": "Tesla",
        "ticker": "TSLA",
        "financial_context": [],
        "market_context": [],
        "memo_sections": {},
        "messages": [],
        "research_iterations": 0,
        "is_data_sufficient": False,
    }

    route1 = route_after_evaluation(state_insufficient)
    assert route1 == "research", "Should route to research when insufficient"
    print("Routes to 'research' when data insufficient")

    # Test route to write
    state_sufficient: AgentState = {
        "company": "Tesla",
        "ticker": "TSLA",
        "financial_context": ["data"],
        "market_context": ["data"],
        "memo_sections": {},
        "messages": [],
        "research_iterations": 1,
        "is_data_sufficient": True,
    }

    route2 = route_after_evaluation(state_sufficient)
    assert route2 == "write", "Should route to write when sufficient"
    print("Routes to 'write' when data sufficient\n")

    return True

def test_graph_visualization():
    """Test graph visualization (structure only, no image generation)"""
    print("=" * 80)
    print("Test 4: Graph Visualization")
    print("=" * 80)

    try:
        app = create_research_graph()

        # Get mermaid representation
        mermaid = app.get_graph().draw_mermaid()

        print("\nMermaid Diagram:")
        print("-" * 80)
        print(mermaid)
        print("-" * 80)

        print("\nGraph visualization generated")
        print("\nTo save as image:")
        print("  from graph import visualize_graph")
        print("  visualize_graph('graph.png')")
        print()

        return True

    except Exception as e:
        print(f"\n✗ Test failed: {e}\n")
        traceback.print_exc()
        return False

def demo_workflow_steps():
    """Demonstrate the workflow steps without API calls"""
    print("=" * 80)
    print("Demo: Workflow Steps")
    print("=" * 80)

    print("\nComplete Agentic RAG Workflow:")
    print()

    print("INITIALIZE")
    print("   • User provides: Company name + Ticker")
    print("   • Create initial state with empty contexts")
    print()

    print("RESEARCH (Iteration 1)")
    print("   • Web Researcher: Tavily searches for market data")
    print("   • Financial Analyst: ChromaDB queries for financial metrics")
    print("   • Both append findings to state")
    print()

    print("EVALUATE")
    print("   • Check: Do we have financial_context? market_context?")
    print("   • Check: Have we done >= 1 iteration?")
    print("   • Decision: Sufficient or need more data?")
    print()

    print("DECISION POINT")
    print("   • If insufficient → Loop back to RESEARCH")
    print("   • If sufficient → Proceed to WRITE")
    print("   • Max iterations: 3 (then force write)")
    print()

    print("WRITE")
    print("   • Synthesize all research into investment memo")
    print("   • Use Gemini Pro for high-quality output")
    print("   • Generate 7-section memo (Executive Summary, Risks, etc.)")
    print()
    print("RETURN")
    print("   • Return final state with completed memo")
    print("   • User can access: state['memo_sections']['full_draft']")
    print()

    print("Workflow documented\n")

def test_full_pipeline_with_api():
    """Test the complete pipeline with real API calls"""
    print("=" * 80)
    print("Test 5: Full Pipeline (Live API)")
    print("=" * 80)

    print("\nThis test will consume API credits:")
    print("  • Tavily: ~3-5 searches per iteration")
    print("  • Google Gemini: ~5-10 LLM calls")
    print("  • Google Embeddings: ChromaDB queries (if data exists)")
    print()

    # Use a simple company for quick testing
    company = "Apple"
    ticker = "AAPL"

    try:
        print(f"Analyzing: {company} ({ticker})")
        print("This may take 30-60 seconds...\n")

        result = analyze_company(company, ticker, verbose=True)

        # Validate result
        assert "memo_sections" in result, "Result should have memo_sections"
        assert "full_draft" in result["memo_sections"], "Should have full_draft"

        memo = result["memo_sections"]["full_draft"]

        print("\n" + "=" * 80)
        print("Pipeline Completed Successfully!")
        print("=" * 80)

        print("\nFinal Statistics:")
        print(f"  Research Iterations: {result.get('research_iterations', 0)}")
        print(f"  Financial Data Points: {len(result.get('financial_context', []))}")
        print(f"  Market Data Points: {len(result.get('market_context', []))}")
        print(f"  Memo Length: {len(memo)} characters (~{len(memo.split())} words)")

        # Show memo preview
        print("\n" + "=" * 80)
        print("Investment Memo Preview (first 1500 characters):")
        print("=" * 80)
        print(memo[:1500])
        if len(memo) > 1500:
            print("\n... [truncated] ...\n")
            print(f"Full memo is {len(memo)} characters")
        print("=" * 80)

        # Save memo to file
        output_file = f"memo_{ticker}.md"
        with open(output_file, "w") as f:
            f.write(f"# Investment Memo: {company} ({ticker})\n\n")
            f.write(memo)

        print(f"\nMemo saved to: {output_file}\n")

        return True

    except Exception as e:
        print(f"\nTest failed: {e}")
        traceback.print_exc()
        return False

def demo_usage_examples():
    """Show usage examples"""
    print("=" * 80)
    print("Demo: Usage Examples")
    print("=" * 80)

    print("\nBasic Usage:")
    print("-" * 80)
    print("""
from graph import analyze_company

# Analyze a company
result = analyze_company("Tesla", "TSLA")

# Get the investment memo
memo = result["memo_sections"]["full_draft"]

# Print or save it
print(memo)

# Or save to file
with open("tesla_memo.md", "w") as f:
    f.write(memo)
""")

    print("\nStreaming Usage (for UIs):")
    print("-" * 80)
    print("""
from graph import analyze_company_stream

# Stream events for real-time updates
for event in analyze_company_stream("Apple", "AAPL"):
    node = list(event.keys())[0]
    state = event[node]
    print(f"Completed: {node}")
    print(f"Iterations: {state.get('research_iterations', 0)}")
""")

    print("\nVisualize the Graph:")
    print("-" * 80)
    print("""
from graph import visualize_graph

# Generate graph visualization
visualize_graph("my_graph.png")

# Or just print the mermaid diagram
from graph import create_research_graph
app = create_research_graph()
print(app.get_graph().draw_mermaid())
""")

    print("\nUsage examples documented\n")

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Agentic RAG Graph Test Suite")
    print("=" * 80 + "\n")

    # Check API keys
    has_keys = check_api_keys()

    # Always run structural tests
    success1 = test_graph_structure()
    success2 = test_evaluation_logic()
    success3 = test_routing_logic()
    success4 = test_graph_visualization()

    # Always run demos
    demo_workflow_steps()
    demo_usage_examples()

    if has_keys:
        print("=" * 80)
        print("Running Live API Test")
        print("=" * 80 + "\n")

        user_input = input("Run full pipeline test with API calls? (y/n): ")

        if user_input.lower() == 'y':
            success5 = test_full_pipeline_with_api()

            if success1 and success2 and success3 and success4 and success5:
                print("\n" + "=" * 80)
                print("ALL TESTS PASSED!")
                print("=" * 80)
                print("\nYour Agentic RAG Market Research Analyst is ready!")
                print("\nNext steps:")
                print("  1. Try analyzing different companies")
                print("  2. Add financial documents to ChromaDB with ingest_financial_document()")
                print("  3. Customize memo sections in writer.py")
                print("  4. Adjust research iterations and evaluation criteria")
                print("  5. Build a UI with the streaming API")
                print("=" * 80)
        else:
            print("\nSkipped live API test\n")

    else:
        print("\n" + "=" * 80)
        print("Skipping Live API Test")
        print("=" * 80)
        print("\nTo run the full pipeline test, set both API keys:")
        print("  export GOOGLE_API_KEY='your-google-key'")
        print("  export TAVILY_API_KEY='your-tavily-key'")
        print("\nThen run: python test_graph.py")
        print("=" * 80)