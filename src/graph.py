"""
Graph Assembly - Complete Agentic RAG Workflow

This module assembles all the components into a LangGraph StateGraph:
- Web Researcher: Gathers market intelligence and real-time data
- Financial Analyst: Extracts structured financial metrics from vector DB
- Writer: Synthesizes findings into professional investment memo

The graph implements an agentic loop:
1. Research Phase: Web + Financial analysts run (can be parallel)
2. Evaluation: Check if data is sufficient for memo
3. Loop: If insufficient, research again (max iterations)
4. Write: Generate final investment memo
5. Return: Complete state with memo

Key Feature: Conditional edges based on data sufficiency evaluation
"""

from typing import Dict, Any, Literal
from langgraph.graph import StateGraph, END
from state import AgentState
from web_researcher import iterative_web_research_node
from financial_analyst import financial_analyst_node
from writer import writer_node

# ============================================================================
# Evaluation Node - Decides if we have enough data
# ============================================================================

def evaluate_data_sufficiency(state: AgentState) -> Dict[str, Any]:
    """
    Evaluate if we have sufficient data to write an investment memo.

    This is a simple heuristic-based evaluation. In production, you might
    use an LLM to assess data quality and completeness.

    Criteria:
    - Have we gathered market context? (web research)
    - Have we gathered financial context? (vector DB)
    - Have we done at least 1 research iteration?
    - Are we under max iterations?

    Args:
        state: Current agent state

    Returns:
        Updated state with is_data_sufficient flag
    """

    financial_context = state.get("financial_context", [])
    market_context = state.get("market_context", [])
    iterations = state.get("research_iterations", 0)

    # Simple heuristic: need both types of data
    has_financial = len(financial_context) > 0
    has_market = len(market_context) > 0

    # Consider sufficient if we have both and did at least 1 iteration
    # Or if we've hit max iterations (give up and write with what we have)
    MAX_ITERATIONS = 3

    is_sufficient = (has_financial and has_market and iterations >= 1) or iterations >= MAX_ITERATIONS

    print(f"\n{'='*80}")
    print("Data Sufficiency Evaluation")
    print(f"{'='*80}")
    print(f"Financial Context: {len(financial_context)} items")
    print(f"Market Context: {len(market_context)} items")
    print(f"Research Iterations: {iterations}/{MAX_ITERATIONS}")
    print(f"Sufficient: {'✓' if is_sufficient else '✗'}")
    print(f"{'='*80}\n")

    return {
        "is_data_sufficient": is_sufficient,
        "research_iterations": iterations + 1
    }

# ============================================================================
# Routing Function - Decides next step based on data sufficiency
# ============================================================================

def route_after_evaluation(state: AgentState) -> Literal["research", "write"]:
    """
    Route to either more research or final write step.

    This is called by a conditional edge in the graph.

    Args:
        state: Current agent state

    Returns:
        "research" if we need more data, "write" if ready to generate memo
    """
    if state.get("is_data_sufficient", False):
        print("→ Routing to: WRITE (generating investment memo)\n")
        return "write"
    else:
        print("→ Routing to: RESEARCH (gathering more data)\n")
        return "research"

# ============================================================================
# Research Phase Nodes - Parallel Execution
# ============================================================================

def start_research_phase(state: AgentState) -> Dict[str, Any]:
    """
    Print status message at the start of a research iteration.

    This is a lightweight node that just logs progress before
    the parallel research nodes execute.

    Args:
        state: Current agent state

    Returns:
        Empty dict (no state updates)
    """
    print(f"\n{'='*80}")
    print(f"Starting Research Phase - Iteration {state.get('research_iterations', 0) + 1}")
    print(f"{'='*80}\n")

    return {}

def web_research_wrapper(state: AgentState) -> Dict[str, Any]:
    """
    Wrapper for Web Researcher node (enables parallel execution).

    This wrapper allows the web researcher to be a separate node
    in the graph that runs in parallel with the financial analyst.

    Args:
        state: Current agent state

    Returns:
        Web research results
    """
    print("Web Researcher: Starting parallel execution...")
    return iterative_web_research_node(state)

def financial_analysis_wrapper(state: AgentState) -> Dict[str, Any]:
    """
    Wrapper for Financial Analyst node (enables parallel execution).

    This wrapper allows the financial analyst to be a separate node
    in the graph that runs in parallel with the web researcher.

    Args:
        state: Current agent state

    Returns:
        Financial analysis results
    """
    print("Financial Analyst: Starting parallel execution...")
    return financial_analyst_node(state)

# ============================================================================
# Graph Construction
# ============================================================================

def create_research_graph() -> StateGraph:
    """
    Create the complete LangGraph workflow with PARALLEL EXECUTION.

    Graph Structure (NEW - Parallel Research):

        START
          ↓
       start_research (logs iteration)
          ↓
        ┌─┴─┐
        ↓   ↓
    web_research  financial_analysis
        ↓   ↓           (PARALLEL)
        └─┬─┘
          ↓
       EVALUATE (check sufficiency)
          ↓
      [Decision]
       ↙     ↘
    start_research  WRITE
    (loop)            ↓
                     END

    Performance Improvement:
    - Web Researcher and Financial Analyst run concurrently
    - Reduces research time by ~50% (if both take similar time)
    - Both nodes write to separate state keys (no conflicts)
    - State merging happens automatically via operator.add

    Returns:
        Compiled StateGraph ready to execute
    """

    # Initialize graph with our state schema
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("start_research", start_research_phase)
    workflow.add_node("web_research", web_research_wrapper)
    workflow.add_node("financial_analysis", financial_analysis_wrapper)
    workflow.add_node("evaluate", evaluate_data_sufficiency)
    workflow.add_node("write", writer_node)

    # Define edges
    # Start with research phase initialization
    workflow.set_entry_point("start_research")

    # PARALLEL EXECUTION: Both researchers start simultaneously
    workflow.add_edge("start_research", "web_research")
    workflow.add_edge("start_research", "financial_analysis")

    # Both researchers feed into evaluation
    # LangGraph automatically waits for both to complete before proceeding
    workflow.add_edge("web_research", "evaluate")
    workflow.add_edge("financial_analysis", "evaluate")

    # After evaluation, conditionally route
    workflow.add_conditional_edges(
        "evaluate",
        route_after_evaluation,
        {
            "research": "start_research",  # Loop back for more data
            "write": "write"               # Proceed to memo generation
        }
    )

    # After writing, we're done
    workflow.add_edge("write", END)

    # Compile the graph
    app = workflow.compile()

    return app

# ============================================================================
# Convenience Functions
# ============================================================================

def analyze_company(company: str, ticker: str, verbose: bool = True) -> Dict[str, Any]:
    """
    Main entry point - analyze a company and generate investment memo.

    This is the function you call to run the entire pipeline.

    Example:
        result = analyze_company("Tesla", "TSLA")
        memo = result["memo_sections"]["full_draft"]
        print(memo)

    Args:
        company: Company name (e.g., "Tesla")
        ticker: Stock ticker (e.g., "TSLA")
        verbose: Print progress updates

    Returns:
        Final state with completed investment memo
    """

    if verbose:
        print(f"\n{'='*80}")
        print("Agentic RAG Market Research Analyst")
        print(f"{'='*80}")
        print(f"Company: {company}")
        print(f"Ticker: {ticker}")
        print(f"{'='*80}\n")

    # Create initial state
    initial_state: AgentState = {
        "company": company,
        "ticker": ticker,
        "financial_context": [],
        "market_context": [],
        "memo_sections": {},
        "messages": [],
        "research_iterations": 0,
        "is_data_sufficient": False,
    }

    # Create and run the graph
    app = create_research_graph()

    # Execute the workflow
    final_state = app.invoke(initial_state)

    if verbose:
        print(f"\n{'='*80}")
        print("Analysis Complete!")
        print(f"{'='*80}")
        print(f"Total Research Iterations: {final_state.get('research_iterations', 0)}")
        print(f"Financial Data Points: {len(final_state.get('financial_context', []))}")
        print(f"Market Data Points: {len(final_state.get('market_context', []))}")

        if "full_draft" in final_state.get("memo_sections", {}):
            memo_length = len(final_state["memo_sections"]["full_draft"])
            print(f"Memo Length: {memo_length} characters (~{memo_length // 5} words)")

        print(f"{'='*80}\n")

    return final_state

# ============================================================================
# Streaming Support (for UI integration)
# ============================================================================

def analyze_company_stream(
    company: str,
    ticker: str
):
    """
    Stream the analysis process for real-time UI updates.

    This is useful for web applications that want to show progress.

    Example:
        for event in analyze_company_stream("Tesla", "TSLA"):
            print(f"Node: {event['node']}")
            print(f"State: {event['state']}")

    Args:
        company: Company name
        ticker: Stock ticker

    Yields:
        Events with node name and current state
    """

    initial_state: AgentState = {
        "company": company,
        "ticker": ticker,
        "financial_context": [],
        "market_context": [],
        "memo_sections": {},
        "messages": [],
        "research_iterations": 0,
        "is_data_sufficient": False,
    }

    app = create_research_graph()

    # Stream events
    for event in app.stream(initial_state):
        yield event

# ============================================================================
# Visualization
# ============================================================================

def visualize_graph(output_path: str = "graph.png"):
    """
    Generate a visualization of the graph structure.

    Requires: pip install pygraphviz (optional)

    Args:
        output_path: Where to save the graph image
    """
    try:
        app = create_research_graph()

        # Get mermaid diagram
        print("Graph Structure (Mermaid):")
        print(app.get_graph().draw_mermaid())

        # Try to save as image if pygraphviz is available
        try:
            img = app.get_graph().draw_mermaid_png()
            with open(output_path, "wb") as f:
                f.write(img)
            print(f"\nGraph visualization saved to: {output_path}")
        except Exception:
            print("\nNote: Install pygraphviz for image export: pip install pygraphviz")

    except ImportError:
        app = create_research_graph()
        print("Graph Structure (Mermaid):")
        print(app.get_graph().draw_mermaid())
        print("\nNote: Install IPython for enhanced visualization: pip install ipython")

if __name__ == "__main__":
    # Quick test
    print("Graph module loaded successfully!")
    print("\nTo use:")
    print("  from graph import analyze_company")
    print("  result = analyze_company('Tesla', 'TSLA')")
    print("  print(result['memo_sections']['full_draft'])")
    print("\nOr visualize:")
    print("  from graph import visualize_graph")
    print("  visualize_graph()")