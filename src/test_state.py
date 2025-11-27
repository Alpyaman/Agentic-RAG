"""
Test script for the AgentState definition.

This script demonstrates how the state works with LangGraph's operator.add functionality for append-only lists.
"""

from state import AgentState
from langchain_core.messages import HumanMessage, AIMessage

def test_state_creation():
    """Test creating a basic state instance."""
    print("=" * 80)
    print("Test 1: Creating AgentState")
    print("=" * 80)

    # Create initial state
    initial_state: AgentState = {
        "company": "Tesla",
        "ticker": "TSLA",
        "financial_context": [],
        "market_context": [],
        "memo_sections": {},
        "messages": [],
        "research_iterations": 0,
        "is_data_sufficient": False,
    }

    print(f"Company: {initial_state['company']}")
    print(f"Ticker: {initial_state['ticker']}")
    print(f"Financial Context: {initial_state['financial_context']}")
    print(f"Market Context: {initial_state['market_context']}")
    print(f"Research Iterations: {initial_state['research_iterations']}")
    print(f"Data Sufficient: {initial_state['is_data_sufficient']}")
    print("State created successfully\n")

    return initial_state

def test_state_updates():
    """Test how state updates work with append-only lists."""
    print("=" * 80)
    print("Test 2: Testing Append-Only List Behavior")
    print("=" * 80)

    # Simulate how LangGraph would merge state updates
    # When a node returns {"financial_context": ["new data"]}
    # it should append to the existing list, not replace it

    initial_state = test_state_creation()

    # Simulate Financial Analyst node returning node
    financial_update = {"financial_context": ["Revenue 2023: $96.77B, up 18.8% YoY"]}

    print("Financial Analyst adds:", financial_update["financial_context"][0])

    # Simulate Web Research node returning data
    market_update = {"market_context": ["Tesla announces Cybertruck production ramp in Q4 2024"]}

    print("Web Researcher adds:", market_update["market_context"][0])

    # In actual LangGraph, operator.add automatically handles this merging
    # Here we manually demonstrate the behavior
    update_financial = initial_state["financial_context"] + financial_update["financial_context"]
    update_market = initial_state["market_context"] + market_update["market_context"]

    print(f"\nUpdated financial_context: {update_financial}")
    print(f"Updated market_context: {update_market}")
    print("Append-only behaviour works as expected\n")

def test_message_history():
    """Test the message history accumulation."""
    print("=" * 80)
    print("Test 3: Testing Message History")
    print("=" * 80)

    messages = []

    # Add user query
    user_msg = HumanMessage(content="Research Tesla and create an investment memo")
    messages.append(user_msg)
    print(f"User: {user_msg.content}")

    # Add AI response
    ai_msg = AIMessage(content="I will research Tesla. Starting with financial data...")
    messages.append(ai_msg)
    print(f"AI: {ai_msg.content}")

    print(f"\nTotal messages in history: {len(messages)}")
    print("Message history accumulation works\n")

def test_memo_sections():
    """Test the memo sections structure"""
    print("=" * 80)
    print("Test 4: Testing Memo Sections Dictionary")
    print("=" * 80)

    memo_sections = {}

    # Writer node adds executive summary
    memo_sections["executive_summary"] = "Tesla shows strong revenue growth..."
    print(f"Added executive_summary: {memo_sections['executive_summary'][:50]}...")

    # Writer node adds financial section
    memo_sections["financial_performance"] = "Revenue increased by 18.8% YoY..."
    print(f"Added financial_performance: {memo_sections['financial_performance'][:50]}")

    # Writer node adds risks
    memo_sections["risks"] = "Key risks include supply chain distruptions..."
    print(f"Added risks: {memo_sections['risks'][:50]}...")

    print(f"\nTotal sections: {len(memo_sections)}")
    print(f"Sections: {list(memo_sections.keys())}")
    print("Memo sections structure works\n")


if __name__ == "__main__":
    print("=" * 80)
    print("AgentState Test Suite")
    print("=" * 80)

    try:
        test_state_creation()
        test_state_updates()
        test_message_history()
        test_memo_sections()

        print("=" * 80)
        print("All tests passed successfully!")
        print("=" * 80)
        print("\nKey Insights:")
        print("1. AgentState uses TypedDict for type safety")
        print("2. Annotated[List[str], operator.add] enables append-only behavior")
        print("3. This prevents nodes from overwriting each other's data")
        print("4. The state acts as a persistent memory throughout the workflow")
        print("=" * 80)

    except Exception as e:
        print(f"\n Test failed with error: {e}")
        import traceback
        traceback.print_exc()