"""
State definition for the Market Research Analyst Agent

This module defines the AgentState, which serves as the single source of truth for the research campaign.
It is passed from node to node in the LanghGraph, accumulating results and maintaining the workflow context.
"""

from typing import TypedDict, List, Annotated
import operator
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """
    The state schema for the Market Research Analyst Agent.

    This is not merely a list of messages but a structured repository of the research campaign
    ensuring that data flows correctly between specialized nodes (agents) in the graph.
    """

    # ===========================================================================================
    # Research Target Information
    # ===========================================================================================

    company: str
    """The target company name for research."""

    ticker:str
    """The stock ticker symbol."""

    # ===========================================================================================
    # Research Data Accumulators (Append-Only)
    # ===========================================================================================

    financial_context: Annotated[List[str], operator.add]
    """
    Append only list of financial findings from the Vector DB.
    Uses operator.add to ensure that when a node returns new financial data,
    it appends to the existing list rather than overwriting it.
    """

    market_context: Annotated[List[str], operator.add]
    """
    Append only list of web findings from Tavily searches.
    Allows the Web Researcher to run multiple times, building a rich corpus of market intelligence.
    """

    # ===========================================================================================
    # The Iterative Draft
    # ===========================================================================================

    memo_sections: dict
    """
    Dictionary containing the various sections of the investment memo.
    Example structure:
    {
        'executive_summary': '...',
        'company_overview': '...',
        'financial_performance': '...',
        'risks': '...',
        'conclusion': '...',
        'full_draft': '...'
    }
    """

    # ===========================================================================================
    # Conversation History of LLM Context
    # ===========================================================================================

    messages: Annotated[List[BaseMessage], operator.add]
    """
    Chat history for maintaining LLM context across nodes.
    This allows agents to reference previous reasoning and decisions.
    """

    # ===========================================================================================
    # Control Flow Flags
    # ===========================================================================================

    research_iterations: int
    """
    Counter to track the number of research cycles.
    Used to prevent infinite loops by enforcing a maximum iterations limit.
    """

    is_data_sufficient: bool
    """
    Flag indicating whether sufficient data has been gathered.
    Set by the Quality Reviewer node to determine if more research is needed.
    """