"""
Web Researcher Node - Tavily Integration

This module implements the Web Researcher agentü which is responsible for
gathering real-time market intelligence using the Tavily searh API.

The Web Researcher focuses on:
- Recent news and market developments
- Competitor analysis
- Macro-economic factors
- Executive statements and press releases

It uses Tavily with advanced search depth to ensure comprehensive coverage of
reputable financial sources.
"""

from typing import Dict, Any
from langchain_tavily import TavilySearch
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from state import AgentState

# ===========================================================================================
# Tavily Tool Configuration
# ===========================================================================================

def create_tavily_tool() -> TavilySearch:
    """
    Create and configure the Tavily search tool.

    Configuration choice explained:
    - max_results=5: Balances breadth with context window limits.
    - search_depth='advanced': Critical for deep research. Triggers deeper crawl and aggregates data from multiple sources.
    - include_raw_content=True: We get the full text content, allowing our LLM to perform nuanced
    summarization rather than relying on Tavily's internal summarizer

    Returns:
        TavilySearch: Configured Tavily search tool
    """
    tavily_tool = TavilySearch(
        max_results=5,
        search_depth="advanced",
        include_raw_content=True,
    )
    return tavily_tool

# ===========================================================================================
# Web Research Node
# ===========================================================================================

def web_research_node(state: AgentState) -> Dict[str, Any]:
    """
    The Web Research node - executes Tavily searches and synthesizes findings.

    This node operates in the following steps:
    1. Formulates search quaries based on the company name.
    2. Executes Tavily search with advanced depth.
    3. Uses an LLM to sytnhesize the raw results into concise insights.
    4. Returns the synthesis to be appended to market_context

    Args:
        state: The current AgentState containing company information

    Returns:
        Dictionary with updated market_context (appends to existing list)
    """

    company = state.get("company", "")
    ticker = state.get("ticker", "")

    print("=" * 80)
    print(f"Web Researcher: Researching {company} ({ticker})")
    print("=" * 80)

    # Initialize Tavily tool
    tavily_tool = create_tavily_tool()

    # Formulate search query
    # Focus on recent news, market analysis, and competitive landscape
    query = f"{company} {ticker} market analysis recent news competitors financial performance"
    print(f"Search query: {query}")

    # Execute search
    try:
        raw_results = tavily_tool.invoke(query)

        # Handle Tavily API response format
        # The new Tavily API returns a dict with 'results', 'answer', etc.
        if isinstance(raw_results, dict):
            # Extract the actual results array
            search_results = raw_results.get('results', [])
            # Also get the synthesized answer if available
            tavily_answer = raw_results.get('answer', '')

            print(f"Found {len(search_results)} articles\n")
 
            # Display sources
            if search_results:
                print("📰 Sources:")
                for i, result in enumerate(search_results, 1):
                    if isinstance(result, dict):
                        url = result.get('url', 'N/A')
                        title = result.get('title', 'N/A')
                        print(f"  {i}. {title}")
                        print(f"     {url}\n")

            # If we have Tavily's answer, we can use it directly or enhance it
            if tavily_answer:
                print(f"Tavily Summary Available ({len(tavily_answer)} chars)\n")
        elif isinstance(raw_results, list):
            # Fallback: if it's already a list
            search_results = raw_results
            tavily_answer = ''
            print(f"Found {len(search_results)} articles\n")
        else:
            # Unknown format
            print(f"Unexpected result format: {type(raw_results)}")
            search_results = []
            tavily_answer = str(raw_results)

    except Exception as e:
        print(f"Search failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "market_context": [f"Web search failed: {str(e)}"]
        }

    # Synthesize results using LLM
    synthesis = synthesize_search_results(company, ticker, search_results, tavily_answer)

    return {"market_context": [synthesis]}

def synthesize_search_results(company: str, ticker: str, search_results: list, tavily_answer: str = "") -> str:
    """
    Use a LLM to synthesize raw search results into concise market insights.

    This is critical: we don't just dump raw JSON into the state. Instead, we use a LLM to
    compress and structure the findings, preserving the context window for the final Writer node.
    
    Args:
        company: Company name
        ticker: Stock ticker
        search_results: Raw results from Tavily
        tavily_answer: Optional pre-synthesized answer from Tavily

    Returns:
        Ssynthesized market intelligence summary
    """

    # If Tavily already provided a good answer and we don't have detailed results,
    # we can use or enhance it
    if tavily_answer and not search_results:
        return tavily_answer

    # Extract content from search results
    articles_content = []

    for i, result in enumerate(search_results, 1):
        if isinstance(result, str):
            # String format
            articles_content.append(f"Article {i}:\n{result}\n")
        elif isinstance(result, dict):
            # Dict format with title, content, url
            title = result.get('title', '')
            content = result.get('content', '')
            url = result.get('url', '')
            articles_content.append(f"Title: {title}\nURL: {url}\nContent: {content}\n")
        else:
            # Unknown format
            articles_content.append(f"Article {i}:\n{str(result)}\n")

    # If we have no content, return the Tavily answer or a fallback
    if not articles_content:
        if tavily_answer:
            return tavily_answer
        return f"No detailed research results available for {company}."

    combined_content = "\n---\n".join(articles_content)

    # Create synthesis prompt
    syntesis_prompt = ChatPromptTemplate.from_template(
        """You are a market investigator analyzing web research for {company} ({ticker}).
        
        Your task: Synthesize the following articles into a concise market intelligence summary.
        
        Focus on:
        1. **Recent Developments**: Key news, product launches, strategic moves.
        2. **Market Trends**: Industry dynamics, competitive landscape
        3. **Financial Signals**: Revenue trends, profitability, market sentiment
        4. **Risk & Oppurtunities**: Regulatory issues, market headwinds/tailwinds

        Requirements:
        - Be concise but comprehensive (300 - 500 words)
        - Cite sources using [Source: URL] format
        - Focus on facts, not speculation
        - Prioritize reputable sources (Bloomberg, Reuters, WSJ, etc.)

        Articles:
        {articles}

        Synthesis:"""
    )

    # Use a fast model for synthesis (we're not generating the final memo yet)
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0)

    chain = syntesis_prompt | llm

    try:
        result = chain.invoke({
            "company": company,
            "ticker": ticker,
            "articles": combined_content
        })

        # Extract content from Gemini response
        # ChatGoogleGenerativeAI returns AIMessage with content attribute
        if hasattr(result, 'content'):
            synthesis = result.content
        elif isinstance(result, str):
            synthesis = result
        else:
            synthesis = str(result)

        # Handle None or empty response
        if not synthesis or synthesis == 'None':
            print("Empty synthesis response, using fallback\n")
            if tavily_answer:
                return tavily_answer
            # Fallback: return raw titles
            fallback = f"Market Research Summary for {company}:\n\n"
            for i, res in enumerate(search_results, 1):
                if isinstance(res, dict):
                    title = res.get('title', 'N/A')
                    url = res.get('url', 'N/A')
                    content_preview = res.get('content', '')[:200] + "..."
                    fallback += f"{i}. {title}\n   {url}\n   {content_preview}\n\n"
            return fallback

        print(f"Synthesis Complete ({len(synthesis)} chars)\n")
        return synthesis

    except Exception as e:
        print(f"Synthesis failed: {e}")
        import traceback
        traceback.print_exc()

        # Fallback: return Tavily answer or construct summary
        if tavily_answer:
            return tavily_answer

        fallback = f"Market Research Summary for {company}:\n\n"
        for i, res in enumerate(search_results, 1):
            if isinstance(res, dict):
                title = res.get('title', 'N/A')
                url = res.get('url', 'N/A')
                fallback += f"{i}. {title} - {url}\n"
            return fallback

# ===========================================================================================
# Iterative Search  Pattern (Advanced)
# ===========================================================================================

def iterative_web_research_node(state: AgentState) -> Dict[str, Any]:
    """
    Advanced Web Researcher with iterative refinement.

    This implements a "Tree of Thoughts" approach:
    1. Initial broad search.
    2. Analyze gaps in coverage.
    3. Formulate targeted secondary searches.
    4. Synthesize all findings

    This is the pattern described in the document:
    "This agent utilizes a Tree of Thoughts approach. It searches, summarizes
    the results, and then asks 'Is this enough?' If not, it generates a new
    search query based on the gaps in the previous search.

    Args:
        state: Current AgentState
    
    Returns:
        Dictionary with comprehensive market_context
    """

    company = state.get("company", "")
    ticker = state.get("ticker", "")

    print("=" * 80)
    print(f"Iterative Web Researcher: Deep Research on {company}")
    print("=" * 80)

    tavily_tool = create_tavily_tool()
    all_results = []
    all_answers = []

    # Phase 1: Broad initial search
    print("Phase 1: Broad Market Search")
    initial_query = f"{company} {ticker} recent news market analysis"
    raw_response = tavily_tool.invoke(initial_query)

    # Handle response format
    if isinstance(raw_response, dict):
        initial_results = raw_response.get('results', [])
        if raw_response.get('answer'):
            all_answers.append(raw_response['answer'])
    else:
        initial_results = raw_response if isinstance(raw_response, list) else []

    all_results.extend(initial_results)
    print(f"   Found {len(initial_results)} articles\n")

    # Phase 2: Targeted searches
    targeted_queries = [
        f"{company} {ticker} financial results earnings revenue",
        f"{company} {ticker} competitors competitive analysis market share",
        f"{company} risks challenges regulatory",
    ]

    for i, query in enumerate(targeted_queries, 2):
        print(f"   Phase {i}: Targeted Search")
        print(f"   Query: {query}")
        try:
            raw_response = tavily_tool.invoke(query)

            # Handle response format
            if isinstance(raw_response, dict):
                results = raw_response.get('results', [])
                if raw_response.get('answer'):
                    all_answers.append(raw_response['answer'])
            else:
                results = raw_response if isinstance(raw_response, list) else []

            all_results.extend(results)
            print(f"   Found {len(results)} articles\n")
        except Exception as e:
            print(f"   Search failed: {e}\n")

    # Synthesize all findings
    print(f"Total articles collected: {len(all_results)}")
    # Combine all Tavily answers if available
    combined_tavily_answer = "\n\n".join(all_answers) if all_answers else ""
    synthesis = synthesize_search_results(company, ticker, all_results, combined_tavily_answer)

    return {"market_context": [synthesis]}