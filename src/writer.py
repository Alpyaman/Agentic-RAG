"""
Writer Node - Investment Memo Synthesizer

This module implements the Writer agent, which synthesizes research findings into a
professional-grade investment memo.

The Writer focuses on:
- Combining financial and market context into a cohesive narrative
- Structuring output according to VC/PE memo standards
- Citing sources rigorously for credibility
- Providing actionable investment recommendations

Key Feature: Uses Gemini Pro for high-quality memo generation with proper citations.
"""

from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from state import AgentState
import traceback

# ============================================================================
# Memo Structure (Based on Sequoia/a16z Templates)
# ============================================================================

MEMO_SECTIONS = {
    "executive_summary": "High-level investment thesis (BUY/HOLD/SELL)",
    "company_overview": "Business model, products, market position",
    "market_analysis": "TAM/SAM, competitive landscape, industry trends",
    "strategic_moat": "Competitive advantages (network effects, IP, brand)",
    "risks_mitigations": "Key risks and potential mitigation strategies",
    "conclusion": "Final recommendation with supporting examples"
}

# ============================================================================
# Writer Node
# ============================================================================

def writer_node(state: AgentState) -> Dict[str, Any]:
    """
    The writer node - synthesizes research into an investment memo.

    This node operates in the following steps:
    1. Extracts accumulated research from state
    2. Combines financial_context and market_context
    3. Generates structured memo using Gemini Pro
    4. Ensures citations for all claims
    5. Returns formatted memo in memo_sections

    Persona: A professional investment analyst writing for an IC (Investment Comittee).
    Writes clearly, cites rigorously, and provides actionable recommendations.

    Args:
        state: The current AgentState with research findings.

    Returns:
        Dictionary with updated memo_sections
    """

    company = state.get("company", "")
    ticker = state.get("ticker", "")
    financial_context = state.get("financial_context", [])
    market_context = state.get("market_context", [])

    print("\n" + "=" * 80)
    print(f"Writer: Synthesizing Investment Memo for {company} ({ticker})")
    print("\n" + "=" * 80)

    # Combine all research context
    all_financial = "\n\n".join(financial_context) if financial_context else "No financial data available."
    all_market = "\n\n".join(market_context) if market_context else "No market research available."

    print(f"Financial context: {len(all_financial)} characters")
    print(f"Market context: {len(all_market)} characters")

    # Generate the memo
    memo = generate_investment_memo(company= company, ticker= ticker, financial_context= all_financial, market_context= all_market)

    print(f"Investment Memo Generated ({len(memo)} characters)\n")

    return {"memo_sections": {"full_draft": memo}}

def generate_investment_memo(company: str, ticker: str, financial_context: str, market_context: str) -> str:
    """
    Generate a professional investment memo using Gemini Pro.

    The memo follows the standard structure used by top VC/PE firms:
    1. Executive Summary
    2. Company Overview
    3. Market Analysis
    4. Financial Performance
    5. Strategic Moat
    6. Risks & Mitigations
    7. Conclusion

    Args:
        company: Company name
        ticker: Stock ticker
        financial_context: Accumulated financial research
        market_context: Accumulated market research

    Returns:
        Formatted investment memo in Markdown
    """

    memo_prompt = ChatPromptTemplate.from_template(
        """You are a senior investment analyst writing an investment memo for {company} ({ticker}).
        
        Your task: Synthesize the research below into a professional investment memo.
        
        # Research Data
        
        ## Financial Analysis:
        {financial_context}

        ### Market Intelligence:
        {market_context}

        # Instructions:

        Write a comprehensive investment memo with the following sections:

        ## 1. Executive Summary
        - **Investment Thesis**: Clear Buy/Hold/Sell recommendation with 2-3 sentence rationale
        - **Key Highlights**: 3-4 bullet points of the most compelling facts
        - **Valuation**: Brief assessment (overvalued/fairly valued/undervalued)

        ## 2. Company Overview
        - Business model and revenue streams
        - Product/service offerings
        - Market position and scale

        ## 3. Market Analysis
        - Total Addressable Market (TAM) and growth trajectory
        - Competitive landscape and market share
        - Industry trends and dynamics

        ## 4. Financial Performance
        - Revenue trends and growth rates (with specific numbers)
        - Profitability metrics (margins, net income)
        - Key financial ratios (P/E, P/S, debt/equity)
        - Present financial data in a table format when possible

        ## 5. Strategic Moat
        - Competitive advantages (network effects, brand, IP, switching costs)
        - Barriers to entry
        - Sustainability of advantages

        ## 6. Risks & Mitigations
        - Key risk factors (regulatory, competitive, operational, market)
        - Potential mitigation strategies
        - Probability and impact assessment

        # 7. Conclusion
        - Final recommendation (Buy/Hold/Sell)
        - Target price or valuation range (if data available)
        - Key catalysts to watch

        # Critical Requirements
        1. **Citations**: EVERY factual claim MUST include a source citation in the format [Source: URL] or [Source: Document].
        2. **Data-Driven**: Use specific numbers, percentages, and dates from the research.
        3. **Objectivity**: Present both bullish and bearish perspectives.
        4. **Actionable**: Provide clear, implementable recommendations
        5. **Professional Tone**: Write for an Investment Comittee audience
        6. **If data is missing**: Explicitly state "Data not available in research" rather than making assumptions

        # Output Format

        Use Markdown formatting with:
        - Headers (##) for sections
        - **Bold** for emphasis on key points
        - Tables for financial data
        - Bullet points for lists
        - [Source: URL] citations after every factual claim

        Begin writing the investment memo now:

        ---

        # Investment Memo: {company} ({ticker})

        """
    )

    # Use Gemini Pro for high-quality generation
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.3)

    chain = memo_prompt | llm

    try:
        print("Generating memo with Gemini Pro...")

        result = chain.invoke({
            "company": company,
            "ticker": ticker,
            "financial_context": financial_context,
            "market_context": market_context
        })

        memo = result.content if hasattr(result, 'content') else str(result)

        if not memo or memo == 'None':
            print("Empty memo response, using fallback\n")
            return generate_fallback_memo(company, ticker, financial_context, market_context)
        
        return memo
    
    except Exception as e:
        print(f"Memo generation failed: {e}")
        traceback.print_exc()
        return generate_fallback_memo(company, ticker, financial_context, market_context)

def generate_fallback_memo(company: str, ticker: str, financial_context: str, market_context: str) -> str:
    """
    Generate a basic memo when AI generation fails.

    This ensures the system always returns something useful.
    """

    return f"""# Investment Memo: {company} ({ticker})

    ## Executive Summary

    **Status**: Preliminary analysis - Full AI generation unavailable

    This memo compiles available research on {company} ({ticker}). A comprehensive
    analysis requires AI processing which is currently unavailable.

    ## Research Summary

    ### Financial Context
    {financial_context[:2000]}...

    ### Market Context
    {market_context[:2000]}...

    ## Next Steps

    1. Retry memo generation with AI model
    2. Manual review of raw research data
    3. Consult with senior analysts for interpretation

    ---

    *Note: This is a fallback memo generated due to system limitations.*
    """

# ============================================================================
# Structured Memo Generation (Alternative Approach)
# ============================================================================

def writer_node_structured(state: AgentState) -> Dict[str, Any]:
    """
    Alternative Writer that generates each section seperately.

    This approach:
    1. Generates sections one at a time (more control)
    2. Allows section-specific prompts (better quality)
    3. Easier to cache and update individual sections

    Trade-off: More LLM calls, but more control and potentially better quality.
    """

    company = state.get("company", "")
    ticker = state.get("ticker", "")
    financial_context = "\n\n".join(state.get("financial_context", []))
    market_context = "\n\n".join(state.get("market_context", []))

    print("\n" + "=" * 80)
    print(f"Writer (Structured): Generating sections for {company}")
    print("\n" + "=" * 80)

    sections = {}

    # Generate each section separately
    for section_name, section_description in MEMO_SECTIONS.items():
        print(f"    Generating: {section_name}")

        section_content = generate_memo_section(company, ticker, section_name, section_description, financial_context, market_context)

        sections[section_name] = section_content
        print(f"    {section_name} ({len(section_content)} chars)")

    # Combine all sections into final memo
    full_memo = f"# Investment Memo: {company} ({ticker})\n\n"

    for section_name in MEMO_SECTIONS.keys():
        full_memo += f"## {section_name.replace('_', ' ').title()}\n\n"
        full_memo += sections[section_name]
        full_memo += "\n\n---\n\n"

    sections["full_draft"] = full_memo

    print(f"\n Complete memo generated ({len(full_memo)} characters)\n")

    return {"memo_sections": sections}

def generate_memo_section(
    company: str,
    ticker: str,
    section_name: str,
    section_description: str,
    financial_context: str,
    market_context: str
) -> str:
    """
    Generate a single section of the memo.

    This allows for section-specific prompting and better control.
    """

    section_prompt = ChatPromptTemplate.from_template(
        """You are writing the {section_name} section for an investment memo on {company} ({ticker}).

Section purpose: {section_description}

Research available:

Financial Context:
{financial_context}

Market Context:
{market_context}

Write a professional, data-driven {section_name} section. Include:
- Specific numbers and dates from the research
- Citations in [Source: URL] format after every claim
- 2-4 paragraphs or structured bullet points
- Clear, actionable insights

{section_name}:"""
    )

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-live", temperature=0.3)
    chain = section_prompt | llm

    try:
        result = chain.invoke({
            "company": company,
            "ticker": ticker,
            "section_name": section_name,
            "section_description": section_description,
            "financial_context": financial_context[:3000],
            "market_context": market_context[:3000]
        })

        return result.content if hasattr(result, 'content') else str(result)

    except Exception as e:
        return f"[Error generating {section_name}: {str(e)}]"