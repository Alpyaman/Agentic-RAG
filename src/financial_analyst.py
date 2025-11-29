"""
Financial Analyst Node - Vector DB Integration

This module implements the Financial Analyst agent, which is responsible for extracting and
analyzing financial data from structured documents (10-Ks, 10-Qs, earnings transcripts)
stored in a vector database.

The Financial Analyst focuses on:
- Retrieving financial statements and metrics.
- Performing mathematical calculations with 100% accuracy.
- Extracting time-series data (revenue, expenses, ratios)
- Analyzing risks from regulatory filings

Key Feature: Uses a Python REPL tool for calculations to avoid LLM arithmetic errors.
"""

from typing import Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_experimental.tools import PythonREPLTool
from state import AgentState
from llama_parse import LlamaParse
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import traceback

# ===========================================================================================
# Python REPL Tool for Accurate Calculations
# ===========================================================================================

def create_python_repl():
    """
    Create a Python REPL tool for mathematical calculations.

    Why use Python instead of LLM arithmetic?
    - LLMs are notoriously bad at math
    - Financial analysis requires 100% accuracy
    - Python ensures determenistic, precise calculations

    Example usage:
    - Calculate debt-to-equity ratio: debt / equity
    - Compute CAGR: ((ending / beginning) ** (1/years)) - 1
    - Analyze margins: (net_income / revenue) * 100

    Returns:
        PythonREPLTool: Tool for executing Python code
    """
    python_repl = PythonREPLTool()
    return python_repl


# ===========================================================================================
# Vector Store Configuration
# ===========================================================================================

def create_vector_store(collection_name: str = "financial_reports", persist_directory: str = "./chroma_db") -> Chroma:
    """
    Create or load a ChromaDB vector store for financial documents.

    Schema Desing:
    - Collection: financial_reports
    - Metadata fields:
        - ticker: Stock symbol
        - year: Fiscal year
        - quarter: Quarter
        - doc_type: Document type
        - section: Section name

    Args:
        collection_name: Name of the ChromaDB collection
        persist_directory: Directory to persist the database

    Returns:
        Chroma: Configured vector state
    """
    # Use Google's embedding model (consistent with Gemini usage)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp-03-07")

    # Initialize or load ChromaDB
    vectorstore = Chroma(collection_name=collection_name, embedding_function=embeddings, persist_directory=persist_directory)

    return vectorstore


# ===========================================================================================
# Financial Analyst Node
# ===========================================================================================

def financial_analyst_node(state: AgentState) -> Dict[str, Any]:
    """
    The Financial Analyst Node - queries vector DB for financial data.

    This node operates in the following steps:
    1. Analyzes the research requirements from state
    2. Formulates specific financial queries
    3. Retrieves relevant documents from ChromaDB
    4. Uses Python REPL for any calculations
    5. Extracts structured financial metrics
    6. Returns findings to be appended to financial_context

    Persona: A skeptical, data-driven forensic accountant who only trusts verified data from official filings.

    Args:
        state: The current AgentState containing company information

    Returns:
        Dictionary with updated financial_context.
    """

    company = state.get("company", "")
    ticker = state.get("ticker", "")

    print("\n" + "=" * 80)
    print(f"Financial Analyst: Analyzing {company} ({ticker})")
    print("\n" + "=" * 80)

    # Initialize tools
    vectorstore = create_vector_store()
    python_repl = create_python_repl()

    # Define financial queries to research
    # These are the key questions an investor would ask
    queries = [
        f"What is {company}'s revenue for the last 3 years?",
        f"What are the key risk factors for {company}?",
        f"What is {company}'s debt-to-equity ratio?",
        f"What are {company}'s operating margins?"
    ]

    findings = []

    print("Researching Financial Data:\n")

    for query in queries:
        print(f"    Query: {query}")

        try:
            # Retrieves relevant documents (Basic usage for now)
            docs = vectorstore.similarity_search(query, k=3, filter={"ticker": ticker} if ticker else None)

            if docs:
                # Extract content from retrieved documents
                context = "\n\n".join([doc.page_content for doc in docs])

                # Use LLM to extract the answer from context
                answer = extract_financial_answer(query=query, context=context, company=company, python_repl=python_repl)

                findings.append(f"**{query}**\n{answer}")
                print("Found data\n")
            else:
                findings.append(f"**{query}**\n No data available in vector database.")

                print("No data found\n")

        except Exception as e:
            print(f" Error: {e}")
            findings.append(f"**{query}**\nError retrieving data: {str(e)}")

    # Combine all findings
    financial_summary = "\n\n".join(findings)

    return {"financial_context": [financial_summary]}

def extract_financial_answer(query: str, context: str, company: str, python_repl: PythonREPLTool) -> str:
    """
    Use an LLM with Tool Use to extract financial answers from retrieved context.

    This implements a three-pass agent pattern:
    1. Pass 1: LLM extracts data and generates Python code for calculations
    2. Pass 2: Execute Python code via REPL for 100% accurate results
    3. Pass 3: LLM incorporates calculation results into final answer

    This ensures that financial calculations (ratios, growth rates, percentages)
    are mathematically precise, not hallucinated by the LLM.

    Args:
        query: The financial question
        context: Retrieved document content
        company: Company name
        python_repl: Python REPL tool for calculations

    Returns:
        Extracted answer with citations and accurate calculations
    """

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0)

    # ========================================================================
    # PASS 1: Extract data and identify calculations needed
    # ========================================================================

    extraction_prompt = ChatPromptTemplate.from_template(
        """
        You are a forensic accountant analyzing financial documents for {company}.
        
        Your task: Extract data to answer the following question.

        Question: {query}

        Context from financial filings:
        {context}

        Instructions:
        1. Extract all relevant numbers from the context (revenue, debt, equity, margins, etc.)
        2. If calculations are needed (ratios, percentages, growth rates, CAGR):
        - DO NOT calculate yourself (LLMs are bad at math)
        - Instead, generate valid Python code to perform the calculation
        - Use descriptive variable names
        - Format code in a ```python code block
        3. Cite the source (e.g., "From 10-K FY 2023, page 45")
        4. If data is missing, state what's missing

        Example output format:
        "Revenue 2023: $96.77B (From 10-K FY 2023)
        Revenue 2022: $81.46B (From 10-K FY 2022)

        To calculate YoY growth:
        ```python
        revenue_2023 = 96.77
        revenue_2022 = 81.46
        yoy_growth = ((revenue_2023 - revenue_2022) / revenue_2022) * 100
        result = round(yoy_growth, 2)
        ```"

        Your response:"""
        )

    try:
        # Get initial extraction with potential Python code
        chain = extraction_prompt | llm
        result = chain.invoke({
            "company": company,
            "query": query,
            "context": context[:4000]  # Limit context to avoid token limits
        })

        initial_answer = result.content if hasattr(result, 'content') else str(result)

        # ========================================================================
        # PASS 2: Execute any Python code found in the response
        # ========================================================================

        calculation_results = {}

        # Extract Python code blocks
        import re
        code_blocks = re.findall(r'```python\n(.*?)\n```', initial_answer, re.DOTALL)

        if code_blocks:
            print("    Executing calculations via Python REPL...")

            for i, code in enumerate(code_blocks):
                try:
                    # Execute the code via Python REPL
                    # Ensure we capture the 'result' variable
                    execution_code = code.strip()

                    # If code doesn't assign to 'result', capture last expression
                    if 'result' not in execution_code:
                        execution_code += "\nresult = locals().get('result', 'Calculation complete')"

                    # Execute via REPL
                    output = python_repl.run(execution_code)

                    # Parse the result
                    # The REPL returns the output as a string
                    calculation_results[f'calc_{i}'] = output.strip()

                    print(f"       ✓ Calculation {i+1}: {output.strip()}")

                except Exception as e:
                    calculation_results[f'calc_{i}'] = f"Error: {str(e)}"
                    print(f"       ✗ Calculation {i+1} failed: {e}")

         # ========================================================================
        # PASS 3: Incorporate calculation results into final answer
        # ========================================================================

        if calculation_results:
            # Build final answer with precise calculations
            final_prompt = ChatPromptTemplate.from_template(
                """
                You are a forensic accountant finalizing your analysis for {company}.

                You previously extracted data and identified calculations needed:
                {initial_answer}

                Python REPL executed the calculations with these results:
                {calculation_results}

                Your task: Write a final, concise answer to the original question, incorporating the EXACT numerical results from Python.

                Question: {query}

                Instructions:
                1. Use the precise numbers from Python (not rounded estimates)
                2. Format percentages clearly (e.g., "18.8% YoY growth")
                3. Maintain citations to source documents
                4. Keep answer concise (2-3 sentences)

                Final answer:"""
                )

            chain = final_prompt | llm
            final_result = chain.invoke({
                "company": company,
                "query": query,
                "initial_answer": initial_answer,
                "calculation_results": "\n".join([f"{k}: {v}" for k, v in calculation_results.items()])
            })

            return final_result.content if hasattr(final_result, 'content') else str(final_result)

        else:
            # No calculations needed, return initial answer
            return initial_answer
    
    except Exception as e:
        return f"Error extracting answer: {str(e)}"
    
# ===========================================================================================
# Document Ingestion Pipeline (LlamaParse)
# ===========================================================================================

def ingest_financial_document(file_path: str, ticker: str, year: int, doc_type: str="10-K", quarter: Optional[str] = None, vectorstore: Optional[Chroma] = None) -> int:
    """
    Ingest a financial document into the vector database using LlamaParse.

    This function:
    1. Parses the PDF using LlamaParse (preserves table structure)
    2. Splits into chunks with metadata
    3. Embeds and stores in ChromaDB
    
    Args:
        file_path: Path to the PDF document
        ticker: Stock ticker symbol
        year: Fiscal year
        doc_type: Type of document ("10-K", "10-Q", "Earnings")
        quarter: Quarter - only for 10-Qs
        vectorstore: Optional existing vectorstore (creates new if None)

    Returns:
        Number of chunks ingested
    """

    print("\n" + "=" * 80)
    print(f"Ingesting {doc_type} for {ticker} - {year}")
    print("\n" + "=" * 80)

    try:
        # Initialize LlamaParse
        # This uses vision-language models to understand table structure
        parser = LlamaParse(result_type="markdown", verbose=True)

        print("Parsing document with LlamaParse...")
        documents = parser.load_data(file_path)

        # Conver to LangChain Document format
        docs = [
            Document(
                page_content=doc.text,
                metadata={
                    "ticker": ticker,
                    "year": year,
                    "doc_type": doc_type,
                    "quarter": quarter or "FY",
                    "source": file_path
                }
            )
            for doc in documents
        ]

        # Split into chunks
        # We use larger chunks for financial data to preserve context
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", " "]
        )

        splits = text_splitter.split_documents(docs)

        print(f"Created {len(splits)} chunks")

        # Initialize vectorstore if not provided
        if vectorstore is None:
            vectorstore = create_vector_store()

        # Add to vector store
        print("Adding to ChromaDB...")
        vectorstore.add_documents(splits)

        print(f"Successfully ingested {len(splits)} chunks\n")

        return len(splits)
    
    except ImportError as e:
        print(f"Error: {str(e)}")
        raise # Re-raise the exception so caller knows it failed

    except Exception as e:
        print(f"Ingestion failed: {str(e)}\n")
        traceback.print_exc()
        raise 