# Agentic RAG Market Research Analyst

An AI-powered investment research system that generates professional investment memos by autonomously gathering and analyzing market data.

## What is Agentic RAG?

Unlike traditional RAG (Retrieval-Augmented Generation), which follows a linear retrieve → generate flow, **Agentic RAG** features:

- **Iterative Research Loops**: System evaluates its progress and decides whether to gather more data
- **Multi-Source Intelligence**: Combines web search (Tavily) + vector database (ChromaDB)
- **Quality Control Gates**: Validates data sufficiency before proceeding to final output
- **Structured State Management**: Deterministic data flow prevents information loss
- **Professional Output**: Generates VC/PE-grade investment memos with citations

## System Architecture

```
INPUT: Company + Ticker (e.g., "Tesla", "TSLA")
         ↓
┌────────────────────────┐
│   RESEARCH PHASE       │ ← Loop if insufficient data
│                        │
│  • Web Researcher      │   (Tavily + Gemini Flash)
│    → Market trends     │
│    → News & signals    │
│                        │
│  • Financial Analyst   │   (ChromaDB + Python REPL)
│    → Metrics & ratios  │
│    → Risk factors      │
└────────────────────────┘
         ↓
┌────────────────────────┐
│  EVALUATION GATE       │
│  Data sufficient? Y/N  │
└────────────────────────┘
         ↓ (Yes)
┌────────────────────────┐
│    WRITE PHASE         │   (Gemini Pro)
│  7-Section Memo:       │
│  • Executive Summary   │
│  • Company Overview    │
│  • Market Analysis     │
│  • Financial Performance│
│  • Strategic Moat      │
│  • Risks & Mitigations │
│  • Conclusion          │
└────────────────────────┘
         ↓
OUTPUT: Investment Memo (Markdown)
```

## Key Features

### 1. **Iterative "Tree of Thoughts" Research**
- Performs initial broad search
- Identifies information gaps
- Conducts targeted follow-up searches
- Synthesizes findings across multiple iterations

### 2. **Python REPL for Accurate Calculations**
- LLMs are bad at math (might compute `100 / 3 = 33`)
- Python REPL ensures 100% accurate financial ratios
- Critical for debt-to-equity, margins, CAGR, P/E ratios

### 3. **Table-Aware Document Parsing**
- LlamaParse preserves table structure from 10-Ks and 10-Qs
- Essential for reading balance sheets and income statements
- Outputs tables as Markdown for LLM comprehension

### 4. **Citation Enforcement**
- Every factual claim includes `[Source: URL]`
- Enables verification by investment committees
- Prevents hallucination through grounding

### 5. **Cost-Optimized Model Selection**
- **Gemini 2.0 Flash Lite**: Quick synthesis tasks (~50% cheaper than GPT-4)
- **Gemini 2.5 Pro**: Final memo generation (highest quality)
- **Google Embeddings**: Vector database (free tier available)
- **Local Embeddings**: Option to use HuggingFace models (100% free, no API)

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set API Keys

**Required:**
```bash
export GOOGLE_API_KEY='your-google-api-key'
export TAVILY_API_KEY='your-tavily-api-key'
```

**Optional (for document ingestion):**
```bash
export LLAMA_CLOUD_API_KEY='your-llama-cloud-key'
```

Get API keys:
- **Google AI Studio**: https://aistudio.google.com/apikey (free tier available)
- **Tavily**: https://tavily.com (1000 free searches/month)
- **LlamaCloud**: https://cloud.llamaindex.ai (optional, for document parsing)

**Note:** You can also copy `.env.example` to `.env` and fill in your keys.

### 3. Run Analysis

**Interactive mode:**
```bash
python src/main.py
```

**Direct mode:**
```bash
python src/main.py "Tesla" "TSLA"
python src/main.py "Apple Inc" "AAPL" --output apple_memo.md
```

**Python API:**
```python
from graph import analyze_company

# Analyze company
result = analyze_company("Tesla", "TSLA")

# Get the memo
memo = result["memo_sections"]["full_draft"]
print(memo)

# Access research data
print(f"Financial data points: {len(result['financial_context'])}")
print(f"Market data points: {len(result['market_context'])}")
print(f"Research iterations: {result['research_iterations']}")
```

## Project Structure

```
Agentic-RAG/
├── src/
│   ├── state.py                    # AgentState definition (TypedDict)
│   ├── web_researcher.py           # Tavily web search + synthesis
│   ├── financial_analyst.py        # ChromaDB vector DB + Python REPL
│   ├── writer.py                   # Investment memo generation
│   ├── graph.py                    # LangGraph workflow (parallel execution)
│   ├── main.py                     # CLI interface
│   ├── ingest.py                   # Single document ingestion CLI
│   ├── batch_ingest.py             # Batch ingestion from directory/manifest
│   ├── auto_download.py            # SEC EDGAR auto-downloader
│   │
│   ├── test_state.py               # State management tests
│   ├── test_web_researcher.py      # Web research tests
│   ├── test_financial_analyst.py   # Financial analysis tests
│   ├── test_writer.py              # Memo generation tests
│   └── test_graph.py               # End-to-end workflow tests
│
├── data/
│   └── manifest.csv                # Sample batch ingestion manifest
│
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variables template
├── README.md                       # This file
└── Building an Agentic RAG Analyst.docx  # Original design document
```

## Testing

Run individual test suites:

```bash
# Test state management
python src/test_state.py

# Test web research (requires API keys)
python src/test_web_researcher.py

# Test financial analysis (requires GOOGLE_API_KEY)
python src/test_financial_analyst.py

# Test memo generation (requires GOOGLE_API_KEY)
python src/test_writer.py

# Test complete workflow (requires both API keys)
python src/test_graph.py
```

All tests work without API keys (graceful degradation) but show limited functionality.

## Data Collection & Ingestion

The system provides powerful tools for collecting and ingesting financial documents into the vector database.

### Automated SEC Filing Downloads

Download 10-K, 10-Q, 8-K, and DEF 14A filings directly from SEC EDGAR:

```bash
# Download single filing for a specific year
python src/auto_download.py TSLA --years 2023 --filing-type 10-K

# Download and automatically ingest into vector database
python src/auto_download.py TSLA --years 2023 --filing-type 10-K --ingest

# Download multiple years at once
python src/auto_download.py AAPL --years 2021 2022 2023 --filing-type 10-K

# Download quarterly reports
python src/auto_download.py TSLA --years 2023 --filing-type 10-Q --ingest

# Batch download from manifest file
python src/auto_download.py --manifest data/manifest.csv
```

**Features:**
- Respects SEC rate limits (10 requests/second)
- Automatic HTML to ingestible format conversion
- Downloads saved to `./sec-edgar-filings/`
- Processed files saved to `./data/`

### Manual Document Ingestion

Ingest individual financial documents (PDF, HTML, DOCX, PPTX, TXT):

```bash
# Ingest a 10-K filing
python src/ingest.py data/TSLA_2023_10K.pdf TSLA 2023

# Ingest a quarterly report with quarter specification
python src/ingest.py data/AAPL_Q3_2023.pdf AAPL 2023 --quarter Q3 --type 10-Q

# Interactive mode (prompts for inputs)
python src/ingest.py
```

### Batch Ingestion

Process multiple documents at once using directory auto-discovery or manifest file:

```bash
# Auto-discover and ingest all PDFs in a directory
# Expected filename format: TICKER_YEAR_TYPE.pdf (e.g., TSLA_2023_10K.pdf)
python src/batch_ingest.py data/financial_reports/

# Use a manifest file for precise control
python src/batch_ingest.py --manifest data/manifest.csv
```

**Manifest CSV Format** (`data/manifest.csv`):
```csv
file_path,ticker,year,quarter,doc_type
data/TSLA_2023_10K.pdf,TSLA,2023,,10-K
data/AAPL_Q3_2023.pdf,AAPL,2023,Q3,10-Q
data/NVDA_2023_10K.pdf,NVDA,2023,,10-K
```

**Batch Processing Features:**
- Success/failure tracking with summary reports
- Intelligent filename parsing
- Validation and error handling
- Interactive confirmation before processing

## Advanced Usage

### Adding Financial Documents to Vector DB

```python
from financial_analyst import ingest_financial_document

# Ingest a 10-K filing
ingest_financial_document(
    file_path="./data/TSLA_2023_10K.pdf",
    ticker="TSLA",
    year=2023,
    doc_type="10-K"
)

# Ingest a quarterly report
ingest_financial_document(
    file_path="./data/TSLA_Q3_2023_10Q.pdf",
    ticker="TSLA",
    year=2023,
    quarter="Q3",
    doc_type="10-Q"
)
```

### Streaming API for Real-Time Progress

```python
from graph import analyze_company_stream

# Stream events for UI integration
for event in analyze_company_stream("Tesla", "TSLA"):
    node = list(event.keys())[0]
    state = event[node]

    print(f"✓ Completed: {node}")
    print(f"  Iterations: {state.get('research_iterations', 0)}")

    if 'memo_sections' in state and 'full_draft' in state['memo_sections']:
        print(f"  Memo ready! ({len(state['memo_sections']['full_draft'])} chars)")
```

### Visualizing the Graph

```python
from graph import visualize_graph

# Generate Mermaid diagram
visualize_graph()

# Or get the graph structure programmatically
from graph import create_research_graph
app = create_research_graph()
print(app.get_graph().draw_mermaid())
```

## 🎓 How It Works

### 1. State Management (`state.py`)

The system uses a **TypedDict** with **operator.add** for append-only lists:

```python
class AgentState(TypedDict):
    company: str
    ticker: str
    financial_context: Annotated[List[str], operator.add]  # Append-only
    market_context: Annotated[List[str], operator.add]     # Append-only
    memo_sections: dict
    messages: Annotated[List[BaseMessage], operator.add]
    research_iterations: int
    is_data_sufficient: bool
```

**Why operator.add?** Without it, nodes would overwrite previous research. With it, eachnode appends to the growing context.

### 2. Web Researcher (`web_researcher.py`)

**Workflow:**
1. Formulates search query for company
2. Calls Tavily API with `search_depth="advanced"`
3. Retrieves 5 articles with full content
4. Uses Gemini Flash to synthesize into 300-500 word summary
5. Appends to `market_context[]`

**Iterative Version:**
- Phase 1: Broad initial search
- Phase 2: Financial results focus
- Phase 3: Competitive landscape
- Phase 4: Risk factors
- Final: Comprehensive synthesis

### 3. Financial Analyst (`financial_analyst.py`)

**Workflow:**
1. Queries ChromaDB vector database with semantic search
2. Filters by metadata: `{ticker: "TSLA", year: 2023}`
3. Retrieves relevant financial document chunks
4. Uses LLM to extract answers from context
5. For calculations, delegates to **Python REPL** (not LLM)
6. Appends to `financial_context[]`

**Document Ingestion:**
- Uses **LlamaParse** (vision-language model)
- Preserves table structure as Markdown
- Critical for financial statements

### 4. Writer (`writer.py`)

**Workflow:**
1. Combines all `financial_context` + `market_context`
2. Invokes **Gemini Pro** (higher quality than Flash)
3. Generates 7-section memo:
   - Executive Summary (Buy/Hold/Sell thesis)
   - Company Overview (business model, products)
   - Market Analysis (TAM, competition, trends)
   - Financial Performance (revenue, margins, metrics)
   - Strategic Moat (competitive advantages)
   - Risks & Mitigations (key risks, hedge strategies)
   - Conclusion (final recommendation + rationale)
4. Enforces `[Source: URL]` citations
5. Returns to `memo_sections["full_draft"]`

### 5. Graph Orchestration (`graph.py`)

**Nodes:**
- `start_research`: Initializes research phase
- `web_research`: Tavily web search + synthesis (runs in parallel)
- `financial_analysis`: ChromaDB query + Python REPL (runs in parallel)
- `evaluate`: Checks data sufficiency
- `write`: Generates final memo

**Edges:**
- `START → start_research`: Initialize
- `start_research → web_research`: Begin web research (parallel)
- `start_research → financial_analysis`: Begin financial analysis (parallel)
- `web_research → evaluate`: Sync point
- `financial_analysis → evaluate`: Sync point
- `evaluate → [conditional]`:
  - If insufficient → `start_research` (loop for more data)
  - If sufficient → `write` (proceed to memo generation)
- `write → END`: Done

**Quality Gate Logic:**
```python
has_financial = len(financial_context) > 0
has_market = len(market_context) > 0
iterations >= 1

is_sufficient = (has_financial and has_market and iterations >= 1)
                or iterations >= 3  # Force write after 3 loops
```

## Technology Stack

| Component | Technology | Why? |
|-----------|-----------|------|
| **Orchestration** | LangGraph | State graphs with conditional routing |
| **LLM (Synthesis)** | Gemini 2.0 Flash Lite | Fast, cheap, 1M token context |
| **LLM (Writing)** | Gemini 2.5 Pro | Highest quality for final output |
| **Web Search** | Tavily | Advanced search depth, reputable sources |
| **Vector DB** | ChromaDB | Local persistence, no external DB |
| **Embeddings** | Google Gemini / HuggingFace | Free tier API or local models |
| **Doc Parsing** | LlamaParse | Vision-language model for tables |
| **Calculations** | Python REPL | 100% accurate math (vs hallucinating LLM) |

## Cost Optimization

### Use Local Embeddings (Zero API Costs)

The system supports local HuggingFace embeddings, completely eliminating embedding API costs:

**How it works:**
- Downloads `sentence-transformers/all-MiniLM-L6-v2` model (~80MB)
- Runs embeddings locally on CPU (no GPU required)
- 100% free, no API calls, no rate limits
- Enabled by default in `financial_analyst.py`

**To use Google API embeddings instead:**
```python
from financial_analyst import create_vector_store

vectorstore = create_vector_store(use_local_embeddings=False)
```

### Model Selection by Task

Different tasks use different models to optimize cost and quality:

- **Web Research**: `gemini-2.0-flash-lite`
  - Fast synthesis of web search results
  - Low cost per token
  - 1M token context window

- **Financial Analysis**: `gemini-2.0-flash-lite`
  - Quick data extraction
  - Calculations delegated to Python REPL (free, accurate)

- **Final Memo**: `gemini-2.5-pro`
  - Highest quality for investor-grade output
  - Worth the cost for final deliverable

### API Usage Estimates

**Typical analysis (single company):**
- Web research: ~2-3 Flash Lite calls (~$0.01)
- Financial analysis: ~2-3 Flash Lite calls (~$0.01)
- Final memo: 1 Pro call (~$0.03)
- **Total: ~$0.05 per analysis**

**Free tier limits:**
- Gemini Flash Lite: 15 requests/min, 1500/day
- Gemini Pro: 2 requests/min, 50/day
- Tavily: 1000 searches/month
- **You can run ~50 analyses per day on free tier**

## Performance Optimizations

### Parallel Research Execution

Web Researcher and Financial Analyst run concurrently for ~50% speed improvement:

```python
# In graph.py - parallel execution
app.add_node("web_research", iterative_web_research_node)
app.add_node("financial_analysis", financial_analyst_node)

# Both nodes read from start_research and run in parallel
app.add_edge("start_research", "web_research")
app.add_edge("start_research", "financial_analysis")
```

**Benefits:**
- Cut research time from ~60s to ~30s
- No data conflicts (operator.add handles merging)
- Automatic synchronization before evaluation

### Streaming for Real-Time Progress

Monitor analysis progress in real-time for UI integration:

```python
from graph import analyze_company_stream

for event in analyze_company_stream("Tesla", "TSLA"):
    node = list(event.keys())[0]
    state = event[node]

    print(f"✓ Completed: {node}")
    print(f"  Iterations: {state.get('research_iterations', 0)}")

    if 'memo_sections' in state and 'full_draft' in state['memo_sections']:
        print(f"  Memo ready! ({len(state['memo_sections']['full_draft'])} chars)")
```

### Caching Strategies

**Vector Database Persistence:**
- ChromaDB persists to `./chroma_db/`
- Documents ingested once, queried unlimited times
- No re-parsing of PDFs after initial ingestion

**Document Pre-Processing:**
- Batch ingest documents during off-hours
- Analysis queries cached embeddings (fast retrieval)
- LlamaParse results cached by document hash

## Example Output

Here's what the system generates for Tesla (TSLA):

```markdown

# Investment Memo: Tesla, Inc. (TSLA)

## Executive Summary

**Recommendation: HOLD**

Tesla maintains a dominant position in the EV market with strong revenue growth
(+18.8% YoY to $96.77B in 2023) [Source: SEC 10-K Filing], but faces increasing
competitive pressure and valuation concerns (P/E: 294.19) [Source: Yahoo Finance].
The company's vertical integration and charging network provide a strategic moat,
but execution risks on new products (Cybertruck, Optimus) warrant a cautious stance...

## Company Overview

Tesla, Inc. is the world's leading electric vehicle manufacturer and clean energy
company. The business model encompasses:
- **Automotive**: Model S, 3, X, Y production and sales
- **Energy**: Solar panels, Powerwall, Megapack storage
- **Services**: Supercharger network, insurance, FSD subscriptions

[continues with detailed analysis...]
```

## Configuration

### Adjusting Research Iterations

Edit `src/graph.py`:
```python
MAX_ITERATIONS = 5  # Default is 3
```

### Customizing Memo Sections

Edit `src/writer.py`:
```python
MEMO_SECTIONS = {
    "executive_summary": "...",
    "your_custom_section": "Your section description",
    # ... add more sections
}
```

### Changing Model Temperature

Edit `src/writer.py`:
```python
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0.5  # Default is 0.3 (more creative)
)
```

## Troubleshooting

### ImportError: No module named 'langchain_experimental'

```bash
pip install langchain-experimental
```

### Tavily API returns dictionary, not list

This is expected behavior with the new `langchain-tavily` package. The code handles both formats automatically.

### Google API quota exceeded (429 error)

Free tier limits:
- **Gemini 2.0 Flash Lite**: 15 requests/minute, 1500/day
- **Gemini 2.5 Pro**: 2 requests/minute, 50/day
- **Embeddings**: 1500 requests/day

Solutions:
- Upgrade to paid tier or space out requests
- Use local embeddings (HuggingFace) to eliminate embedding API calls
- Reduce MAX_ITERATIONS in `src/graph.py` to use fewer API calls

### ChromaDB returns no results

The vector database is initially empty. You have several options:

1. **Let the system work with web data only** (Financial Analyst will report "No data")
2. **Download and ingest SEC filings automatically:**
   ```bash
   python src/auto_download.py TSLA --years 2023 --filing-type 10-K --ingest
   ```
3. **Ingest existing documents:**
   ```bash
   python src/ingest.py data/TSLA_2023_10K.pdf TSLA 2023
   ```
4. **Batch ingest multiple documents:**
   ```bash
   python src/batch_ingest.py data/financial_reports/
   ```

## Documentation

- **README.md**: Complete setup and usage guide (this file)
- **CHANGELOG.md**: Project history and recent updates
- **.env.example**: Environment variables template
- **Building an Agentic RAG Analyst.docx**: Original design document
- **Code Documentation**: Comprehensive docstrings in all modules
- **Test Suite**: See `src/test_*.py` for usage examples and test cases

## Contributing

This is an educational project implementing the concepts from "Building an Agentic RAG Analyst".

**Recently Completed:**
- [x] Parallel execution of Web + Financial researchers
- [x] SEC EDGAR auto-downloader
- [x] Batch ingestion system
- [x] Local embeddings support

**Potential Improvements:**
- [ ] LLM-based data sufficiency evaluation (instead of heuristic)
- [ ] Multi-company comparison mode
- [ ] Export to PDF with charts/graphs
- [ ] Technical analysis node (price charts, volume)
- [ ] Sentiment analysis from earnings calls
- [ ] Risk scoring model
- [ ] Portfolio optimization
- [ ] API rate limiting and retry logic
- [ ] Caching layer for API responses

## License

See the original document for licensing details.

## Acknowledgments

- **LangGraph**: For state graph orchestration framework
- **Tavily**: For advanced web search API
- **Google AI**: For Gemini models and embeddings
- **LlamaIndex**: For LlamaParse document parsing

---

**Built with ❤️ following the "Building an Agentic RAG Analyst" guide**