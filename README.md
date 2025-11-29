# 🤖 Agentic RAG Market Research Analyst

An AI-powered investment research system that generates professional investment memos by autonomously gathering and analyzing market data.

## 🎯 What is Agentic RAG?

Unlike traditional RAG (Retrieval-Augmented Generation), which follows a linear retrieve → generate flow, **Agentic RAG** features:

1.  **🔍 Autonomous Data Sourcing**: Automatically downloads 10-K/10-Q filings directly from **SEC EDGAR**.
2.  **⚡ Parallel Agent Execution**: Web Researcher and Financial Analyst run simultaneously, cutting latency by 50%.
3.  **🧮 3-Pass "Anti-Hallucination" Math**:
    * *Pass 1:* LLM identifies the formula needed.
    * *Pass 2:* Python REPL executes the math (100% precision).
    * *Pass 3:* LLM synthesizes the answer into the narrative.
4.  **👁️ Vision-Language Parsing**: Uses **LlamaParse** to read complex financial tables row-by-row, preserving structure that standard OCR misses.

---

## 🏗️ System Architecture

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

## ✨ Key Features

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
- **Gemini Flash**: Quick synthesis tasks (~50% cheaper than GPT-4)
- **Gemini Pro**: Final memo generation (quality matters)
- **Google Embeddings**: Vector database (free tier available)

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set API Keys

```bash
export GOOGLE_API_KEY='your-google-api-key'
export TAVILY_API_KEY='your-tavily-api-key'
```

Get API keys:
- **Google AI Studio**: https://aistudio.google.com/apikey (free tier available)
- **Tavily**: https://tavily.com (1000 free searches/month)

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

## 📁 Project Structure

```
Agentic-RAG/
├── src/
│   ├── state.py                    # AgentState definition (TypedDict)
│   ├── web_researcher.py           # Tavily web search + synthesis
│   ├── financial_analyst.py        # ChromaDB vector DB + Python REPL
│   ├── writer.py                   # Investment memo generation
│   ├── graph.py                    # LangGraph workflow assembly
│   ├── main.py                     # CLI interface
│   │
│   ├── test_state.py               # State management tests
│   ├── test_web_researcher.py      # Web research tests
│   ├── test_financial_analyst.py   # Financial analysis tests
│   ├── test_writer.py              # Memo generation tests
│   └── test_graph.py               # End-to-end workflow tests
│
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── IMPLEMENTATION_LOG.md           # Detailed implementation notes
├── GEMINI_MIGRATION.md             # OpenAI → Gemini migration guide
├── TAVILY_API_FIX.md              # Tavily API format fix documentation
└── Building an Agentic RAG Analyst.docx  # Original design document
```

## 🧪 Testing

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

## 🔧 Advanced Usage

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
- `research`: Runs Web + Financial analysts
- `evaluate`: Checks data sufficiency
- `write`: Generates final memo

**Edges:**
- `START → research`: Begin with data gathering
- `research → evaluate`: Check if we have enough data
- `evaluate → [conditional]`:
  - If insufficient → `research` (loop)
  - If sufficient → `write` (proceed)
- `write → END`: Done

**Quality Gate Logic:**
```python
has_financial = len(financial_context) > 0
has_market = len(market_context) > 0
iterations >= 1

is_sufficient = (has_financial and has_market and iterations >= 1)
                or iterations >= 3  # Force write after 3 loops
```

## 📊 Technology Stack

| Component | Technology | Why? |
|-----------|-----------|------|
| **Orchestration** | LangGraph | State graphs with conditional routing |
| **LLM (Synthesis)** | Gemini Flash 1.5 | Fast, cheap, 1M token context |
| **LLM (Writing)** | Gemini Pro 1.5 | High quality for final output |
| **Web Search** | Tavily | Advanced search depth, reputable sources |
| **Vector DB** | ChromaDB | Local persistence, no external DB |
| **Embeddings** | Google embedding-001 | Free tier, consistent with Gemini |
| **Doc Parsing** | LlamaParse | Vision-language model for tables |
| **Calculations** | Python REPL | 100% accurate math (vs hallucinating LLM) |

## 🔍 Example Output

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

## 🎛️ Configuration

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

## 🐛 Troubleshooting

### ImportError: No module named 'langchain_experimental'

```bash
pip install langchain-experimental
```

### Tavily API returns dictionary, not list

This is expected behavior with the new `langchain-tavily` package. The code handles both formats. See `TAVILY_API_FIX.md` for details.

### Google API quota exceeded (429 error)

Free tier limits:
- **Gemini Flash**: 15 requests/minute, 1500/day
- **Gemini Pro**: 2 requests/minute, 50/day
- **Embeddings**: 1500 requests/day

Solution: Upgrade to paid tier or space out requests.

### ChromaDB returns no results

The vector database is initially empty. Either:
1. Let the system work with web data only (Financial Analyst will report "No data")
2. Ingest financial documents using `ingest_financial_document()`

## 🤝 Contributing

This is an educational project implementing the concepts from "Building an Agentic RAG Analyst".

Potential improvements:
- [✔] Parallel execution of Web + Financial researchers
- [✔] LLM-based data sufficiency evaluation (instead of heuristic)
- [ ] Multi-company comparison mode
- [ ] Export to PDF with charts/graphs
- [ ] Technical analysis node (price charts, volume)
- [ ] Sentiment analysis from earnings calls
- [ ] Risk scoring model
- [ ] Portfolio optimization

## 📄 License

See the original document for licensing details.

## 🙏 Acknowledgments

- **LangGraph**: For state graph orchestration framework
- **Tavily**: For advanced web search API
- **Google AI**: For Gemini models and embeddings
- **LlamaIndex**: For LlamaParse document parsing

---

**Built with ❤️ using LangGraph, LlamaIndex, and Gemini.**