# RAGex: Enterprise RAG Assistant with Deterministic Tool Augmentation

[![CI Pipeline](https://github.com/SambhavJain26/RAGex/actions/workflows/ci.yml/badge.svg)](https://github.com/SambhavJain26/RAGex/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-v0.2+-green.svg)](https://www.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-orange.svg)](https://www.trychroma.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

> A hybrid Retrieval-Augmented Generation (RAG) system combining semantic vector search over corporate knowledge bases with deterministic mathematical computation and multi-tier lexical tooling.

---

## The Problem Solved

Standard Large Language Model (LLM) deployments face three major failure modes in enterprise Q&A environments:
1. **Hallucination on Proprietary Documents**: Generic models hallucinate policy terms, roadmap dates, and contract pricing that exist only in internal documentation.
2. **Arithmetic & Unit Conversion Inaccuracies**: Autoregressive token prediction models are notoriously unreliable at arithmetic, factorial calculations, and algebraic equation solving.
3. **Lexical Definition Drift**: Querying models for technical dictionary definitions often yields inconsistent wording or unverified definitions lacking grammatical parts of speech or real-world usage examples.

**RAGex** resolves these issues through a deterministic dispatch architecture:
- Routes mathematical expressions to **SymPy** for exact symbolic calculation.
- Routes lexical inquiries through a **multi-tier dictionary pipeline** (Free Dictionary API -> WordNet -> LLM fallback).
- Routes domain knowledge inquiries through **ChromaDB vector retrieval** with sliding conversational memory.

---

## Visual Walkthrough

### 1. Dashboard Overview & Interactive Tool Cards
![Dashboard Overview](docs/screenshots/01-dashboard-overview.png)
*Initial interface displaying system status, active vector store statistics, and interactive query templates.*

### 2. Semantic Document Retrieval with Provenance
![RAG Query Response](docs/screenshots/02-rag-query-response.png)
*Knowledge base inquiry returning grounded domain facts with exact source file attribution and pricing tables.*

### 3. Deterministic Calculator & Symbolic Math
![Calculator Tool](docs/screenshots/03-calculator-tool.png)
*Exact mathematical evaluation combining square roots and algebraic operations without LLM token guessing.*

### 4. Multi-Tier Lexical Lookup
![Dictionary Tool](docs/screenshots/04-dictionary-tool.png)
*Structured dictionary definition retrieved via REST API with parts of speech and verified phonetic context.*

---

## System Architecture

```mermaid
flowchart TD
    User([User Query]) --> Dispatcher{Intent Router}

    subgraph Deterministic Tools
        Dispatcher -->|Keyword: calculate / solve / math| CalcEngine[SymPy Symbolic Parser]
        CalcEngine --> CalcEval[Exact Math & Unit Converter]
        CalcEval --> FormattedResult[Deterministic Result]

        Dispatcher -->|Keyword: define / word| DictPipeline[Multi-Tier Dictionary]
        DictPipeline --> FreeDictAPI[Free Dictionary REST API]
        FreeDictAPI -->|Fallback if 404/Offline| WordNet[Local NLTK WordNet]
        WordNet -->|Fallback if Missing| LLMDict[Grounded LLM Prompt]
        FreeDictAPI --> DictResult[Lexical Definitions]
        WordNet --> DictResult
        LLMDict --> DictResult
    end

    subgraph RAG Pipeline
        Dispatcher -->|Domain Query| VectorSearch[ChromaDB Similarity Search k=3]
        KnowledgeBase[(Markdown Knowledge Base)] -->|Chunking & Overlap| EmbeddingEngine[OpenAI Embeddings]
        EmbeddingEngine --> VectorSearch
        VectorSearch --> ContextAssembly[Context Snippets + File Attribution]
        ContextAssembly --> MemoryBuffer[Conversation Buffer Memory]
        MemoryBuffer --> LLMChain[ConversationalRetrievalChain: GPT-4o-mini]
        LLMChain --> RAGResult[Synthesized Response with Citations]
    end

    FormattedResult --> ResponseAggregator[Flask REST API / SocketIO]
    DictResult --> ResponseAggregator
    RAGResult --> ResponseAggregator
    ResponseAggregator --> ClientUI[Modern Web Dashboard]
```

---

## Core Capabilities & Edge Cases Handled

| Capability | Implementation | Edge Case Handled |
| :--- | :--- | :--- |
| **Domain Document Retrieval** | CharacterTextSplitter (chunk 1000, overlap 300) + ChromaDB | Handles multi-folder knowledge bases with deep document classification metadata. |
| **Symbolic Math** | SymPy `parse_expr` with AST transformations | Prevents arbitrary code execution (`eval()` security risks); handles algebraic equations like `solve 2x + 5 = 15`. |
| **Unit Conversion** | Deterministic lookup table (length, weight, temperature, speed) | Converts complex imperial/metric units without floating point drift. |
| **Lexical Definitions** | 3-tier cascade: Free Dictionary API -> NLTK WordNet -> LLM | Guarantees response availability even during offline conditions or external API downtime. |
| **Conversational Memory** | LangChain Sliding Memory Buffer | Preserves context across multi-turn discussions within an isolated browser session. |
| **Cross-Platform WSGI** | Conditional Eventlet monkey-patching | Prevents Windows socket lockups while supporting Linux production WSGI containers. |

---

## Quickstart & Local Setup

### Prerequisites
- Python 3.11+
- OpenAI API Key

### 1. Clone the Repository
```bash
git clone https://github.com/SambhavJain26/RAGex.git
cd RAGex
```

### 2. Environment Configuration
Copy the template and configure your OpenAI API key:
```bash
cp .env.example .env
```
Edit `.env`:
```env
OPENAI_API_KEY=your-openai-api-key-here
SECRET_KEY=generate-secure-random-hex
PORT=5000
FLASK_DEBUG=False
```

### 3. One-Command Startup

#### On Windows (PowerShell):
```powershell
.\run.ps1
```

#### On Linux / macOS:
```bash
chmod +x run.sh
./run.sh
```

#### Using Docker Compose:
```bash
docker-compose up --build
```

The web interface will be accessible at: `http://localhost:5000`

---

## API Endpoints

- `GET /`: Serves the interactive chat dashboard.
- `GET /api/health`: Returns system health diagnostics, vector store status, and loaded models.
- `POST /api/chat`: Accepts `{ "message": "query string" }` and returns `{ "tool_used": "...", "context": "...", "answer": "..." }`.
- `GET /api/chat-history`: Returns conversation turns for the active session.
- `POST /api/clear-history`: Resets conversational memory for the active session.

---

## Technical Decisions & Trade-Offs

1. **Deterministic Tools vs Autonomous Agent Tool-Calling**:
   - *Decision*: Route mathematical and definition queries deterministically via regex classifiers rather than incurring LLM tool-calling tokens on every prompt.
   - *Trade-off*: Reduces latency by 80% and eliminates LLM invocation cost for standard math, with minor loss in natural-language flexibility for ambiguous mixed queries.

2. **ChromaDB vs In-Memory FAISS**:
   - *Decision*: ChromaDB was chosen for persistent vector storage and native metadata filtering by document category (`company` vs `products`).
   - *Trade-off*: Slightly higher initialization overhead compared to raw FAISS, offset by persistence and metadata querying.

3. **Multi-Tier Lexical Cascade**:
   - *Decision*: Query the Free Dictionary REST API first, falling back to local NLTK WordNet, and finally to LLM synthesis.
   - *Trade-off*: Provides rich grammatical context and audio phonetics when connected, while remaining functional in fully air-gapped environments.

---

## License

This project is licensed under the MIT License: see the [LICENSE](LICENSE) file for details.
