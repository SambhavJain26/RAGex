# AGENTS.md: Architectural Constraints & System Operational Specifications

## 1. System Mission & Operational Scope
RAGex is an enterprise hybrid Question-Answering system combining dense semantic vector retrieval (ChromaDB + OpenAI embeddings) with deterministic execution tools (SymPy mathematical parser, unit converter, and multi-tier lexical dictionary).

The objective is to eliminate generic LLM hallucinations on domain documentation while avoiding arithmetic inaccuracy and non-deterministic lexical lookups.

---

## 2. Architecture & Invariant Rules

### Invariant 1: Deterministic Math Execution (Zero Unsafe Eval)
- All arithmetic, factorial, trigonometric, and equation-solving queries must be processed via symbolic AST transformations using SymPy.
- Never pass unvalidated user strings into raw Python `eval()` without strict sandbox scoping (`{"__builtins__": None}`).
- Any numerical result must be returned with clean formatting (integers truncated cleanly, float decimals stripped of trailing zeroes).

### Invariant 2: Multi-Tier Lexical Fallback
- Lexical queries routed to the Dictionary tool must execute with high resilience:
  1. Free Dictionary REST API (primary structured source with definitions and examples).
  2. Local NLTK WordNet synset mapping (offline deterministic fallback).
  3. Grounded LLM dictionary synthesis (last-resort prompt with strict formatting).

### Invariant 3: Semantic Grounding & Attribution
- All domain knowledge queries must pass through ChromaDB similarity search (k=3).
- Context snippets must include document type and file source attribution so users can verify provenance.
- In-memory conversation buffer must persist context across multi-turn interactions within a session without leaking state between concurrent sessions.

### Invariant 4: Cross-Platform Deployment Resilience
- Eventlet monkey-patching must remain strictly conditional to avoid Windows socket/thread conflicts during development.
- Production deployments on Linux must support containerized WSGI workers via Gunicorn and Eventlet.

---

## 3. Style & Writing Standards (STRICT)
- **Em Dash Ban**: Em dashes (Unicode \u2014) and en dashes (Unicode \u2013) are strictly prohibited across all code comments, documentation, and commits. Use colons, parentheses, commas, or standard single ASCII hyphens (-) exclusively.
- **Engineer-Led Framing**: All documentation and commit histories reflect engineering architecture, technical trade-offs, and systematic implementation.

---

## 4. Verification & Testing Gates
Before committing changes to this repository, verify the following gates:
1. **Syntax Check**: `python -m py_compile app.py rag_utils.py` exits with 0.
2. **Calculator Test**: Expressions like `3x5`, `sqrt(144) + 25`, and unit conversions evaluate accurately.
3. **Health Check**: `GET /api/health` returns status `healthy` with indexed document counts.
4. **Git Hygiene**: No byte-compiled files (`*.pyc`), `__pycache__`, or local `.env` files committed.
