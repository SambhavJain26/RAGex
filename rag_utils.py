import os
import glob
import re
import logging
import math
import warnings
from typing import Tuple, List, Dict, Any, Optional

warnings.filterwarnings("ignore")
import requests
from dotenv import load_dotenv
import nltk
from nltk.corpus import wordnet as wn
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
from sympy import symbols, solve, Eq

load_dotenv(override=True)

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

try:
    from langchain_classic.memory import ConversationBufferMemory
    from langchain_classic.chains import ConversationalRetrievalChain
except ImportError:
    try:
        from langchain.memory import ConversationBufferMemory
        from langchain.chains import ConversationalRetrievalChain
    except ImportError:
        ConversationBufferMemory = None
        ConversationalRetrievalChain = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

MODEL = os.getenv("RAG_MODEL", "gpt-4o-mini")
DB_DIR = os.getenv("VECTOR_DB_DIR", "vector_db")

# Ensure NLTK corpora are downloaded
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')


class RAGChatbot:
    """Enterprise RAG Assistant combining vector search with deterministic tooling."""

    def __init__(self, knowledge_base_dir: str = "knowledge-base"):
        self.knowledge_base_dir = knowledge_base_dir
        self.documents: List[Document] = []
        self.chat_history: List[Dict[str, str]] = []
        
        self.load_documents()
        self.setup_vectorstore()
        self.setup_rag_chain()

    def load_documents(self) -> None:
        """Scan and load all markdown files from the knowledge base directory."""
        logger.info(f"Loading documents from {self.knowledge_base_dir}...")
        self.documents = []
        folders = glob.glob(os.path.join(self.knowledge_base_dir, "*"))
        
        for folder in folders:
            if not os.path.isdir(folder):
                continue
            doc_type = os.path.basename(folder)
            loader = DirectoryLoader(
                folder,
                glob="**/*.md",
                loader_cls=TextLoader,
                loader_kwargs={'encoding': 'utf-8'}
            )
            for doc in loader.load():
                doc.metadata["doc_type"] = doc_type
                self.documents.append(doc)
                
        logger.info(f"Loaded {len(self.documents)} knowledge base documents.")

    def setup_vectorstore(self) -> None:
        """Initialize embeddings, chunk documents, and persist Chroma vector store."""
        logger.info("Configuring ChromaDB vector store...")
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
        chunks = splitter.split_documents(self.documents) if self.documents else []
        
        self.embeddings = OpenAIEmbeddings()
        
        if os.path.exists(DB_DIR):
            try:
                Chroma(persist_directory=DB_DIR, embedding_function=self.embeddings).delete_collection()
            except Exception as e:
                logger.warning(f"Could not reset existing collection: {e}")
        
        if chunks:
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=DB_DIR
            )
        else:
            self.vectorstore = Chroma(
                embedding_function=self.embeddings,
                persist_directory=DB_DIR
            )
            
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        logger.info("ChromaDB vector store setup complete.")

    def setup_rag_chain(self) -> None:
        """Configure conversational retrieval chain with sliding memory buffer."""
        logger.info("Initializing conversational retrieval chain...")
        self.llm = ChatOpenAI(model_name=MODEL, temperature=0.7)
        
        if ConversationBufferMemory is not None and ConversationalRetrievalChain is not None:
            self.memory = ConversationBufferMemory(
                memory_key='chat_history',
                return_messages=True,
                output_key='answer'
            )
            self.rag_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.retriever,
                memory=self.memory,
                return_source_documents=True,
            )
            logger.info("LangChain ConversationalRetrievalChain setup complete.")
        else:
            self.memory = None
            self.rag_chain = None
            logger.warning("ConversationalRetrievalChain not available; using fallback retrieval.")

    def get_context_str_and_docs(self, query: str) -> Tuple[str, List[Document]]:
        """Retrieve top semantic documents and format context snippets for attribution."""
        if hasattr(self.retriever, "invoke"):
            docs = self.retriever.invoke(query)
        else:
            docs = self.retriever.get_relevant_documents(query)

        context_snippets = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "Unknown")
            doc_type = doc.metadata.get("doc_type", "Unknown")
            snippet = f"[{i+1}] Source: {doc_type}/{os.path.basename(source)}\n{doc.page_content[:200]}..."
            context_snippets.append(snippet)
            
        return "\n\n".join(context_snippets), docs

    def calculator_tool(self, expr: str) -> str:
        """Deterministic mathematical engine supporting equations, units, and expressions."""
        original_expr = expr
        expr = expr.lower().strip()

        # 1. Equation solving mode
        if "solve" in expr or "=" in expr:
            try:
                equation_match = re.search(r'solve\s+(.*?)(?=$)', expr) if "solve" in expr else None
                equation_str = equation_match.group(1).strip() if equation_match else expr

                if "=" in equation_str:
                    left, right = equation_str.split("=", 1)
                    equation_str = f"{left}-({right})"

                x = symbols('x')
                equation = parse_expr(
                    equation_str,
                    transformations=(standard_transformations + (implicit_multiplication_application,))
                )
                solution = solve(equation, x)

                if solution:
                    if len(solution) == 1:
                        return f"x = {solution[0]}"
                    else:
                        return f"Solutions: {', '.join([f'x = {sol}' for sol in solution])}"
                else:
                    return "No solutions found."
            except Exception as e:
                logger.error(f"Equation solving fallback: {e}")

        # 2. Factorial parsing
        if "!" in expr and "!=" not in expr:
            try:
                num_match = re.search(r'(\d+)!', expr)
                if num_match:
                    num = int(num_match.group(1))
                    return str(math.factorial(num))
            except Exception as e:
                return f"Factorial error: {e}"

        # 3. Unit conversion tables
        unit_conversion_match = re.search(r'(\d+(?:\.\d+)?)\s+(\w+)\s+to\s+(\w+)', expr)
        if unit_conversion_match:
            value, from_unit, to_unit = unit_conversion_match.groups()
            conversions = {
                ('feet', 'meters'): lambda v: v * 0.3048,
                ('meters', 'feet'): lambda v: v * 3.28084,
                ('inches', 'cm'): lambda v: v * 2.54,
                ('cm', 'inches'): lambda v: v * 0.393701,
                ('miles', 'km'): lambda v: v * 1.60934,
                ('km', 'miles'): lambda v: v * 0.621371,
                ('pounds', 'kg'): lambda v: v * 0.453592,
                ('kg', 'pounds'): lambda v: v * 2.20462,
                ('celsius', 'fahrenheit'): lambda v: v * 9/5 + 32,
                ('fahrenheit', 'celsius'): lambda v: (v - 32) * 5/9,
                ('liters', 'gallons'): lambda v: v * 0.264172,
                ('gallons', 'liters'): lambda v: v * 3.78541,
                ('ounces', 'grams'): lambda v: v * 28.3495,
                ('grams', 'ounces'): lambda v: v * 0.03527396,
                ('yards', 'meters'): lambda v: v * 0.9144,
                ('meters', 'yards'): lambda v: v * 1.09361,
                ('mph', 'kph'): lambda v: v * 1.60934,
                ('kph', 'mph'): lambda v: v * 0.621371,
                ('acres', 'hectares'): lambda v: v * 0.404686,
                ('hectares', 'acres'): lambda v: v * 2.47105,
            }
            if (from_unit, to_unit) in conversions:
                res = conversions[(from_unit, to_unit)](float(value))
                return f"{value} {from_unit} = {res:.4f} {to_unit}"
            else:
                return f"Conversion from {from_unit} to {to_unit} is not supported."

        # 4. Canonical substitutions
        replacements = {
            "multiplied by": "*",
            "times": "*",
            "plus": "+",
            "minus": "-",
            "divided by": "/",
            "over": "/",
            "cosec": "1/sin",
            "sec": "1/cos",
            "cot": "1/tan",
            "arcsin": "asin",
            "arccos": "acos",
            "arctan": "atan",
            "÷": "/",
            "mod": "%",
            "sqrt": "sqrt",
            "squared": "**2",
            "cubed": "**3",
            "to the power of": "**",
            "^": "**",
            "percent": "/100",
            "%": "/100",
            "pi": "pi",
            "sin": "sin",
            "cos": "cos",
            "tan": "tan",
            "log": "log",
            "ln": "ln",
        }

        expr = expr.replace(" x ", " * ")
        expr = re.sub(r'(\d+)x(\d+)', r'\1*\2', expr)

        for word, sym in replacements.items():
            expr = expr.replace(word, sym)

        transformations = standard_transformations + (implicit_multiplication_application,)

        try:
            parsed_expr = parse_expr(expr, transformations=transformations)
            result = parsed_expr.evalf()

            if result.is_integer and float(result) < 10**12:
                return str(int(result))
            else:
                result_float = float(result)
                if abs(result_float) < 1e-10:
                    return "0"
                elif abs(result_float) > 1e10 or abs(result_float) < 1e-6:
                    return f"{result_float:.6e}"
                else:
                    formatted = f"{result_float:.6f}".rstrip('0').rstrip('.')
                    return formatted if formatted else "0"

        except Exception as e:
            logger.error(f"SymPy evaluation error: {e}")
            # Safe AST fallback dictionary
            try:
                safe_dict = {
                    "math": math,
                    "sin": math.sin,
                    "cos": math.cos,
                    "tan": math.tan,
                    "asin": math.asin,
                    "acos": math.acos,
                    "atan": math.atan,
                    "sqrt": math.sqrt,
                    "pi": math.pi,
                    "e": math.e,
                    "log": math.log10,
                    "ln": math.log,
                    "cosec": lambda x: 1/math.sin(x),
                    "sec": lambda x: 1/math.cos(x),
                    "cot": lambda x: 1/math.tan(x),
                }
                eval_expr = re.sub(r'(\d+)x(\d+)', r'\1*\2', expr)
                result = eval(eval_expr, {"__builtins__": None}, safe_dict)

                if isinstance(result, float):
                    if abs(result) < 1e-10:
                        return "0"
                    elif abs(result - round(result)) < 1e-10:
                        return str(int(result))
                    else:
                        return f"{result:.6f}".rstrip('0').rstrip('.')
                return str(result)
            except Exception as e2:
                return f"Could not calculate: '{original_expr}'. Error: {e2}"

    def dictionary_tool(self, term: str) -> str:
        """Multi-tier lexical lookup: Free Dictionary API first, WordNet fallback second."""
        term = term.strip().lower()
        term = re.sub(r'[^\w\s]', '', term)

        results = []
        definition_found = False

        # Tier 1: Free Dictionary REST API
        try:
            logger.info(f"Looking up '{term}' in Free Dictionary API...")
            url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{term}"
            response = requests.get(url, timeout=5)

            if response.status_code == 200:
                data = response.json()
                if data and isinstance(data, list) and len(data) > 0:
                    entry = data[0]
                    api_results = []
                    for meaning in entry.get('meanings', [])[:3]:
                        part_of_speech = meaning.get('partOfSpeech', '')
                        definitions = meaning.get('definitions', [])
                        if definitions:
                            definition = definitions[0].get('definition', '')
                            if definition:
                                api_results.append(f"{part_of_speech}: {definition}")
                                example = definitions[0].get('example', '')
                                if example:
                                    api_results.append(f"   Example: {example}")

                    if api_results:
                        results.append("\n".join(api_results))
                        definition_found = True
        except Exception as e:
            logger.error(f"Free Dictionary API lookup failed: {e}")

        # Tier 2: Local NLTK WordNet
        if not definition_found:
            try:
                synsets = wn.synsets(term)
                if synsets:
                    lines = []
                    pos_mapping = {
                        'n': 'noun',
                        'v': 'verb',
                        'a': 'adjective',
                        's': 'adjective satellite',
                        'r': 'adverb'
                    }
                    for syn in synsets[:3]:
                        pos_full = pos_mapping.get(syn.pos(), syn.pos())
                        lines.append(f"{pos_full}: {syn.definition()}")
                    if lines:
                        results.append("\n".join(lines))
                        definition_found = True
            except Exception as e:
                logger.error(f"WordNet lookup failed: {e}")

        # Tier 3: LLM lexical fallback
        if not definition_found:
            try:
                prompt = f"Provide a concise dictionary definition for '{term}'. Format with part of speech and clear definition."
                res = self.llm.invoke(prompt)
                if res and len(res.content) > 10:
                    results.append(res.content)
                    definition_found = True
            except Exception as e:
                logger.error(f"LLM dictionary fallback failed: {e}")

        if results:
            return "\n\n".join(results)
        return f"No definition found for '{term}'."

    def process_message(self, message: str) -> Dict[str, str]:
        """Dispatch query to appropriate engine (Calculator, Dictionary, or RAG)."""
        lower = message.lower()
        self.chat_history.append({"role": "user", "content": message})

        if "calculate" in lower:
            tool_used = "Calculator Tool"
            expr = re.sub(r"(?i).*calculate\s*", "", message).strip()
            answer = self.calculator_tool(expr)
            context_snippets = "Deterministic calculation engine invoked (no vector context required)."
        elif "define" in lower:
            tool_used = "Dictionary Tool"
            term = re.sub(r"(?i).*define\s*", "", message).strip(" ?.")
            answer = self.dictionary_tool(term)
            context_snippets = "Lexical dictionary engine invoked (no vector context required)."
        else:
            tool_used = "RAG Pipeline"
            context_snippets, _ = self.get_context_str_and_docs(message)

            if self.rag_chain is not None:
                result = self.rag_chain.invoke({"question": message})
                answer = result.get("answer", "")
            else:
                # Direct LLM invocation with retrieved context
                system_prompt = (
                    "You are an expert AI assistant answering questions using the provided context.\n\n"
                    f"Context:\n{context_snippets}\n\n"
                    f"Question: {message}"
                )
                response = self.llm.invoke(system_prompt)
                answer = response.content

        self.chat_history.append({"role": "assistant", "content": answer})

        return {
            "tool_used": tool_used,
            "context": context_snippets,
            "answer": answer
        }


def initialize_chatbot() -> RAGChatbot:
    """Factory function for initializing the RAGChatbot instance."""
    return RAGChatbot()
