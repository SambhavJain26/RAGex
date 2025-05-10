import os
import glob
import re
import logging
import math
import requests
from typing import Tuple, List, Dict, Any
import nltk
from nltk.corpus import wordnet as wn

# Install required packages
try:
    import sympy
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "sympy"])

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

MODEL = "gpt-4o-mini"
DB_DIR = "vector_db"

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class RAGChatbot:
    def __init__(self):
        self.load_documents()
        self.setup_vectorstore()
        self.setup_rag_chain()
        self.chat_history = []

    def load_documents(self):
        logger.info("Loading documents...")
        self.documents = []
        folders = glob.glob("knowledge-base/*")
        for folder in folders:
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
        logger.info(f"Loaded {len(self.documents)} documents")

    def setup_vectorstore(self):
        logger.info("Setting up vector store...")
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
        chunks = splitter.split_documents(self.documents)
        
        self.embeddings = OpenAIEmbeddings()
        if os.path.exists(DB_DIR):
            Chroma(persist_directory=DB_DIR, embedding_function=self.embeddings).delete_collection()
        
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=DB_DIR
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        logger.info("Vector store setup complete")

    def setup_rag_chain(self):
        logger.info("Setting up RAG chain...")
        self.llm = ChatOpenAI(model_name=MODEL, temperature=0.7)
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
        logger.info("RAG chain setup complete")

    def get_context_str_and_docs(self, query: str) -> Tuple[str, List[Document]]:
        docs = self.retriever.get_relevant_documents(query)
        context_snippets = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "Unknown")
            doc_type = doc.metadata.get("doc_type", "Unknown")
            snippet = f"[{i+1}] From {doc_type}/{os.path.basename(source)}:\n{doc.page_content[:200]}..."
            context_snippets.append(snippet)
        return "\n\n".join(context_snippets), docs

    def calculator_tool(self, expr: str) -> str:
        """Advanced calculator using SymPy for comprehensive math support."""
        try:
            # Try to import required libraries
            import sympy as sp
            from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
            import re
            import math
            from sympy import symbols, solve, Eq
        except ImportError:
            logger.warning("SymPy not installed. Installing now...")
            import subprocess
            subprocess.check_call(["pip", "install", "sympy"])
            import sympy as sp
            from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
        
        original_expr = expr
        expr = expr.lower().strip()
        
        # Check for equation solving
        if "solve" in expr or "=" in expr:
            try:
                # Extract the equation part
                equation_match = re.search(r'solve\s+(.*?)(?=$)', expr) if "solve" in expr else None
                if equation_match:
                    equation_str = equation_match.group(1).strip()
                else:
                    equation_str = expr
                
                # Handle equals sign
                if "=" in equation_str:
                    left, right = equation_str.split("=", 1)
                    equation_str = f"{left}-({right})"
                
                # Define the variable (assume x if not specified)
                x = symbols('x')
                
                # Parse and solve the equation
                equation = parse_expr(equation_str, transformations=(standard_transformations + (implicit_multiplication_application,)))
                solution = solve(equation, x)
                
                if solution:
                    if len(solution) == 1:
                        return f"x = {solution[0]}"
                    else:
                        return f"Solutions: {', '.join([f'x = {sol}' for sol in solution])}"
                else:
                    return "No solutions found"
            except Exception as e:
                logger.error(f"Error solving equation: {e}")
                # Fall back to regular calculation if equation solving fails
        
        # Handle special cases like factorial
        if "!" in expr and not "!=" in expr:
            try:
                # Extract the factorial expression
                num_match = re.search(r'(\d+)!', expr)
                if num_match:
                    num = int(num_match.group(1))
                    return str(math.factorial(num))
            except Exception as e:
                return f"Factorial error: {e}"
        
        # Preprocess common expressions
        replacements = {
            "multiplied by": "*",
            "times": "*",
            "plus": "+",
            "minus": "-",
            "divided by": "/",
            "over": "/",
            " x ": " * ",
            "cosec": "1/sin",
            "sec": "1/cos",
            "cot": "1/tan",
            "arcsin": "asin",
            "arccos": "acos",
            "arctan": "atan",
            "x": "*",  # For expressions like "3x5"
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
        
        # Process replacements - need to be careful with "x" since it could be a variable
        # First replace " x " with " * " (space-delimited multiplication)
        expr = expr.replace(" x ", " * ")
        
        # Handle expressions like "3x5" where x is clearly multiplication
        # Look for patterns like digit + x + digit
        expr = re.sub(r'(\d+)x(\d+)', r'\1*\2', expr)
        
        # For other replacements
        for word, sym in replacements.items():
            if word != "x":  # Skip "x" as we handled it specially above
                expr = expr.replace(word, sym)
        
        # Set up parsing transformations
        transformations = standard_transformations + (implicit_multiplication_application,)
        
        try:
            # Convert unit calculations like "5 feet to meters"
            unit_conversion_match = re.search(r'(\d+(?:\.\d+)?)\s+(\w+)\s+to\s+(\w+)', expr)
            if unit_conversion_match:
                value, from_unit, to_unit = unit_conversion_match.groups()
                
                # Define unit conversions
                unit_conversions = {
                    ('feet', 'meters'): lambda x: x * 0.3048,
                    ('meters', 'feet'): lambda x: x * 3.28084,
                    ('inches', 'cm'): lambda x: x * 2.54,
                    ('cm', 'inches'): lambda x: x * 0.393701,
                    ('miles', 'km'): lambda x: x * 1.60934,
                    ('km', 'miles'): lambda x: x * 0.621371,
                    ('pounds', 'kg'): lambda x: x * 0.453592,
                    ('kg', 'pounds'): lambda x: x * 2.20462,
                    ('celsius', 'fahrenheit'): lambda x: x * 9/5 + 32,
                    ('fahrenheit', 'celsius'): lambda x: (x - 32) * 5/9,
                    ('liters', 'gallons'): lambda x: x * 0.264172,
                    ('gallons', 'liters'): lambda x: x * 3.78541,
                    ('ounces', 'grams'): lambda x: x * 28.3495,
                    ('grams', 'ounces'): lambda x: x * 0.03527396,
                    ('yards', 'meters'): lambda x: x * 0.9144,
                    ('meters', 'yards'): lambda x: x * 1.09361,
                    ('mph', 'kph'): lambda x: x * 1.60934,
                    ('kph', 'mph'): lambda x: x * 0.621371,
                    ('acres', 'hectares'): lambda x: x * 0.404686,
                    ('hectares', 'acres'): lambda x: x * 2.47105,
                    # Add more conversions as needed
                }
                
                if (from_unit, to_unit) in unit_conversions:
                    result = unit_conversions[(from_unit, to_unit)](float(value))
                    return f"{value} {from_unit} = {result:.4f} {to_unit}"
                else:
                    return f"Conversion from {from_unit} to {to_unit} not supported"
            
            # Try to parse and evaluate the expression using SymPy
            parsed_expr = parse_expr(expr, transformations=transformations)
            result = parsed_expr.evalf()
            
            # Format the result
            if result.is_integer and float(result) < 10**12:  # if integer and not too large
                return str(int(result))
            else:
                # Format to handle small and large numbers appropriately
                result_float = float(result)
                if abs(result_float) < 1e-10:
                    return "0"
                elif abs(result_float) > 1e10 or abs(result_float) < 1e-6:
                    return f"{result_float:.6e}"
                else:
                    formatted = f"{result_float:.6f}".rstrip('0').rstrip('.')
                    return formatted if formatted else "0"
            
        except Exception as e:
            logger.error(f"SymPy calculation error: {e}")
            
            # Last resort: Try using eval with a limited scope (less capable but might work for simple expressions)
            try:
                # Create a very limited safe scope for basic calculations
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
                    # Add inverse trigonometric functions
                    "cosec": lambda x: 1/math.sin(x),
                    "sec": lambda x: 1/math.cos(x),
                    "cot": lambda x: 1/math.tan(x),
                }
                
                # Explicitly handle "3x5" type expressions by replacing x with *
                eval_expr = re.sub(r'(\d+)x(\d+)', r'\1*\2', expr)
                
                result = eval(eval_expr, {"__builtins__": None}, safe_dict)
                
                # Format the result
                if isinstance(result, float):
                    if abs(result) < 1e-10:
                        return "0"
                    elif abs(result - round(result)) < 1e-10:
                        return str(int(result))
                    else:
                        return f"{result:.6f}".rstrip('0').rstrip('.')
                else:
                    return str(result)
            except Exception as e2:
                return f"Could not calculate: '{original_expr}'. Error: {e2}"

    def dictionary_tool(self, term: str) -> str:
        """Enhanced dictionary tool using multiple sources: Free Dictionary API first, then WordNet."""
        import requests
        import json
        from nltk.corpus import wordnet as wn
        
        term = term.strip().lower()
        term = re.sub(r'[^\w\s]', '', term)  # Remove punctuation
        
        results = []
        definition_found = False
        
        # 1. First try Free Dictionary API (primary source)
        try:
            logger.info(f"Looking up '{term}' in Free Dictionary API")
            url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{term}"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                if data and isinstance(data, list) and len(data) > 0:
                    entry = data[0]
                    
                    api_results = []
                    
                    # Process meanings
                    for meaning in entry.get('meanings', [])[:3]:  # Limit to 3 meanings
                        part_of_speech = meaning.get('partOfSpeech', '')
                        definitions = meaning.get('definitions', [])
                        
                        if definitions:
                            definition = definitions[0].get('definition', '')
                            if definition:
                                api_results.append(f"{part_of_speech}: {definition}")
                                
                                # Add example if available
                                example = definitions[0].get('example', '')
                                if example:
                                    api_results.append(f"   Example: {example}")
                    
                    if api_results:
                        results.append("\n".join(api_results))
                        definition_found = True
        except Exception as e:
            logger.error(f"Error using Free Dictionary API: {e}")
        
        # 2. Try WordNet only if Free Dictionary API didn't return results
        if not definition_found:
            synsets = wn.synsets(term)
            if synsets:
                lines = []
                for syn in synsets[:3]:
                    pos = syn.pos()
                    pos_mapping = {
                        'n': 'noun',
                        'v': 'verb',
                        'a': 'adjective',
                        's': 'adjective satellite',
                        'r': 'adverb'
                    }
                    pos_full = pos_mapping.get(pos, pos)
                    lines.append(f"{pos_full}: {syn.definition()}")
                if lines:
                    results.append("\n".join(lines))
                    definition_found = True
        
        # 3. As a last resort, use the LLM (only if enabled and no other definitions found)
        use_llm_fallback = False  
        
        if not definition_found and use_llm_fallback:
            try:
                logger.info(f"Term '{term}' not found in dictionaries, using LLM as last resort")
                prompt = f"Provide a clear, concise dictionary definition for the term: '{term}'. Give 1-3 definitions with their part of speech, like a dictionary entry."
                
                response = self.llm.invoke(prompt)
                
                if response and len(response.content) > 10:
                    results.append("AI-generated definition:")
                    results.append(response.content)
            except Exception as e:
                logger.error(f"Error using LLM for definition: {e}")

        if results:
            return "\n\n".join(results)
        else:
            return f"No definition found for '{term}'."

    def process_message(self, message: str) -> Dict[str, str]:
        lower = message.lower()
        
        self.chat_history.append({"role": "user", "content": message})

        if "calculate" in lower:
            tool_used = "Calculator Tool"
            logger.info(f"Routing to Calculator Tool for expr: {message}")
            expr = re.sub(r"(?i).*calculate\s*", "", message).strip()
            answer = self.calculator_tool(expr)
            context_snippets = "No context retrieval for calculator tool"
        elif "define" in lower:
            tool_used = "Dictionary Tool"
            logger.info(f"Routing to Dictionary Tool for term: {message}")
            term = re.sub(r"(?i).*define\s*", "", message).strip(" ?.")
            answer = self.dictionary_tool(term)
            context_snippets = "No context retrieval for dictionary tool"
        else:
            tool_used = "RAG Pipeline"
            logger.info(f"Routing to RAG pipeline for question: {message}")
            
            context_snippets, _ = self.get_context_str_and_docs(message)
            
            result = self.rag_chain.invoke({"question": message})
            answer = result["answer"]
        
        self.chat_history.append({"role": "assistant", "content": answer})
        
        return {
            "tool_used": tool_used,
            "context": context_snippets,
            "answer": answer
        }

def initialize_chatbot():
    return RAGChatbot()