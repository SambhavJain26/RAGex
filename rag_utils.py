import os
import glob
import re
import logging
from typing import Tuple, List, Dict, Any
import nltk
from nltk.corpus import wordnet as wn

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
        expr = expr.lower()
        replacements = {
            "multiplied by": "*",
            "times": "*",
            "plus": "+",
            "minus": "-",
            "divided by": "/",
            "over": "/",
            " x ": " * ",
        }
        for word, sym in replacements.items():
            expr = expr.replace(word, sym)
        expr = re.sub(r"[^0-9\+\-\*/\.\(\)\s]", "", expr)
        try:
            return str(eval(expr, {"__builtins__": {}}, {}))
        except Exception as e:
            return f"Calculation error: {e}"

    def dictionary_tool(self, term: str) -> str:
        synsets = wn.synsets(term)
        if not synsets:
            return f"No definition found for '{term}'."
        lines = []
        for syn in synsets[:3]:
            lines.append(f"{syn.pos()}: {syn.definition()}")
        return "\n".join(lines)

    def process_message(self, message: str) -> Dict[str, str]:
        lower = message.lower()
        
        self.chat_history.append({"role": "user", "content": message})
        
        # Determining which tool to use
        if "calculate" in lower:
            tool_used = "Calculator Tool"
            logger.info(f"Routing to Calculator Tool for expr: {message}")
            expr = re.sub(r"(?i).*calculate\s*", "", message)
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
        
        # Update chat history with bot response
        self.chat_history.append({"role": "assistant", "content": answer})
        
        return {
            "tool_used": tool_used,
            "context": context_snippets,
            "answer": answer
        }

def initialize_chatbot():
    return RAGChatbot()