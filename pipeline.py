import os
import glob
import re
import logging
from dotenv import load_dotenv
import gradio as gr
import nltk
from nltk.corpus import wordnet as wn

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


MODEL = "gpt-4o-mini"
DB_DIR = "vector_db"

load_dotenv(override=True)
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')

# logging each step
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


folders = glob.glob("knowledge-base/*")
documents = []
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
        documents.append(doc)

# creating chunks
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
chunks = splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
if os.path.exists(DB_DIR):
    Chroma(persist_directory=DB_DIR, embedding_function=embeddings).delete_collection()

# storing in chroma vectorstore
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=DB_DIR
)

llm = ChatOpenAI(model_name=MODEL, temperature=0.7)
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})      # retrieves the top 3 chunks

# Modified to expose the retrieved documents
def get_context_str_and_docs(query):
    docs = retriever.get_relevant_documents(query)
    context_snippets = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "Unknown")
        doc_type = doc.metadata.get("doc_type", "Unknown")
        snippet = f"[{i+1}] From {doc_type}/{os.path.basename(source)}:\n{doc.page_content[:200]}..."
        context_snippets.append(snippet)
    return "\n\n".join(context_snippets), docs

# creating the rag pipeline using langchain
rag_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True,  
)

def calculator_tool(expr: str) -> str:
    expr = expr.lower()
    replacements = {
        "multiplied by": "*",
        "times": "*",
        "plus": "+",
        "minus": "-",
        "divided by": "/",
        "over": "/",
        # allow 'x' as multiplication
        " x ": " * ",
    }
    for word, sym in replacements.items():
        expr = expr.replace(word, sym)
    expr = re.sub(r"[^0-9\+\-\*/\.\(\)\s]", "", expr)
    try:
        return str(eval(expr, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Calculation error: {e}"

def dictionary_tool(term: str) -> str:
    synsets = wn.synsets(term)
    if not synsets:
        return f"No definition found for '{term}'."
    lines = []
    for syn in synsets[:3]:
        lines.append(f"{syn.pos()}: {syn.definition()}")
    return "\n".join(lines)

def chat(message, history):
    lower = message.lower()
    response_parts = []
    
    if "calculate" in lower:  
        tool_used = "Tool used: Calculator Tool"
        logger.info("Routing to Calculator Tool for expr: %s", message)
        expr = re.sub(r"(?i).*calculate\s*", "", message)
        answer = calculator_tool(expr)
        context_snippets = "No context retrieval for calculator tool"
    elif "define" in lower:
        tool_used = "Tool used: Dictionary Tool"
        logger.info("Routing to Dictionary Tool for term: %s", message)
        term = re.sub(r"(?i).*define\s*", "", message).strip(" ?.")
        answer = dictionary_tool(term)
        context_snippets = "No context retrieval for dictionary tool"
    else:
        tool_used = "Tool used: RAG Pipeline"
        logger.info("Routing to RAG pipeline for question: %s", message)
        
        context_snippets, _ = get_context_str_and_docs(message)
        
        result = rag_chain.invoke({"question": message})
        answer = result["answer"]
    
    response_parts.append(f"### {tool_used}\n")
    response_parts.append(f"### Retrieved Context\n```\n{context_snippets}\n```\n")
    response_parts.append(f"### Answer\n{answer}")
    
    return "\n\n".join(response_parts)


# simple chat ui using gradio
view = gr.ChatInterface(
    chat,
    type="messages",
    chatbot=gr.Chatbot(render=True, height=600),
).launch(inbrowser=True)