# RAGex: Retrieval-Augmented Generation with External Tools

## Project Description

RAGEX (Retrieval-Augmented Generation with External Tools) is a RAG-Powered Multi-Agent Q&A Assistant that retrieves relevant information from a knowledge base and uses extra external tools (calculator, dictionary) to answer user queries.

### Core Components

* **LLM Used**: OpenAI's GPT-4o-mini
* **Vector Database**: Chroma (stored in `vector_db` directory)
* **UI Tool**: Gradio (provides a simple web-based chat interface)
* **Knowledge Base Data**: Text files in `knowledge-base` directory

## Setup Instructions

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/SambhavJain26/RAGex.git
   cd RAGEX
   ```

2. **Create Virtual Environment (Optional)**:

   ```bash
   python -m venv ragenv
   ragenv/Scripts/activate     # On Mac: source ragenv/bin/activate
   ```

3. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Setup Environment Variables**:

   * Create a `.env` file in the project directory.
   * Add your OpenAI API key:

     ```
     OPENAI_API_KEY=your-openai-api-key
     ```

5. **Run the Application**:

   ```bash
   python pipeline.py
   ```

   * This will launch the Gradio UI in your web browser. If it does not open automatically, the URL will be displayed in the terminal.

## Usage

* Ask any question, and the system will retrieve relevant context from the knowledge base and generate an answer.
* Use "calculate" in your query to perform mathematical calculations.
* Use "define" to get the definition of any word.

## Directory Structure

* `knowledge-base/` - Directory containing text documents for context retrieval.
* `vector_db/` - Directory for storing vector embeddings (Chroma).
* `pipeline.py` - Main application file.
* `requirements.txt` - Project dependencies.
* `.env` - Environment file for API keys.

