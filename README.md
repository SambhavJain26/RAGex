# RAGex: Retrieval-Augmented Generation with External Tools

## Project Description

RAGex is a RAG-Powered Multi-Agent Q&A Assistant that retrieves relevant information from a knowledge base and uses external tools (calculator, dictionary) to answer user queries.

### Core Components

* **LLM Used**: OpenAI's GPT-4o-mini
* **Vector Database**: Chroma (stored in `vector_db` directory)
* **UI Tools**:

  * Gradio (pipeline.py for simple web-based chat interface)
  * Flask (app.py for a more customizable web app)
* **Knowledge Base Data**: Text files in `knowledge-base` directory

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/SambhavJain26/RAGex.git
cd RAGEX
```

### 2. Create Virtual Environment (Optional)

```bash
python -m venv ragenv

# On macOS/Linux:
source ragenv/bin/activate

# On Windows:
ragenv/Scripts/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Initialize NLTK WordNet

Before running the app, download the WordNet corpus:

```bash
python -c "import nltk; nltk.download('wordnet')"
```

### 5. Setup Environment Variables

* Create a `.env` file in the project directory.
* Add your OpenAI API key and Flask secret key:

  ```
  OPENAI_API_KEY=your-openai-api-key
  SECRET_KEY=flask_secret_key
  ```
* To automatically generate a secure Flask secret key, use:

  ```bash
  python -c "import secrets; print(secrets.token_hex(24))"
  ```

### 6. Running the Application

#### Using Flask (Recommended)

```bash
python app.py
```

* This will start the Flask app at `http://localhost:5000`.
* If you encounter any issues with Flask, you can use the Gradio UI (below).

#### Using Gradio (Alternative)

```bash
python pipeline.py
```

* This will launch a Gradio UI in your web browser. If it does not open automatically, the URL will be displayed in the terminal.

## Usage

* Ask any question, and the system will retrieve relevant context from the knowledge base and generate an answer.
* Use "calculate" in your query to perform mathematical calculations.
* Use "define" to get the definition of any word.

## Directory Structure

* `knowledge-base/` - Directory containing text documents for context retrieval.
* `vector_db/` - Directory for storing vector embeddings (Chroma).
* `pipeline.py` - Main application file for Gradio UI.
* `app.py` - Main application file for Flask UI.
* `rag_utils.py` - Utility functions used by Flask app.
* `requirements.txt` - Project dependencies.
* `.env` - Environment file for API keys and Flask secret.

## License

This project is licensed under the MIT License.
