import os
import sys
import uuid
import logging
from typing import Dict, List, Any
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, session
from flask_socketio import SocketIO

# Conditional eventlet monkey patching for non-Windows container environments
if os.name != 'nt':
    try:
        import eventlet
        eventlet.monkey_patch()
    except ImportError:
        pass

load_dotenv(override=True)
if not os.getenv('OPENAI_API_KEY'):
    raise ValueError("OPENAI_API_KEY not configured in environment or .env file.")

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize Flask application
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', os.urandom(24).hex())

# Initialize SocketIO with fallback for Windows threading
async_mode = 'threading' if os.name == 'nt' else None
socketio = SocketIO(app, cors_allowed_origins="*", async_mode=async_mode)

# Lazy/Global chatbot reference
from rag_utils import initialize_chatbot, RAGChatbot

logger.info("Initializing RAG chatbot subsystem...")
chatbot = initialize_chatbot()
logger.info("RAG chatbot ready.")

# In-memory session store
chat_sessions: Dict[str, List[Dict[str, Any]]] = {}


@app.route('/')
def index():
    """Render main chat interface with active session ID."""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
        chat_sessions[session['session_id']] = []
    return render_template('index.html')


@app.route('/api/health', methods=['GET'])
def health():
    """System health check and diagnostic status."""
    return jsonify({
        'status': 'healthy',
        'model': os.getenv('RAG_MODEL', 'gpt-4o-mini'),
        'documents_indexed': len(chatbot.documents),
        'vector_store': 'ChromaDB',
        'tools_available': ['Calculator Tool (SymPy)', 'Dictionary Tool (API + WordNet)', 'RAG Retrieval Pipeline']
    })


@app.route('/api/chat', methods=['POST'])
def chat():
    """Process incoming query and route to calculation, dictionary, or RAG engine."""
    data = request.get_json(silent=True) or {}
    user_message = data.get('message', '').strip()
    session_id = session.get('session_id')

    if not user_message:
        return jsonify({'error': 'No message provided'}), 400

    try:
        response = chatbot.process_message(user_message)

        if session_id:
            if session_id not in chat_sessions:
                chat_sessions[session_id] = []
            chat_sessions[session_id].append({
                'user': user_message,
                'bot': response
            })

        return jsonify(response)
    except Exception as e:
        logger.exception(f"Error processing message: {str(e)}")
        return jsonify({'error': f'Failed to process query: {str(e)}'}), 500


@app.route('/api/chat-history', methods=['GET'])
def get_chat_history():
    """Retrieve chat history for current active session."""
    session_id = session.get('session_id')
    if not session_id or session_id not in chat_sessions:
        return jsonify([])
    return jsonify(chat_sessions[session_id])


@app.route('/api/clear-history', methods=['POST'])
def clear_history():
    """Reset chat history for current session."""
    session_id = session.get('session_id')
    if session_id and session_id in chat_sessions:
        chat_sessions[session_id] = []
    return jsonify({'status': 'success'})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    logger.info(f"Starting RAGex on http://0.0.0.0:{port} (debug={debug_mode})")
    socketio.run(app, host='0.0.0.0', port=port, debug=debug_mode, allow_unsafe_werkzeug=True)
