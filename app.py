import os
from flask import Flask, render_template, request, jsonify, session
from flask_socketio import SocketIO
from dotenv import load_dotenv
import logging
import uuid

from rag_utils import initialize_chatbot

load_dotenv(override=True)
if not os.getenv('OPENAI_API_KEY'):
    raise ValueError("OPENAI_API_KEY not set in environment or .env file")

# Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', os.urandom(24).hex())
socketio = SocketIO(app)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Initializing chatbot
chatbot = initialize_chatbot()

chat_sessions = {}

@app.route('/')
def index():
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
        chat_sessions[session['session_id']] = []
    
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message')
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
        logger.error(f"Error processing message: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/chat-history', methods=['GET'])
def get_chat_history():
    session_id = session.get('session_id')
    if not session_id or session_id not in chat_sessions:
        return jsonify([])
    return jsonify(chat_sessions[session_id])

@app.route('/api/clear-history', methods=['POST'])
def clear_history():
    session_id = session.get('session_id')
    if session_id and session_id in chat_sessions:
        chat_sessions[session_id] = []
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    socketio.run(app, host='0.0.0.0', port=port, debug=True)