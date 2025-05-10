document.addEventListener('DOMContentLoaded', () => {
    const chatForm = document.getElementById('chat-form');
    const messageInput = document.getElementById('message-input');
    const chatMessages = document.getElementById('chat-messages');
    const newChatBtn = document.getElementById('new-chat-btn');
    
    // Auto-resize the input field
    messageInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
    });

    // Submit when pressing Enter (without shift)
    messageInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            chatForm.dispatchEvent(new Event('submit')); 
        }
    });

    // IMMEDIATELY clear any existing welcome message to prevent both showing
    // This is critical to fix the issue with both welcome messages showing on refresh
    const existingWelcomeMessage = document.querySelector('.welcome-message');
    if (existingWelcomeMessage) {
        existingWelcomeMessage.remove();
    }

    // Set up event listeners for question items - this needs to be called whenever new questions are added
    function setupQuestionItemListeners() {
        const questionItems = document.querySelectorAll('.question-item');
        
        questionItems.forEach(item => {
            item.addEventListener('click', function() {
                const question = this.textContent.trim();
                messageInput.value = question;
                messageInput.focus();
                
                // Auto-resize the input field to match content
                messageInput.style.height = 'auto';
                messageInput.style.height = (messageInput.scrollHeight) + 'px';
            });
        });
    }
    
    // Load chat history on page load
    loadChatHistory();

    // Chat form submission
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const message = messageInput.value.trim();
        if (!message) return;
        
        messageInput.value = '';
        messageInput.style.height = 'auto';
        
        addUserMessage(message);

        const typingIndicator = addTypingIndicator();
        
        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ message })
            });
            
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            
            const data = await response.json();
            
            typingIndicator.remove();
            addBotResponse(data);

            scrollToBottom();
        } catch (error) {
            console.error('Error:', error);
            typingIndicator.remove();
            addErrorMessage();
        }
    });
    
    // New chat button - clear history and show welcome message with grid
    newChatBtn.addEventListener('click', async () => {
        try {
            await fetch('/api/clear-history', { method: 'POST' });
            chatMessages.innerHTML = '';
            
            // Create welcome message with grid of questions
            addWelcomeMessage();
            
            // Set up listeners for the newly added question items
            setupQuestionItemListeners();
            
        } catch (error) {
            console.error('Error clearing history:', error);
        }
    });
    
    // Function to add welcome message with grid
    function addWelcomeMessage() {
        const welcomeDiv = document.createElement('div');
        welcomeDiv.className = 'welcome-message';
        welcomeDiv.innerHTML = `
            <h1>Welcome to RAG Chatbot</h1>
            <p>Ask me anything about your knowledge base, or try one of these examples:</p>
            
            <div class="tools-grid">
                <div class="row row-cols-1 row-cols-md-2 g-4">
                    <!-- RAG Pipeline Tool -->
                    <div class="col">
                        <div class="card tool-card">
                            <div class="card-header">
                                <i class="fas fa-search me-2"></i> RAG Pipeline Tool
                            </div>
                            <div class="card-body">
                                <p class="card-text">Ask questions about your knowledge base.</p>
                                <ul class="example-questions">
                                    <li class="question-item">Who is Avery Lancaster?</li>
                                    <li class="question-item">Explain CarLLM in a few lines</li>
                                    <li class="question-item">What are the prices of CarLLM?</li>
                                    <li class="question-item">Tell me about InsureLLM</li>
                                    <li class="question-item">Different products of InsureLLM</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Calculator Tool -->
                    <div class="col">
                        <div class="card tool-card">
                            <div class="card-header">
                                <i class="fas fa-calculator me-2"></i> Calculator Tool
                            </div>
                            <div class="card-body">
                                <p class="card-text">Perform calculations of various types.</p>
                                <ul class="example-questions">
                                    <li class="question-item">Calculate 3x5</li>
                                    <li class="question-item">Calculate sqrt(144) + 25</li>
                                    <li class="question-item">Calculate 5 feet to meters</li>
                                    <li class="question-item">Calculate sin(45) + cos(30)</li>
                                    <li class="question-item">Calculate 10!</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Dictionary Tool -->
                    <div class="col">
                        <div class="card tool-card">
                            <div class="card-header">
                                <i class="fas fa-book me-2"></i> Dictionary Tool
                            </div>
                            <div class="card-body">
                                <p class="card-text">Get definitions for words and phrases.</p>
                                <ul class="example-questions">
                                    <li class="question-item">Define serendipity</li>
                                    <li class="question-item">Define algorithm</li>
                                    <li class="question-item">Define ephemeral</li>
                                    <li class="question-item">Define pragmatic</li>
                                    <li class="question-item">Define ubiquitous</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
        chatMessages.appendChild(welcomeDiv);
    }
    
    // Add user message to chat
    function addUserMessage(message) {
        const welcomeMessage = document.querySelector('.welcome-message');
        if (welcomeMessage) {
            welcomeMessage.remove();
        }
        
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message user-message';
        messageDiv.innerHTML = `<div class="message-content">${escapeHtml(message)}</div>`;
        chatMessages.appendChild(messageDiv);
        scrollToBottom();
    }
    
    // Add bot response to chat
    function addBotResponse(data) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message bot-message';
        
        let html = '<div class="message-content">';

        html += `<div class="response-section">
                    <div class="section-title">Tool Used: ${escapeHtml(data.tool_used)}</div>
                </div>`;

        if (data.context && data.context !== 'No context retrieval for calculator tool' && 
            data.context !== 'No context retrieval for dictionary tool') {
            html += `<div class="response-section">
                        <div class="section-title">Retrieved Context:</div>
                        <div class="context-container">${escapeHtml(data.context)}</div>
                    </div>`;
        }

        html += `<div class="response-section">
                    <div class="section-title">Answer:</div>
                    <div>${renderMarkdown(data.answer)}</div>
                </div>`;
        
        html += '</div>';
        messageDiv.innerHTML = html;
        chatMessages.appendChild(messageDiv);
        
        document.querySelectorAll('pre code').forEach((block) => {
            hljs.highlightElement(block);
        });
        
        scrollToBottom();
    }
    
    // Add typing indicator 
    function addTypingIndicator() {
        const typingDiv = document.createElement('div');
        typingDiv.className = 'typing-indicator';
        typingDiv.innerHTML = `<span></span><span></span><span></span>`;
        chatMessages.appendChild(typingDiv);
        scrollToBottom();
        return typingDiv;
    }
    
    // Add error message
    function addErrorMessage() {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message bot-message error-message';
        messageDiv.innerHTML = `<div class="message-content">
            <p>Sorry, there was an error processing your request. Please try again.</p>
        </div>`;
        chatMessages.appendChild(messageDiv);
        scrollToBottom();
    }
    
    // Load chat history
    async function loadChatHistory() {
        try {
            const response = await fetch('/api/chat-history');
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            
            const history = await response.json();
            
            if (history.length === 0) {
                // If no history, add welcome message
                addWelcomeMessage();
                setupQuestionItemListeners();
                return;
            }
            
            chatMessages.innerHTML = '';

            history.forEach(item => {
                addUserMessage(item.user);
                addBotResponse(item.bot);
            });
            
        } catch (error) {
            console.error('Error loading chat history:', error);
            // If error loading history, display welcome message
            addWelcomeMessage();
            setupQuestionItemListeners();
        }
    }
    
    // Scroll to bottom of chat
    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    // Escape HTML to prevent XSS
    function escapeHtml(unsafe) {
        return unsafe
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }
    
    // Render markdown 
    function renderMarkdown(text) {
        marked.setOptions({
            highlight: function(code, lang) {
                const language = hljs.getLanguage(lang) ? lang : 'plaintext';
                return hljs.highlight(code, { language }).value;
            },
            langPrefix: 'hljs language-',
            breaks: true
        });
        
        return marked.parse(text);
    }
});