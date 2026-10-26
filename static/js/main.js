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

    // Set up click listeners for example question cards
    function setupQuestionItemListeners() {
        const questionItems = document.querySelectorAll('.question-item');
        questionItems.forEach(item => {
            // Remove existing listener to avoid duplication
            item.onclick = function() {
                const question = this.textContent.trim();
                messageInput.value = question;
                messageInput.focus();
                messageInput.style.height = 'auto';
                messageInput.style.height = (messageInput.scrollHeight) + 'px';
            };
        });
    }

    // Load initial chat history or welcome message
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
    
    // New chat session
    newChatBtn.addEventListener('click', async () => {
        try {
            await fetch('/api/clear-history', { method: 'POST' });
            chatMessages.innerHTML = '';
            addWelcomeMessage();
            setupQuestionItemListeners();
        } catch (error) {
            console.error('Error resetting chat:', error);
        }
    });
    
    // Function to add welcome message with tool grid
    function addWelcomeMessage() {
        const welcomeDiv = document.createElement('div');
        welcomeDiv.className = 'welcome-message';
        welcomeDiv.innerHTML = `
            <div class="welcome-hero mb-4">
                <h1 class="fw-bold">RAGex Intelligence Assistant</h1>
                <p class="text-muted lead">Hybrid question-answering system combining semantic vector retrieval with deterministic tools.</p>
            </div>
            
            <div class="tools-grid">
                <div class="row row-cols-1 row-cols-md-3 g-4">
                    <div class="col">
                        <div class="card tool-card h-100 border-0 shadow-sm">
                            <div class="card-header bg-primary text-white py-3">
                                <i class="fas fa-database me-2"></i> RAG Vector Retrieval
                            </div>
                            <div class="card-body">
                                <p class="card-text text-muted small">Query domain documents in the Insurellm knowledge base with semantic grounding.</p>
                                <ul class="example-questions list-unstyled mb-0">
                                    <li class="question-item"><i class="fas fa-chevron-right me-2 text-primary"></i>Who is Avery Lancaster?</li>
                                    <li class="question-item"><i class="fas fa-chevron-right me-2 text-primary"></i>Explain Carllm in a few lines</li>
                                    <li class="question-item"><i class="fas fa-chevron-right me-2 text-primary"></i>What are the prices of Carllm?</li>
                                    <li class="question-item"><i class="fas fa-chevron-right me-2 text-primary"></i>Tell me about Insurellm</li>
                                    <li class="question-item"><i class="fas fa-chevron-right me-2 text-primary"></i>What is the 2025 roadmap for Carllm?</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                    
                    <div class="col">
                        <div class="card tool-card h-100 border-0 shadow-sm">
                            <div class="card-header bg-success text-white py-3">
                                <i class="fas fa-calculator me-2"></i> Deterministic Calculator
                            </div>
                            <div class="card-body">
                                <p class="card-text text-muted small">Perform zero-hallucination math, algebraic equations, and unit conversions via SymPy.</p>
                                <ul class="example-questions list-unstyled mb-0">
                                    <li class="question-item"><i class="fas fa-chevron-right me-2 text-success"></i>Calculate 3x5</li>
                                    <li class="question-item"><i class="fas fa-chevron-right me-2 text-success"></i>Calculate sqrt(144) + 25</li>
                                    <li class="question-item"><i class="fas fa-chevron-right me-2 text-success"></i>Calculate 5 feet to meters</li>
                                    <li class="question-item"><i class="fas fa-chevron-right me-2 text-success"></i>Calculate sin(45) + cos(30)</li>
                                    <li class="question-item"><i class="fas fa-chevron-right me-2 text-success"></i>Calculate 10!</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                    
                    <div class="col">
                        <div class="card tool-card h-100 border-0 shadow-sm">
                            <div class="card-header bg-danger text-white py-3">
                                <i class="fas fa-book me-2"></i> Lexical Dictionary
                            </div>
                            <div class="card-body">
                                <p class="card-text text-muted small">Fetch lexical definitions with parts of speech, examples, and WordNet fallback.</p>
                                <ul class="example-questions list-unstyled mb-0">
                                    <li class="question-item"><i class="fas fa-chevron-right me-2 text-danger"></i>Define serendipity</li>
                                    <li class="question-item"><i class="fas fa-chevron-right me-2 text-danger"></i>Define algorithm</li>
                                    <li class="question-item"><i class="fas fa-chevron-right me-2 text-danger"></i>Define ephemeral</li>
                                    <li class="question-item"><i class="fas fa-chevron-right me-2 text-danger"></i>Define pragmatic</li>
                                    <li class="question-item"><i class="fas fa-chevron-right me-2 text-danger"></i>Define ubiquitous</li>
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
        
        let badgeClass = 'tool-badge-rag';
        let iconClass = 'fa-database';
        if (data.tool_used === 'Calculator Tool') {
            badgeClass = 'tool-badge-calc';
            iconClass = 'fa-calculator';
        } else if (data.tool_used === 'Dictionary Tool') {
            badgeClass = 'tool-badge-dict';
            iconClass = 'fa-book';
        }

        let html = '<div class="message-content">';
        html += `<div class="tool-badge ${badgeClass}"><i class="fas ${iconClass} me-2"></i>${escapeHtml(data.tool_used)}</div>`;

        if (data.context && 
            data.context !== 'Deterministic calculation engine invoked (no vector context required).' && 
            data.context !== 'Lexical dictionary engine invoked (no vector context required).' &&
            data.context !== 'No context retrieval for calculator tool' && 
            data.context !== 'No context retrieval for dictionary tool') {
            html += `<div class="context-card">
                        <div class="context-header"><i class="fas fa-file-lines me-2"></i>Retrieved Knowledge Attribution</div>
                        <div class="context-body">${escapeHtml(data.context)}</div>
                    </div>`;
        }

        html += `<div class="answer-body">${renderMarkdown(data.answer)}</div>`;
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
        messageDiv.innerHTML = `<div class="message-content text-danger">
            <p class="mb-0"><i class="fas fa-circle-exclamation me-2"></i>Unable to process query. Please verify server connection and try again.</p>
        </div>`;
        chatMessages.appendChild(messageDiv);
        scrollToBottom();
    }
    
    // Load chat history
    async function loadChatHistory() {
        try {
            const response = await fetch('/api/chat-history');
            if (!response.ok) throw new Error('Network error');
            const history = await response.json();
            
            if (history.length === 0) {
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
            console.error('Error loading history:', error);
            addWelcomeMessage();
            setupQuestionItemListeners();
        }
    }
    
    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    function escapeHtml(unsafe) {
        if (!unsafe) return '';
        return String(unsafe)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }
    
    function renderMarkdown(text) {
        if (!text) return '';
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
