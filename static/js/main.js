document.addEventListener('DOMContentLoaded', () => {
    const chatForm = document.getElementById('chat-form');
    const messageInput = document.getElementById('message-input');
    const chatMessages = document.getElementById('chat-messages');
    const newChatBtn = document.getElementById('new-chat-btn');
    
    messageInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
    });

    messageInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            chatForm.dispatchEvent(new Event('submit')); 
        }
    });

    loadChatHistory();

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
    
    newChatBtn.addEventListener('click', async () => {
        try {
            await fetch('/api/clear-history', { method: 'POST' });
            chatMessages.innerHTML = '';
            const welcomeDiv = document.createElement('div');
            welcomeDiv.className = 'welcome-message';
            welcomeDiv.innerHTML = `
                <h1>Welcome to RAG Chatbot</h1>
                <p>Ask me anything about your knowledge base, or try:</p>
                <ul>
                    <li>"What is Insurellm and its products"</li>
                    <li>"Calculate 25 * 16"</li>
                    <li>"Define knowledge"</li>
                </ul>
            `;
            chatMessages.appendChild(welcomeDiv);
        } catch (error) {
            console.error('Error clearing history:', error);
        }
    });
    
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
    
    function addTypingIndicator() {
        const typingDiv = document.createElement('div');
        typingDiv.className = 'typing-indicator';
        typingDiv.innerHTML = `<span></span><span></span><span></span>`;
        chatMessages.appendChild(typingDiv);
        scrollToBottom();
        return typingDiv;
    }
    
    function addErrorMessage() {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message bot-message error-message';
        messageDiv.innerHTML = `<div class="message-content">
            <p>Sorry, there was an error processing your request. Please try again.</p>
        </div>`;
        chatMessages.appendChild(messageDiv);
        scrollToBottom();
    }
    
    async function loadChatHistory() {
        try {
            const response = await fetch('/api/chat-history');
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            
            const history = await response.json();
            
            if (history.length === 0) {
                return;
            }
            
            chatMessages.innerHTML = '';

            history.forEach(item => {
                addUserMessage(item.user);
                addBotResponse(item.bot);
            });
            
        } catch (error) {
            console.error('Error loading chat history:', error);
        }
    }
    
    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    function escapeHtml(unsafe) {
        return unsafe
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }
    
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