"""
ZSE Enhanced Playground UI

Beautiful chat interface with:
- Chat bubbles with markdown rendering
- Code syntax highlighting  
- Conversation sidebar
- System prompts and parameters
- Streaming responses
- Dark/light theme
"""

def get_enhanced_playground_html() -> str:
    """Generate the enhanced playground HTML."""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ZSE Playground</title>
    <link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>⚡</text></svg>">
    <!-- Markdown rendering -->
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <!-- Code highlighting -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/highlight.js@11/styles/github-dark.min.css">
    <script src="https://cdn.jsdelivr.net/npm/highlight.js@11"></script>
    <style>
        :root {
            --bg-primary: #0a0a0a;
            --bg-secondary: #141414;
            --bg-tertiary: #1a1a1a;
            --bg-hover: #222;
            --text-primary: #fff;
            --text-secondary: #a0a0a0;
            --text-muted: #666;
            --border-color: #2a2a2a;
            --accent: #3b82f6;
            --accent-hover: #2563eb;
            --user-bubble: #3b82f6;
            --assistant-bubble: #1e293b;
            --success: #10b981;
            --error: #ef4444;
        }
        
        .light-theme {
            --bg-primary: #ffffff;
            --bg-secondary: #f8f9fa;
            --bg-tertiary: #f1f3f5;
            --bg-hover: #e9ecef;
            --text-primary: #212529;
            --text-secondary: #6c757d;
            --text-muted: #adb5bd;
            --border-color: #dee2e6;
            --user-bubble: #3b82f6;
            --assistant-bubble: #f1f3f5;
        }
        
        * { box-sizing: border-box; margin: 0; padding: 0; }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            height: 100vh;
            overflow: hidden;
        }
        
        /* Layout */
        .app {
            display: flex;
            height: 100vh;
        }
        
        /* Sidebar */
        .sidebar {
            width: 280px;
            background: var(--bg-secondary);
            border-right: 1px solid var(--border-color);
            display: flex;
            flex-direction: column;
            flex-shrink: 0;
        }
        
        .sidebar-header {
            padding: 16px;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        
        .sidebar-header h1 {
            font-size: 18px;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .new-chat-btn {
            background: var(--accent);
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 13px;
            font-weight: 500;
            display: flex;
            align-items: center;
            gap: 6px;
            transition: background 0.2s;
        }
        
        .new-chat-btn:hover {
            background: var(--accent-hover);
        }
        
        .conversations-list {
            flex: 1;
            overflow-y: auto;
            padding: 8px;
        }
        
        .conversation-item {
            padding: 12px;
            border-radius: 8px;
            cursor: pointer;
            margin-bottom: 4px;
            transition: background 0.2s;
        }
        
        .conversation-item:hover {
            background: var(--bg-hover);
        }
        
        .conversation-item.active {
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
        }
        
        .conversation-title {
            font-size: 14px;
            font-weight: 500;
            margin-bottom: 4px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        
        .conversation-meta {
            font-size: 12px;
            color: var(--text-muted);
            display: flex;
            justify-content: space-between;
        }
        
        .sidebar-footer {
            padding: 12px;
            border-top: 1px solid var(--border-color);
            display: flex;
            gap: 8px;
        }
        
        .sidebar-btn {
            flex: 1;
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            color: var(--text-secondary);
            padding: 8px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 6px;
            transition: all 0.2s;
        }
        
        .sidebar-btn:hover {
            background: var(--bg-hover);
            color: var(--text-primary);
        }
        
        /* Main Chat Area */
        .main {
            flex: 1;
            display: flex;
            flex-direction: column;
            min-width: 0;
        }
        
        .chat-header {
            padding: 12px 20px;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: var(--bg-secondary);
        }
        
        .chat-title {
            font-size: 14px;
            font-weight: 600;
        }
        
        .header-actions {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        
        .model-select {
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 13px;
            cursor: pointer;
        }
        
        .icon-btn {
            background: none;
            border: none;
            color: var(--text-secondary);
            cursor: pointer;
            padding: 6px;
            border-radius: 4px;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.2s;
        }
        
        .icon-btn:hover {
            color: var(--text-primary);
            background: var(--bg-hover);
        }
        
        /* Messages */
        .messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 16px;
        }
        
        .message {
            display: flex;
            gap: 12px;
            max-width: 900px;
            animation: fadeIn 0.3s ease;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .message.user {
            flex-direction: row-reverse;
            align-self: flex-end;
        }
        
        .message.assistant {
            align-self: flex-start;
        }
        
        .message-avatar {
            width: 32px;
            height: 32px;
            border-radius: 6px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 16px;
            flex-shrink: 0;
        }
        
        .message.user .message-avatar {
            background: var(--user-bubble);
        }
        
        .message.assistant .message-avatar {
            background: var(--assistant-bubble);
            border: 1px solid var(--border-color);
        }
        
        .message-content {
            padding: 12px 16px;
            border-radius: 12px;
            max-width: 700px;
            line-height: 1.6;
        }
        
        .message.user .message-content {
            background: var(--user-bubble);
            color: white;
        }
        
        .message.assistant .message-content {
            background: var(--assistant-bubble);
            border: 1px solid var(--border-color);
        }
        
        .message-content p { margin: 0 0 12px 0; }
        .message-content p:last-child { margin-bottom: 0; }
        
        .message-content pre {
            background: #0d1117;
            border-radius: 8px;
            padding: 12px;
            overflow-x: auto;
            margin: 12px 0;
            font-size: 13px;
        }
        
        .message-content code {
            font-family: 'SF Mono', Monaco, monospace;
            font-size: 13px;
        }
        
        .message-content code:not(pre code) {
            background: rgba(0,0,0,0.2);
            padding: 2px 6px;
            border-radius: 4px;
        }
        
        .message-content ul, .message-content ol {
            margin: 12px 0;
            padding-left: 24px;
        }
        
        .message-content li {
            margin: 4px 0;
        }
        
        .message-content blockquote {
            border-left: 3px solid var(--accent);
            padding-left: 12px;
            margin: 12px 0;
            color: var(--text-secondary);
        }
        
        .message-content table {
            border-collapse: collapse;
            width: 100%;
            margin: 12px 0;
        }
        
        .message-content th, .message-content td {
            border: 1px solid var(--border-color);
            padding: 8px 12px;
            text-align: left;
        }
        
        .message-content th {
            background: var(--bg-tertiary);
        }
        
        .typing-indicator {
            display: flex;
            gap: 4px;
            padding: 8px 0;
        }
        
        .typing-indicator span {
            width: 8px;
            height: 8px;
            background: var(--text-muted);
            border-radius: 50%;
            animation: bounce 1.4s infinite ease-in-out;
        }
        
        .typing-indicator span:nth-child(1) { animation-delay: -0.32s; }
        .typing-indicator span:nth-child(2) { animation-delay: -0.16s; }
        
        @keyframes bounce {
            0%, 80%, 100% { transform: scale(0); }
            40% { transform: scale(1); }
        }
        
        /* Input Area */
        .input-area {
            padding: 16px 20px 20px;
            border-top: 1px solid var(--border-color);
            background: var(--bg-secondary);
        }
        
        .input-container {
            max-width: 900px;
            margin: 0 auto;
            display: flex;
            gap: 12px;
            align-items: flex-end;
        }
        
        .input-wrapper {
            flex: 1;
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            display: flex;
            align-items: flex-end;
            transition: border-color 0.2s;
        }
        
        .input-wrapper:focus-within {
            border-color: var(--accent);
        }
        
        .message-input {
            flex: 1;
            background: none;
            border: none;
            color: var(--text-primary);
            padding: 14px 16px;
            font-size: 14px;
            line-height: 1.5;
            resize: none;
            min-height: 48px;
            max-height: 200px;
            font-family: inherit;
        }
        
        .message-input:focus {
            outline: none;
        }
        
        .message-input::placeholder {
            color: var(--text-muted);
        }
        
        .send-btn {
            background: var(--accent);
            border: none;
            color: white;
            width: 48px;
            height: 48px;
            border-radius: 10px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.2s;
            flex-shrink: 0;
        }
        
        .send-btn:hover:not(:disabled) {
            background: var(--accent-hover);
        }
        
        .send-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        .send-btn svg {
            width: 20px;
            height: 20px;
        }
        
        /* Settings Panel */
        .settings-panel {
            width: 320px;
            background: var(--bg-secondary);
            border-left: 1px solid var(--border-color);
            padding: 16px;
            display: none;
            overflow-y: auto;
        }
        
        .settings-panel.open {
            display: block;
        }
        
        .settings-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 20px;
        }
        
        .settings-header h3 {
            font-size: 14px;
            font-weight: 600;
        }
        
        .settings-group {
            margin-bottom: 20px;
        }
        
        .settings-label {
            font-size: 12px;
            font-weight: 500;
            color: var(--text-secondary);
            margin-bottom: 8px;
            display: block;
        }
        
        .settings-textarea {
            width: 100%;
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            padding: 10px;
            border-radius: 6px;
            font-size: 13px;
            font-family: inherit;
            resize: vertical;
            min-height: 100px;
        }
        
        .settings-textarea:focus {
            outline: none;
            border-color: var(--accent);
        }
        
        .settings-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
        }
        
        .settings-row label {
            font-size: 13px;
            color: var(--text-secondary);
        }
        
        .settings-slider {
            width: 120px;
            accent-color: var(--accent);
        }
        
        .settings-value {
            font-size: 12px;
            color: var(--text-muted);
            min-width: 40px;
            text-align: right;
        }
        
        /* Empty State */
        .empty-state {
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            color: var(--text-muted);
            text-align: center;
            padding: 40px;
        }
        
        .empty-state svg {
            width: 64px;
            height: 64px;
            margin-bottom: 16px;
            opacity: 0.5;
        }
        
        .empty-state h2 {
            font-size: 20px;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 8px;
        }
        
        .empty-state p {
            max-width: 400px;
            line-height: 1.6;
        }
        
        /* Responsive */
        @media (max-width: 768px) {
            .sidebar {
                display: none;
            }
            
            .settings-panel {
                position: absolute;
                right: 0;
                top: 0;
                bottom: 0;
                z-index: 100;
            }
        }
        
        /* Scrollbar */
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: var(--border-color); border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }
    </style>
</head>
<body>
    <div class="app">
        <!-- Sidebar -->
        <aside class="sidebar">
            <div class="sidebar-header">
                <h1><span>⚡</span> ZSE Chat</h1>
            </div>
            <button class="new-chat-btn" onclick="createNewChat()" style="margin: 12px;">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <line x1="12" y1="5" x2="12" y2="19"></line>
                    <line x1="5" y1="12" x2="19" y2="12"></line>
                </svg>
                New Chat
            </button>
            <div class="conversations-list" id="conversations-list">
                <!-- Conversations loaded dynamically -->
            </div>
            <div class="sidebar-footer">
                <button class="sidebar-btn" onclick="toggleTheme()">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <circle cx="12" cy="12" r="5"></circle>
                        <line x1="12" y1="1" x2="12" y2="3"></line>
                        <line x1="12" y1="21" x2="12" y2="23"></line>
                        <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line>
                        <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line>
                        <line x1="1" y1="12" x2="3" y2="12"></line>
                        <line x1="21" y1="12" x2="23" y2="12"></line>
                        <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line>
                        <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>
                    </svg>
                    Theme
                </button>
                <button class="sidebar-btn" onclick="window.location.href='/dashboard'">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <rect x="3" y="3" width="7" height="7"></rect>
                        <rect x="14" y="3" width="7" height="7"></rect>
                        <rect x="14" y="14" width="7" height="7"></rect>
                        <rect x="3" y="14" width="7" height="7"></rect>
                    </svg>
                    Dashboard
                </button>
            </div>
        </aside>
        
        <!-- Main Chat -->
        <main class="main">
            <header class="chat-header">
                <span class="chat-title" id="chat-title">New Chat</span>
                <div class="header-actions">
                    <select class="model-select" id="model-select">
                        <option value="">Select model...</option>
                    </select>
                    <button class="icon-btn" onclick="toggleSettings()" title="Settings">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <circle cx="12" cy="12" r="3"></circle>
                            <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
                        </svg>
                    </button>
                    <button class="icon-btn" onclick="deleteCurrentChat()" title="Delete">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <polyline points="3 6 5 6 21 6"></polyline>
                            <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                        </svg>
                    </button>
                </div>
            </header>
            
            <div class="messages" id="messages">
                <div class="empty-state" id="empty-state">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                        <path d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"></path>
                    </svg>
                    <h2>Start a conversation</h2>
                    <p>Select a model and type your message to begin chatting with your local LLM.</p>
                </div>
            </div>
            
            <div class="input-area">
                <div class="input-container">
                    <div class="input-wrapper">
                        <textarea 
                            class="message-input" 
                            id="message-input"
                            placeholder="Type your message..." 
                            rows="1"
                            onkeydown="handleKeyDown(event)"
                            oninput="autoResize(this)"
                        ></textarea>
                    </div>
                    <button class="send-btn" id="send-btn" onclick="sendMessage()">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <line x1="22" y1="2" x2="11" y2="13"></line>
                            <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
                        </svg>
                    </button>
                </div>
            </div>
        </main>
        
        <!-- Settings Panel -->
        <aside class="settings-panel" id="settings-panel">
            <div class="settings-header">
                <h3>Settings</h3>
                <button class="icon-btn" onclick="toggleSettings()">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <line x1="18" y1="6" x2="6" y2="18"></line>
                        <line x1="6" y1="6" x2="18" y2="18"></line>
                    </svg>
                </button>
            </div>
            
            <div class="settings-group">
                <label class="settings-label">System Prompt</label>
                <textarea 
                    class="settings-textarea" 
                    id="system-prompt"
                    placeholder="You are a helpful assistant..."
                    onchange="saveSettings()"
                ></textarea>
            </div>
            
            <div class="settings-group">
                <label class="settings-label">Parameters</label>
                <div class="settings-row">
                    <label>Temperature</label>
                    <input type="range" class="settings-slider" id="temperature" min="0" max="2" step="0.1" value="0.7" onchange="saveSettings()">
                    <span class="settings-value" id="temperature-value">0.7</span>
                </div>
                <div class="settings-row">
                    <label>Top P</label>
                    <input type="range" class="settings-slider" id="top-p" min="0" max="1" step="0.05" value="0.9" onchange="saveSettings()">
                    <span class="settings-value" id="top-p-value">0.9</span>
                </div>
                <div class="settings-row">
                    <label>Max Tokens</label>
                    <input type="range" class="settings-slider" id="max-tokens" min="64" max="4096" step="64" value="1024" onchange="saveSettings()">
                    <span class="settings-value" id="max-tokens-value">1024</span>
                </div>
            </div>
        </aside>
    </div>
    
    <script>
        // State
        let currentConversationId = null;
        let conversations = [];
        let isGenerating = false;
        let currentWs = null;
        
        // Initialize
        document.addEventListener('DOMContentLoaded', () => {
            loadConversations();
            loadModels();
            initSliders();
            
            // Check for saved theme
            if (localStorage.getItem('theme') === 'light') {
                document.body.classList.add('light-theme');
            }
        });
        
        // Theme
        function toggleTheme() {
            document.body.classList.toggle('light-theme');
            localStorage.setItem('theme', document.body.classList.contains('light-theme') ? 'light' : 'dark');
        }
        
        // Settings Panel
        function toggleSettings() {
            document.getElementById('settings-panel').classList.toggle('open');
        }
        
        function initSliders() {
            ['temperature', 'top-p', 'max-tokens'].forEach(id => {
                const slider = document.getElementById(id);
                const valueEl = document.getElementById(id + '-value');
                slider.addEventListener('input', () => {
                    valueEl.textContent = slider.value;
                });
            });
        }
        
        function saveSettings() {
            if (!currentConversationId) return;
            
            const settings = {
                system_prompt: document.getElementById('system-prompt').value,
                temperature: parseFloat(document.getElementById('temperature').value),
                top_p: parseFloat(document.getElementById('top-p').value),
                max_tokens: parseInt(document.getElementById('max-tokens').value),
            };
            
            fetch(`/api/chat/conversations/${currentConversationId}`, {
                method: 'PATCH',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    system_prompt: settings.system_prompt,
                    settings: settings
                })
            });
        }
        
        // Conversations
        async function loadConversations() {
            try {
                const res = await fetch('/api/chat/conversations?limit=50');
                const data = await res.json();
                conversations = data.conversations;
                renderConversations();
            } catch (e) {
                console.error('Failed to load conversations:', e);
            }
        }
        
        function renderConversations() {
            const list = document.getElementById('conversations-list');
            
            if (conversations.length === 0) {
                list.innerHTML = '<div style="padding: 20px; color: var(--text-muted); text-align: center; font-size: 13px;">No conversations yet</div>';
                return;
            }
            
            list.innerHTML = conversations.map(conv => `
                <div class="conversation-item ${conv.id === currentConversationId ? 'active' : ''}" 
                     onclick="selectConversation('${conv.id}')">
                    <div class="conversation-title">${escapeHtml(conv.title)}</div>
                    <div class="conversation-meta">
                        <span>${conv.message_count} messages</span>
                        <span>${formatDate(conv.updated_at)}</span>
                    </div>
                </div>
            `).join('');
        }
        
        async function createNewChat() {
            try {
                const model = document.getElementById('model-select').value;
                const res = await fetch('/api/chat/conversations', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        title: 'New Chat',
                        model: model || null
                    })
                });
                const conv = await res.json();
                await loadConversations();
                selectConversation(conv.id);
            } catch (e) {
                console.error('Failed to create conversation:', e);
            }
        }
        
        async function selectConversation(convId) {
            currentConversationId = convId;
            renderConversations();
            
            try {
                // Load conversation details
                const convRes = await fetch(`/api/chat/conversations/${convId}`);
                const conv = await convRes.json();
                
                document.getElementById('chat-title').textContent = conv.title;
                
                if (conv.model) {
                    document.getElementById('model-select').value = conv.model;
                }
                
                // Load settings
                document.getElementById('system-prompt').value = conv.system_prompt || '';
                if (conv.settings) {
                    if (conv.settings.temperature !== undefined) {
                        document.getElementById('temperature').value = conv.settings.temperature;
                        document.getElementById('temperature-value').textContent = conv.settings.temperature;
                    }
                    if (conv.settings.top_p !== undefined) {
                        document.getElementById('top-p').value = conv.settings.top_p;
                        document.getElementById('top-p-value').textContent = conv.settings.top_p;
                    }
                    if (conv.settings.max_tokens !== undefined) {
                        document.getElementById('max-tokens').value = conv.settings.max_tokens;
                        document.getElementById('max-tokens-value').textContent = conv.settings.max_tokens;
                    }
                }
                
                // Load messages
                const msgRes = await fetch(`/api/chat/conversations/${convId}/messages`);
                const msgData = await msgRes.json();
                renderMessages(msgData.messages);
                
            } catch (e) {
                console.error('Failed to load conversation:', e);
            }
        }
        
        async function deleteCurrentChat() {
            if (!currentConversationId) return;
            if (!confirm('Delete this conversation?')) return;
            
            try {
                await fetch(`/api/chat/conversations/${currentConversationId}`, { method: 'DELETE' });
                currentConversationId = null;
                await loadConversations();
                
                // Clear messages
                document.getElementById('messages').innerHTML = document.getElementById('empty-state').outerHTML;
                document.getElementById('chat-title').textContent = 'New Chat';
            } catch (e) {
                console.error('Failed to delete:', e);
            }
        }
        
        // Messages
        function renderMessages(messages) {
            const container = document.getElementById('messages');
            
            if (messages.length === 0) {
                container.innerHTML = `
                    <div class="empty-state">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                            <path d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"></path>
                        </svg>
                        <h2>Start a conversation</h2>
                        <p>Type your message below to begin chatting.</p>
                    </div>
                `;
                return;
            }
            
            container.innerHTML = messages.map(msg => createMessageHtml(msg.role, msg.content)).join('');
            container.scrollTop = container.scrollHeight;
            
            // Highlight code blocks
            container.querySelectorAll('pre code').forEach(block => {
                hljs.highlightElement(block);
            });
        }
        
        function createMessageHtml(role, content) {
            const avatar = role === 'user' ? '👤' : '⚡';
            const parsed = role === 'user' ? escapeHtml(content) : marked.parse(content);
            
            return `
                <div class="message ${role}">
                    <div class="message-avatar">${avatar}</div>
                    <div class="message-content">${parsed}</div>
                </div>
            `;
        }
        
        function addMessage(role, content) {
            const container = document.getElementById('messages');
            
            // Remove empty state if present
            const emptyState = container.querySelector('.empty-state');
            if (emptyState) emptyState.remove();
            
            container.insertAdjacentHTML('beforeend', createMessageHtml(role, content));
            container.scrollTop = container.scrollHeight;
            
            // Highlight code blocks
            container.querySelectorAll('pre code').forEach(block => {
                hljs.highlightElement(block);
            });
        }
        
        function addTypingIndicator() {
            const container = document.getElementById('messages');
            container.insertAdjacentHTML('beforeend', `
                <div class="message assistant" id="typing-message">
                    <div class="message-avatar">⚡</div>
                    <div class="message-content">
                        <div class="typing-indicator">
                            <span></span><span></span><span></span>
                        </div>
                    </div>
                </div>
            `);
            container.scrollTop = container.scrollHeight;
        }
        
        function updateAssistantMessage(content) {
            const typing = document.getElementById('typing-message');
            if (typing) {
                typing.querySelector('.message-content').innerHTML = marked.parse(content);
                typing.querySelectorAll('pre code').forEach(block => {
                    hljs.highlightElement(block);
                });
            }
            document.getElementById('messages').scrollTop = document.getElementById('messages').scrollHeight;
        }
        
        function finalizeAssistantMessage() {
            const typing = document.getElementById('typing-message');
            if (typing) {
                typing.removeAttribute('id');
            }
        }
        
        // Chat
        async function sendMessage() {
            const input = document.getElementById('message-input');
            const content = input.value.trim();
            
            if (!content || isGenerating) return;
            
            const model = document.getElementById('model-select').value;
            if (!model) {
                alert('Please select a model first');
                return;
            }
            
            // Create conversation if needed
            if (!currentConversationId) {
                await createNewChat();
            }
            
            // Get title from first message
            const isFirstMessage = document.querySelector('.message') === null;
            
            // Add user message to UI
            input.value = '';
            autoResize(input);
            addMessage('user', content);
            
            // Save user message
            await fetch(`/api/chat/conversations/${currentConversationId}/messages`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ role: 'user', content })
            });
            
            // Update title if first message
            if (isFirstMessage) {
                const title = content.slice(0, 50) + (content.length > 50 ? '...' : '');
                await fetch(`/api/chat/conversations/${currentConversationId}`, {
                    method: 'PATCH',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ title })
                });
                document.getElementById('chat-title').textContent = title;
                loadConversations();
            }
            
            // Start generation
            isGenerating = true;
            document.getElementById('send-btn').disabled = true;
            addTypingIndicator();
            
            // Stream response via WebSocket
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            currentWs = new WebSocket(`${protocol}//${window.location.host}/ws/chat`);
            
            let fullResponse = '';
            
            currentWs.onopen = () => {
                // Get messages for context
                fetch(`/api/chat/conversations/${currentConversationId}/messages`)
                    .then(res => res.json())
                    .then(data => {
                        const messages = [];
                        
                        // Add system prompt
                        const systemPrompt = document.getElementById('system-prompt').value;
                        if (systemPrompt) {
                            messages.push({ role: 'system', content: systemPrompt });
                        }
                        
                        // Add conversation history
                        data.messages.forEach(m => {
                            messages.push({ role: m.role, content: m.content });
                        });
                        
                        // Send request
                        currentWs.send(JSON.stringify({
                            model,
                            messages,
                            temperature: parseFloat(document.getElementById('temperature').value),
                            top_p: parseFloat(document.getElementById('top-p').value),
                            max_tokens: parseInt(document.getElementById('max-tokens').value),
                            stream: true
                        }));
                    });
            };
            
            currentWs.onmessage = (event) => {
                const data = JSON.parse(event.data);
                
                if (data.type === 'token') {
                    fullResponse += data.content;
                    updateAssistantMessage(fullResponse);
                } else if (data.type === 'done') {
                    finalizeAssistantMessage();
                    isGenerating = false;
                    document.getElementById('send-btn').disabled = false;
                    
                    // Save assistant message
                    fetch(`/api/chat/conversations/${currentConversationId}/messages`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            role: 'assistant',
                            content: fullResponse,
                            tokens: data.tokens,
                            latency_ms: data.latency_ms
                        })
                    });
                    
                    currentWs.close();
                } else if (data.type === 'error') {
                    updateAssistantMessage('Error: ' + data.message);
                    finalizeAssistantMessage();
                    isGenerating = false;
                    document.getElementById('send-btn').disabled = false;
                }
            };
            
            currentWs.onerror = () => {
                updateAssistantMessage('Connection error. Please try again.');
                finalizeAssistantMessage();
                isGenerating = false;
                document.getElementById('send-btn').disabled = false;
            };
        }
        
        // Models
        async function loadModels() {
            try {
                const res = await fetch('/v1/models');
                const data = await res.json();
                const select = document.getElementById('model-select');
                
                if (data.data && data.data.length > 0) {
                    select.innerHTML = '<option value="">Select model...</option>' +
                        data.data.map(m => `<option value="${m.id}">${m.id}</option>`).join('');
                } else {
                    select.innerHTML = '<option value="">No models loaded</option>';
                }
            } catch (e) {
                console.error('Failed to load models:', e);
            }
        }
        
        // Utilities
        function handleKeyDown(event) {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                sendMessage();
            }
        }
        
        function autoResize(el) {
            el.style.height = 'auto';
            el.style.height = Math.min(el.scrollHeight, 200) + 'px';
        }
        
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
        
        function formatDate(isoString) {
            const date = new Date(isoString);
            const now = new Date();
            const diff = now - date;
            
            if (diff < 60000) return 'Just now';
            if (diff < 3600000) return Math.floor(diff / 60000) + 'm ago';
            if (diff < 86400000) return Math.floor(diff / 3600000) + 'h ago';
            if (diff < 604800000) return Math.floor(diff / 86400000) + 'd ago';
            
            return date.toLocaleDateString();
        }
    </script>
</body>
</html>'''
