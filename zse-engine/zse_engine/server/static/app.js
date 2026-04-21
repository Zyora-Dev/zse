/* ZSE Dashboard — Vanilla JavaScript */

// ==================== State ====================
let apiKey = localStorage.getItem('zse-api-key') || '';
let currentSession = null;
let isGenerating = false;

// ==================== Config ====================
const API_BASE = '';  // Same origin

// ==================== Navigation ====================
function switchPanel(panel) {
    document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
    document.getElementById('panel-' + panel).classList.add('active');
    document.querySelector(`.nav-item[data-panel="${panel}"]`).classList.add('active');

    if (panel === 'keys') loadKeys();
    if (panel === 'stats') loadStats();
    if (panel === 'lora') refreshLora();
    if (panel === 'rag') loadRAGDocuments();
}

// ==================== Toast ====================
function toast(message, type = 'success') {
    const container = document.getElementById('toast-container');
    const el = document.createElement('div');
    el.className = `toast ${type}`;
    el.textContent = message;
    container.appendChild(el);
    setTimeout(() => el.remove(), 4000);
}

// ==================== API ====================
async function api(method, path, body = null) {
    const opts = {
        method,
        headers: { 'Content-Type': 'application/json' },
    };
    if (apiKey) opts.headers['Authorization'] = `Bearer ${apiKey}`;
    if (body) opts.body = JSON.stringify(body);

    const resp = await fetch(API_BASE + path, opts);
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error?.message || `HTTP ${resp.status}`);
    return data;
}

function promptForKey() {
    const key = prompt('Enter your API key (or admin key for full access):');
    if (key) {
        apiKey = key;
        localStorage.setItem('zse-api-key', key);
        toast('API key saved');
        loadModels();
        loadLoraForSelect();
        loadSessions();
    }
}

// ==================== Sessions ====================
async function loadSessions() {
    try {
        if (!apiKey) return;
        const data = await api('GET', '/v1/dashboard/sessions');
        const list = document.getElementById('sessions-list');
        list.innerHTML = '';
        (data.sessions || []).forEach(s => {
            const div = document.createElement('div');
            div.className = 'session-item' + (s.session_id === currentSession ? ' active' : '');
            div.dataset.sessionId = s.session_id;

            const label = document.createElement('span');
            label.className = 'session-label';
            label.textContent = s.preview || 'New chat';
            label.onclick = () => loadSession(s.session_id);

            const del = document.createElement('span');
            del.className = 'session-delete';
            del.innerHTML = '&times;';
            del.title = 'Delete';
            del.onclick = (e) => { e.stopPropagation(); deleteSession(s.session_id); };

            div.appendChild(label);
            div.appendChild(del);
            list.appendChild(div);
        });
    } catch (e) {}
}

async function loadSession(sessionId) {
    try {
        const data = await api('GET', `/v1/dashboard/session/${sessionId}`);
        currentSession = sessionId;

        const msgs = document.getElementById('messages');
        msgs.innerHTML = '';

        (data.messages || []).forEach(m => {
            addMessage(m.role, m.content);
        });

        // Highlight active
        document.querySelectorAll('.session-item').forEach(s => {
            s.classList.toggle('active', s.dataset.sessionId === sessionId);
        });

        // Switch to chat panel
        switchPanel('chat');
    } catch (err) {
        toast(err.message, 'error');
    }
}

async function deleteSession(sessionId) {
    try {
        await api('DELETE', `/v1/dashboard/session/${sessionId}`);
        if (currentSession === sessionId) newChat();
        loadSessions();
        toast('Chat deleted');
    } catch (err) {
        toast(err.message, 'error');
    }
}

async function saveMessage(role, content) {
    try {
        if (!apiKey || !currentSession) return;
        const model = document.getElementById('model-select').value || '';
        const loraId = document.getElementById('lora-select').value || undefined;
        await api('POST', '/v1/dashboard/session/save', {
            session_id: currentSession,
            role, content, model,
            lora_id: loraId,
        });
    } catch (e) {}
}

// ==================== Chat ====================
function newChat() {
    currentSession = 'session-' + Date.now() + '-' + Math.random().toString(36).slice(2, 8);
    const msgs = document.getElementById('messages');
    msgs.innerHTML = `
        <div class="welcome" id="welcome">
            <h2>ZSE Inference Engine</h2>
            <p>Zero-dependency LLM serving. Ask anything.</p>
        </div>
    `;
    document.querySelectorAll('.session-item').forEach(s => s.classList.remove('active'));
    document.getElementById('chat-input').focus();
}

function addMessage(role, content) {
    // Hide welcome
    const welcome = document.getElementById('welcome');
    if (welcome) welcome.style.display = 'none';

    const messages = document.getElementById('messages');
    const div = document.createElement('div');
    div.className = `message ${role}`;

    const inner = document.createElement('div');
    inner.className = 'msg-content';
    inner.innerHTML = formatContent(content);
    div.appendChild(inner);

    messages.appendChild(div);
    messages.scrollTop = messages.scrollHeight;
    return inner;
}

function formatContent(text) {
    // Code blocks
    text = text.replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>');
    // Inline code
    text = text.replace(/`([^`]+)`/g, '<code>$1</code>');
    // Bold
    text = text.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    // Line breaks
    text = text.replace(/\n/g, '<br>');
    return text;
}

async function sendMessage() {
    if (isGenerating) return;

    const input = document.getElementById('chat-input');
    const text = input.value.trim();
    if (!text) return;

    if (!apiKey) {
        promptForKey();
        if (!apiKey) return;
    }

    if (!currentSession) newChat();

    // User message
    addMessage('user', text);
    saveMessage('user', text);
    input.value = '';
    input.style.height = 'auto';

    // Typing indicator
    const typingDiv = document.createElement('div');
    typingDiv.className = 'typing-indicator';
    typingDiv.innerHTML = '<span></span><span></span><span></span>';
    document.getElementById('messages').appendChild(typingDiv);
    document.getElementById('messages').scrollTop = document.getElementById('messages').scrollHeight;

    isGenerating = true;
    document.getElementById('send-btn').disabled = true;

    const model = document.getElementById('model-select').value || undefined;
    const loraId = document.getElementById('lora-select').value || undefined;
    const temperature = parseFloat(document.getElementById('temp-input').value) || 0.7;
    const useRag = document.getElementById('rag-toggle').checked;

    try {
        // Collect context
        const chatMessages = [];
        document.querySelectorAll('.message').forEach(m => {
            const content = m.querySelector('.msg-content');
            if (!content) return;
            if (m.classList.contains('user')) {
                chatMessages.push({ role: 'user', content: content.textContent });
            } else if (m.classList.contains('assistant')) {
                chatMessages.push({ role: 'assistant', content: content.textContent });
            }
        });

        // Stream
        const response = await fetch(API_BASE + '/v1/chat/completions', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${apiKey}`,
            },
            body: JSON.stringify({
                model, messages: chatMessages, temperature,
                stream: true, lora_id: loraId, max_tokens: 1024,
                rag: useRag,
            }),
        });

        typingDiv.remove();

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.error?.message || `HTTP ${response.status}`);
        }

        const contentDiv = addMessage('assistant', '');
        let fullText = '';

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop();

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const data = line.slice(6);
                    if (data === '[DONE]') break;
                    try {
                        const chunk = JSON.parse(data);
                        const delta = chunk.choices?.[0]?.delta;
                        if (delta?.content) {
                            fullText += delta.content;
                            contentDiv.innerHTML = formatContent(fullText);
                            document.getElementById('messages').scrollTop =
                                document.getElementById('messages').scrollHeight;
                        }
                    } catch (e) {}
                }
            }
        }

        // Save assistant response
        if (fullText) {
            saveMessage('assistant', fullText);
            loadSessions();
        }

    } catch (err) {
        typingDiv.remove();
        toast(err.message, 'error');
    } finally {
        isGenerating = false;
        document.getElementById('send-btn').disabled = false;
        input.focus();
    }
}

// Auto-resize textarea
document.addEventListener('DOMContentLoaded', () => {
    const input = document.getElementById('chat-input');
    if (input) {
        input.addEventListener('input', () => {
            input.style.height = 'auto';
            input.style.height = Math.min(input.scrollHeight, 160) + 'px';
        });
    }
    loadModels();
    loadLoraForSelect();
    loadSessions();
    newChat();
});

// ==================== Models ====================
async function loadModels() {
    try {
        if (!apiKey) return;
        const data = await api('GET', '/v1/models');
        const select = document.getElementById('model-select');
        select.innerHTML = '';
        (data.data || []).forEach(m => {
            const opt = document.createElement('option');
            opt.value = m.id;
            opt.textContent = m.id;
            select.appendChild(opt);
        });
    } catch (e) {}
}

async function loadLoraForSelect() {
    try {
        if (!apiKey) return;
        const data = await api('GET', '/v1/lora/list');
        const select = document.getElementById('lora-select');
        select.innerHTML = '<option value="">none</option>';
        (data.adapters || []).forEach(a => {
            const opt = document.createElement('option');
            opt.value = a.id;
            opt.textContent = `${a.id} (r=${a.rank})`;
            select.appendChild(opt);
        });
    } catch (e) {}
}

// ==================== API Keys ====================
function showCreateKey() {
    const form = document.getElementById('create-key-form');
    form.style.display = form.style.display === 'none' ? 'block' : 'none';
}

async function createKey() {
    try {
        const data = await api('POST', '/v1/admin/keys/create', {
            name: document.getElementById('key-name').value,
            rate_limit_rpm: parseInt(document.getElementById('key-rpm').value) || 60,
            rate_limit_tpm: parseInt(document.getElementById('key-tpm').value) || 100000,
        });
        document.getElementById('new-key-value').textContent = data.key;
        document.getElementById('new-key-display').style.display = 'block';
        document.getElementById('create-key-form').style.display = 'none';
        toast('API key created');
        loadKeys();
    } catch (err) {
        toast(err.message, 'error');
    }
}

async function loadKeys() {
    try {
        const data = await api('GET', '/v1/admin/keys/list');
        const tbody = document.getElementById('keys-table');
        tbody.innerHTML = '';
        (data.keys || []).forEach(k => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td><code>${k.prefix}</code></td>
                <td>${k.name || '-'}</td>
                <td><span class="badge ${k.is_active ? 'badge-active' : 'badge-inactive'}">${k.is_active ? 'Active' : 'Revoked'}</span></td>
                <td>${k.rate_limit_rpm} req/min</td>
                <td>${k.total_requests} reqs / ${formatTokens(k.total_tokens)}</td>
                <td>${formatDate(k.created_at)}</td>
                <td>${k.is_active ? `<button class="btn btn-danger btn-sm" onclick="revokeKey(${k.id})">Revoke</button>` : ''}</td>
            `;
            tbody.appendChild(tr);
        });
    } catch (err) {
        toast(err.message, 'error');
    }
}

async function revokeKey(id) {
    if (!confirm('Revoke this API key?')) return;
    try {
        await api('POST', '/v1/admin/keys/revoke', { id });
        toast('Key revoked');
        loadKeys();
    } catch (err) {
        toast(err.message, 'error');
    }
}

// ==================== Stats ====================
async function loadStats() {
    try {
        const data = await api('GET', '/v1/admin/stats');
        const grid = document.getElementById('stats-grid');
        grid.innerHTML = '';

        const engine = data.engine || {};
        const usage = data.usage || {};
        const keys = data.api_keys || {};

        const stats = [
            { label: 'Tokens/sec', value: engine.tokens_per_sec || 0, color: 'green' },
            { label: 'Requests/sec', value: engine.requests_per_sec || 0, color: 'blue' },
            { label: 'Active Requests', value: engine.active_requests || 0, color: 'blue' },
            { label: 'Queue Depth', value: engine.queue_depth || 0, color: 'yellow' },
            { label: 'Avg TTFT', value: (engine.avg_ttft_ms || 0) + 'ms', color: '' },
            { label: 'Avg TPOT', value: (engine.avg_tpot_ms || 0) + 'ms', color: '' },
            { label: 'Memory', value: ((engine.memory_utilization || 0) * 100).toFixed(1) + '%', color: 'yellow' },
            { label: 'Total Tokens', value: formatTokens(usage.total_tokens || 0), color: '' },
            { label: 'API Keys', value: `${keys.active || 0} / ${keys.total || 0}`, color: '' },
            { label: 'Uptime', value: formatUptime(engine.uptime_s || 0), color: '' },
        ];

        stats.forEach(s => {
            const card = document.createElement('div');
            card.className = 'stat-card';
            card.innerHTML = `
                <div class="label">${s.label}</div>
                <div class="value ${s.color}">${s.value}</div>
            `;
            grid.appendChild(card);
        });

        document.getElementById('stats-uptime').textContent =
            `Uptime: ${formatUptime(engine.uptime_s || 0)}`;
    } catch (err) {
        toast(err.message, 'error');
    }
}

setInterval(() => {
    if (document.getElementById('panel-stats').classList.contains('active')) loadStats();
}, 3000);

// ==================== LoRA ====================
function showLoadLora() {
    const form = document.getElementById('load-lora-form');
    form.style.display = form.style.display === 'none' ? 'block' : 'none';
}

async function loadLora() {
    try {
        const data = await api('POST', '/v1/lora/load', {
            adapter_id: document.getElementById('lora-id').value,
            path: document.getElementById('lora-path').value,
        });
        toast(`Adapter loaded: ${data.adapter_id}`);
        document.getElementById('load-lora-form').style.display = 'none';
        refreshLora();
        loadLoraForSelect();
    } catch (err) {
        toast(err.message, 'error');
    }
}

async function refreshLora() {
    try {
        const data = await api('GET', '/v1/lora/list');
        const tbody = document.getElementById('lora-table');
        tbody.innerHTML = '';
        (data.adapters || []).forEach(a => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td><code>${a.id}</code></td>
                <td>${a.rank}</td>
                <td>${a.alpha}</td>
                <td>${a.num_layers}</td>
                <td>${(a.target_modules || []).join(', ')}</td>
                <td>${formatBytes(a.total_gpu_bytes || 0)}</td>
                <td><button class="btn btn-danger btn-sm" onclick="unloadLora('${a.id}')">Unload</button></td>
            `;
            tbody.appendChild(tr);
        });
    } catch (err) {
        toast(err.message, 'error');
    }
}

async function unloadLora(id) {
    if (!confirm(`Unload adapter "${id}"?`)) return;
    try {
        await api('POST', '/v1/lora/unload', { adapter_id: id });
        toast(`Adapter unloaded: ${id}`);
        refreshLora();
        loadLoraForSelect();
    } catch (err) {
        toast(err.message, 'error');
    }
}

// ==================== RAG ====================
async function uploadDocument(file) {
    if (!file) return;
    if (!apiKey) { promptForKey(); if (!apiKey) return; }

    const progress = document.getElementById('rag-upload-progress');
    const progressText = document.getElementById('rag-upload-text');
    progress.style.display = 'block';
    progressText.textContent = `Uploading ${file.name}...`;

    try {
        const content = await fileToBase64(file);
        const data = await api('POST', '/v1/rag/upload', {
            filename: file.name,
            content: content,
        });
        progress.style.display = 'none';
        toast(`Uploaded: ${file.name} — ${data.chunk_count} chunks, ${data.token_savings_pct}% token savings`);
        loadRAGDocuments();
    } catch (err) {
        progress.style.display = 'none';
        toast(err.message, 'error');
    }
}

function fileToBase64(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => {
            const base64 = reader.result.split(',')[1];
            resolve(base64);
        };
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
}

async function loadRAGDocuments() {
    try {
        if (!apiKey) return;
        const data = await api('GET', '/v1/rag/documents');
        const tbody = document.getElementById('rag-table');
        tbody.innerHTML = '';
        (data.documents || []).forEach(d => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td><code>${escapeHtml(d.name)}</code></td>
                <td>${d.doc_type}</td>
                <td>${d.chunk_count}</td>
                <td>${formatTokens(d.original_tokens)}</td>
                <td>${formatTokens(d.compressed_tokens)}</td>
                <td><span class="savings-badge">${d.token_savings_pct}%</span></td>
                <td><button class="btn btn-danger btn-sm" onclick="deleteRAGDocument(${d.id})">Delete</button></td>
            `;
            tbody.appendChild(tr);
        });

        // Update stats badge
        const stats = await api('GET', '/v1/rag/stats');
        const badge = document.getElementById('rag-stats-badge');
        if (stats.document_count > 0) {
            badge.textContent = `${stats.document_count} docs · ${stats.chunk_count} chunks · ${stats.token_savings_pct}% savings`;
        } else {
            badge.textContent = '';
        }
    } catch (e) {}
}

async function deleteRAGDocument(id) {
    if (!confirm('Delete this document and all its chunks?')) return;
    try {
        await api('DELETE', `/v1/rag/document/${id}`);
        toast('Document deleted');
        loadRAGDocuments();
    } catch (err) {
        toast(err.message, 'error');
    }
}

async function searchRAG() {
    const input = document.getElementById('rag-search-input');
    const query = input.value.trim();
    if (!query) return;
    if (!apiKey) { promptForKey(); if (!apiKey) return; }

    const results = document.getElementById('rag-search-results');
    results.innerHTML = '<div style="color:var(--text-dim);font-size:13px;padding:8px 0">Searching...</div>';

    try {
        const data = await api('POST', '/v1/rag/search', { query, top_k: 5 });
        if (!data.results || data.results.length === 0) {
            results.innerHTML = '<div style="color:var(--text-dim);font-size:13px;padding:8px 0">No results found.</div>';
            return;
        }
        results.innerHTML = '';
        data.results.forEach(r => {
            const div = document.createElement('div');
            div.className = 'search-result';
            div.innerHTML = `
                <div class="search-result-header">
                    <span class="search-result-source">${escapeHtml(r.doc_name)} · chunk ${r.chunk_index}</span>
                    <span class="search-result-score">${(r.score * 100).toFixed(1)}%</span>
                </div>
                <div class="search-result-content">${escapeHtml(r.content.substring(0, 300))}</div>
            `;
            results.appendChild(div);
        });
    } catch (err) {
        results.innerHTML = '';
        toast(err.message, 'error');
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ==================== Helpers ====================
function formatTokens(n) {
    if (n >= 1e9) return (n / 1e9).toFixed(1) + 'B';
    if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
    if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
    return n.toString();
}

function formatBytes(b) {
    if (b >= 1073741824) return (b / 1073741824).toFixed(1) + ' GB';
    if (b >= 1048576) return (b / 1048576).toFixed(1) + ' MB';
    if (b >= 1024) return (b / 1024).toFixed(1) + ' KB';
    return b + ' B';
}

function formatDate(ts) {
    if (!ts) return '-';
    return new Date(ts * 1000).toLocaleDateString();
}

function formatUptime(seconds) {
    if (seconds < 60) return Math.round(seconds) + 's';
    if (seconds < 3600) return Math.round(seconds / 60) + 'm';
    if (seconds < 86400) return (seconds / 3600).toFixed(1) + 'h';
    return (seconds / 86400).toFixed(1) + 'd';
}
