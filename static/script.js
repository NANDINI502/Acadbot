// ============================================================
// ACADBOT — CLIENT-SIDE LOGIC
// ============================================================

const MODES = ['welcome', 'thesis', 'visualizer', 'documents'];
let currentMode = 'welcome';
let currentChatId = null;
let docPath = ''; // Stored path for document Q&A

// ---- Migration ----
function migrateOldChats() {
    let sessions = getSessions();
    let migrated = false;
    ['thesis', 'visualizer', 'documents'].forEach(mode => {
        const oldChat = localStorage.getItem(`chat_${mode}`);
        if (oldChat) {
            const h = JSON.parse(oldChat);
            if (h.length > 0) {
                const id = 'chat_migrated_' + mode;
                const name = localStorage.getItem(`name_${mode}`);
                localStorage.setItem(`chat_msgs_${id}`, oldChat);
                sessions.push({ id, mode, name, timestamp: Date.now() });
                migrated = true;
            }
            localStorage.removeItem(`chat_${mode}`);
            localStorage.removeItem(`name_${mode}`);
        }
    });
    if (migrated) saveSessions(sessions);
}

// ---- Chat History (localStorage) ----
function getSessions() {
    return JSON.parse(localStorage.getItem('chat_sessions') || '[]');
}
function saveSessions(sessions) {
    localStorage.setItem('chat_sessions', JSON.stringify(sessions));
}
function getHistory(id) {
    if (!id) return [];
    return JSON.parse(localStorage.getItem(`chat_msgs_${id}`) || '[]');
}
function saveHistory(id, messages) {
    localStorage.setItem(`chat_msgs_${id}`, JSON.stringify(messages));
}
function addToHistory(mode, role, content, isImage = false) {
    if (!currentChatId) {
        currentChatId = 'chat_' + Date.now();
        const sessions = getSessions();
        sessions.unshift({ id: currentChatId, mode: mode, name: null, timestamp: Date.now() });
        saveSessions(sessions);
    }
    const history = getHistory(currentChatId);
    history.push({ role, content, isImage, timestamp: Date.now() });
    saveHistory(currentChatId, history);
}

// ---- Mode Switching ----
function switchMode(mode, chatId = null) {
    currentMode = mode;
    currentChatId = chatId;
    // Hide all views
    document.getElementById('welcome-screen').style.display = 'none';
    document.querySelectorAll('.chat-view').forEach(v => v.classList.remove('active'));

    if (mode === 'welcome') {
        document.getElementById('welcome-screen').style.display = 'flex';
        document.getElementById('top-bar-title').textContent = 'Welcome';
    } else {
        document.getElementById(`${mode}-view`).classList.add('active');
        const titles = { thesis: '📝 Thesis Writer', visualizer: '📊 Visualizer', documents: '📄 Document Reader' };
        document.getElementById('top-bar-title').textContent = titles[mode] || mode;
        loadChatHistory(mode);
        scrollToBottom(mode);
    }
    updateSidebar();
}

function loadChatHistory(mode) {
    const container = document.getElementById(`${mode}-messages`);
    container.innerHTML = '';
    const history = getHistory(currentChatId);
    history.forEach(msg => {
        appendMessage(mode, msg.role, msg.content, msg.isImage, false);
    });
}

function updateSidebar() {
    const list = document.getElementById('chat-history-list');
    list.innerHTML = '';
    const sessions = getSessions();
    sessions.forEach(session => {
        const mode = session.mode;
        const h = getHistory(session.id);
        if (h.length > 0) {
            let customName = session.name;
            const firstMsg = h.find(m => m.role === 'user');
            const defaultLabel = firstMsg ? firstMsg.content.substring(0, 35) + '...' : mode;
            const label = customName || defaultLabel;
            const icons = { thesis: '📝', visualizer: '📊', documents: '📄' };

            const item = document.createElement('div');
            item.className = 'chat-history-item';
            if (currentChatId === session.id) {
                item.style.background = 'var(--bg-hover)';
            }

            const nameSpan = document.createElement('span');
            nameSpan.className = 'chat-name';
            nameSpan.textContent = `${icons[mode]} ${label}`;
            nameSpan.onclick = () => switchMode(mode, session.id);

            const actionsDiv = document.createElement('div');
            actionsDiv.className = 'chat-actions';

            const renameBtn = document.createElement('button');
            renameBtn.innerHTML = '✏️';
            renameBtn.title = 'Rename Chat';
            renameBtn.onclick = (e) => {
                e.stopPropagation();
                let currentName = label;
                if (!customName && defaultLabel) currentName = defaultLabel;
                const newName = prompt(`Enter new name for ${icons[mode]} chat:`, currentName);
                if (newName !== null && newName.trim() !== '') {
                    session.name = newName.trim();
                    saveSessions(sessions);
                    updateSidebar();
                }
            };

            const deleteBtn = document.createElement('button');
            deleteBtn.innerHTML = '🗑️';
            deleteBtn.title = 'Delete Chat';
            deleteBtn.onclick = (e) => {
                e.stopPropagation();
                if (confirm('Are you sure you want to delete this chat history?')) {
                    localStorage.removeItem(`chat_msgs_${session.id}`);
                    const idx = sessions.findIndex(s => s.id === session.id);
                    if (idx > -1) sessions.splice(idx, 1);
                    saveSessions(sessions);
                    if (currentChatId === session.id) {
                        currentChatId = null;
                        switchMode('welcome');
                    } else {
                        updateSidebar();
                    }
                }
            };

            actionsDiv.appendChild(renameBtn);
            actionsDiv.appendChild(deleteBtn);

            item.appendChild(nameSpan);
            item.appendChild(actionsDiv);
            list.appendChild(item);
        }
    });
    if (list.children.length === 0) {
        list.innerHTML = '<div style="color:var(--text-muted); font-size:0.82rem; padding:8px;">No chats yet</div>';
    }
}

// ---- Message Rendering ----
function appendMessage(mode, role, content, isImage = false, save = true) {
    const container = document.getElementById(`${mode}-messages`);
    const wrapper = document.createElement('div');
    wrapper.className = `message ${role}`;

    const avatar = document.createElement('div');
    avatar.className = 'msg-avatar';
    avatar.textContent = role === 'user' ? '👤' : '🧬';

    const bubble = document.createElement('div');
    bubble.className = 'msg-bubble';

    if (isImage) {
        const img = document.createElement('img');
        img.src = content;
        img.alt = 'Generated';
        bubble.appendChild(img);

        // Download button
        const dl = document.createElement('a');
        dl.href = content;
        dl.download = 'generated_output.png';
        dl.className = 'download-btn';
        dl.textContent = '⬇ Download Image';
        bubble.appendChild(dl);
    } else {
        bubble.innerHTML = formatMarkdown(content);
    }

    wrapper.appendChild(avatar);
    wrapper.appendChild(bubble);
    container.appendChild(wrapper);

    if (save) addToHistory(mode, role, content, isImage);
    scrollToBottom(mode);
}

function showTyping(mode) {
    const container = document.getElementById(`${mode}-messages`);
    const wrapper = document.createElement('div');
    wrapper.className = 'message bot';
    wrapper.id = `${mode}-typing`;

    const avatar = document.createElement('div');
    avatar.className = 'msg-avatar';
    avatar.textContent = '🧬';

    const bubble = document.createElement('div');
    bubble.className = 'msg-bubble';
    bubble.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';

    wrapper.appendChild(avatar);
    wrapper.appendChild(bubble);
    container.appendChild(wrapper);
    scrollToBottom(mode);
}

function hideTyping(mode) {
    const el = document.getElementById(`${mode}-typing`);
    if (el) el.remove();
}

function scrollToBottom(mode) {
    const container = document.getElementById(`${mode}-messages`);
    if (container) container.scrollTop = container.scrollHeight;
}

function formatMarkdown(text) {
    // Basic markdown: bold, italic, code, links, headers
    return text
        .replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>')
        .replace(/`([^`]+)`/g, '<code style="background:var(--bg-card);padding:2px 6px;border-radius:4px;">$1</code>')
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.+?)\*/g, '<em>$1</em>')
        .replace(/^### (.+)$/gm, '<h4 style="margin:8px 0 4px;">$1</h4>')
        .replace(/^## (.+)$/gm, '<h3 style="margin:8px 0 4px;">$1</h3>')
        .replace(/^# (.+)$/gm, '<h2 style="margin:8px 0 4px;">$1</h2>')
        .replace(/\n/g, '<br>');
}

// ---- Send Handlers ----
async function sendThesis() {
    const input = document.getElementById('thesis-input');
    const prompt = input.value.trim();
    if (!prompt) return;

    input.value = '';
    appendMessage('thesis', 'user', prompt);
    showTyping('thesis');

    // Create a streaming bot message bubble
    const container = document.getElementById('thesis-messages');
    hideTyping('thesis');

    const wrapper = document.createElement('div');
    wrapper.className = 'message bot';
    const avatar = document.createElement('div');
    avatar.className = 'msg-avatar';
    avatar.textContent = '🧬';
    const bubble = document.createElement('div');
    bubble.className = 'msg-bubble';
    bubble.innerHTML = '';
    wrapper.appendChild(avatar);
    wrapper.appendChild(bubble);
    container.appendChild(wrapper);

    let fullLatex = '';
    let isLatex = false;
    let displayText = '';
    let sessionId = '';

    try {
        const evtSource = new EventSource(`/api/thesis/stream?topic=${encodeURIComponent(prompt)}`);

        evtSource.onmessage = function (event) {
            const chunk = event.data;

            if (chunk === '[DONE]') {
                evtSource.close();
                addToHistory('thesis', 'bot', displayText);
                updateSidebar();

                // Add download button — downloads .zip with LaTeX + figures
                if (fullLatex.trim()) {
                    const dlBtn = document.createElement('button');
                    dlBtn.className = 'download-btn';
                    dlBtn.textContent = '⬇ Download Thesis (.zip — LaTeX + Figures)';
                    dlBtn.style.cssText = 'margin-top:12px;padding:10px 20px;background:linear-gradient(135deg,#00f5d4,#3a86ff);color:#000;border:none;border-radius:8px;cursor:pointer;font-weight:600;font-size:0.9rem;';
                    dlBtn.onclick = async () => {
                        try {
                            const res = await fetch('/api/thesis/download', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({
                                    latex: fullLatex,
                                    session_id: sessionId,
                                    filename: 'thesis'
                                })
                            });
                            const blob = await res.blob();
                            const url = URL.createObjectURL(blob);
                            const a = document.createElement('a');
                            a.href = url;
                            a.download = 'thesis.zip';
                            a.click();
                            URL.revokeObjectURL(url);
                        } catch (err) {
                            alert('Download failed: ' + err.message);
                        }
                    };
                    bubble.appendChild(dlBtn);
                }
                scrollToBottom('thesis');
                return;
            }

            // Capture session ID for zip bundling
            if (chunk.startsWith('---SESSION:') && chunk.endsWith('---')) {
                sessionId = chunk.replace('---SESSION:', '').replace('---', '');
                return;
            }

            if (chunk === '---LATEX_START---') {
                isLatex = true;
                return;
            }

            // Decode escaped newlines back to real newlines
            const decoded = chunk.replace(/\\n/g, '\n');

            if (isLatex) {
                fullLatex += decoded;
                displayText = '```latex\n' + fullLatex + '\n```';
                bubble.innerHTML = '<div style="position:relative;">' +
                    '<div style="background:#1a1a2e;color:#00f5d4;padding:16px;border-radius:8px;font-family:monospace;font-size:0.85rem;white-space:pre-wrap;overflow-x:auto;max-height:500px;overflow-y:auto;">' +
                    escapeHtml(fullLatex) +
                    '</div></div>';
            } else {
                displayText += decoded + '\n';
                bubble.innerHTML = formatMarkdown(displayText);
            }
            scrollToBottom('thesis');
        };

        evtSource.onerror = function () {
            evtSource.close();
            if (!displayText) {
                bubble.innerHTML = formatMarkdown('🚨 Connection error during streaming.');
                addToHistory('thesis', 'bot', '🚨 Connection error during streaming.');
            }
        };

    } catch (e) {
        bubble.innerHTML = formatMarkdown('🚨 Connection error.');
        addToHistory('thesis', 'bot', '🚨 Connection error.');
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

async function sendVisualizer() {
    const input = document.getElementById('viz-input');
    const fileInput = document.getElementById('viz-file');
    const prompt = input.value.trim();
    if (!prompt) return;

    input.value = '';
    appendMessage('visualizer', 'user', prompt);
    showTyping('visualizer');

    try {
        const formData = new FormData();
        formData.append('prompt', prompt);
        if (fileInput.files.length > 0) {
            formData.append('file', fileInput.files[0]);
            fileInput.value = '';
        }

        const res = await fetch('/api/visualize', { method: 'POST', body: formData });
        const data = await res.json();
        hideTyping('visualizer');

        if (data.type === 'image') {
            appendMessage('visualizer', 'bot', data.url, true);
            if (data.references) {
                appendMessage('visualizer', 'bot', data.references);
            }
        } else {
            appendMessage('visualizer', 'bot', data.response);
        }
    } catch (e) {
        hideTyping('visualizer');
        appendMessage('visualizer', 'bot', '🚨 Connection error.');
    }
}

async function sendDocument() {
    const input = document.getElementById('doc-input');
    const fileInput = document.getElementById('doc-file');
    const prompt = input.value.trim();

    const formData = new FormData();

    // If file is selected, upload it
    if (fileInput.files.length > 0) {
        formData.append('file', fileInput.files[0]);
        formData.append('prompt', prompt || 'Summarize this document.');
        appendMessage('documents', 'user', `📄 Uploaded: ${fileInput.files[0].name}`);
        fileInput.value = '';
    } else if (prompt) {
        formData.append('prompt', prompt);
        formData.append('doc_path', docPath);
        appendMessage('documents', 'user', prompt);
    } else {
        return;
    }

    input.value = '';
    showTyping('documents');

    try {
        const res = await fetch('/api/document', { method: 'POST', body: formData });
        const data = await res.json();
        hideTyping('documents');

        if (data.doc_path) docPath = data.doc_path;
        appendMessage('documents', 'bot', data.response);
    } catch (e) {
        hideTyping('documents');
        appendMessage('documents', 'bot', '🚨 Connection error.');
    }
}

// ---- Key Handlers ----
function handleKey(e, mode) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        if (mode === 'thesis') sendThesis();
        else if (mode === 'visualizer') sendVisualizer();
        else if (mode === 'documents') sendDocument();
    }
}

// ---- File Upload Triggers ----
function triggerFileUpload(id) {
    document.getElementById(id).click();
}

// ---- New Chat ----
function newChat() {
    currentChatId = null;
    docPath = '';
    switchMode('welcome');
}

// ---- Init ----
document.addEventListener('DOMContentLoaded', () => {
    migrateOldChats();
    switchMode('welcome');
});
