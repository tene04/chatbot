// Manage UI functionality

export class UIManager {
    constructor() {
        this.sidebar = document.getElementById('sidebar');
        this.toggleSidebarBtn = document.getElementById('toggleSidebarBtn');
        this.exportBtn = document.getElementById('exportBtn');
        this.healthBtn = document.getElementById('healthBtn');
        this.reinitializeBtn = document.getElementById('reinitializeBtn');
        this.statusIndicator = document.getElementById('statusIndicator');
        this.apiUrl = 'http://localhost:8000';
        
        this.messageHistory = [];
        this.setupEventListeners();
    }

    // Main event listeners configuration
    setupEventListeners() {
        this.toggleSidebarBtn.addEventListener('click', () => {
            this.sidebar.classList.toggle('collapsed');
        });

        this.exportBtn.addEventListener('click', () => this.exportChat());

        this.statusIndicator.addEventListener('click', () => {
            if (window.chatApp && window.chatApp.api && !window.chatApp.api.isConnected) {
                window.chatApp.api.addLog('Retrying connection...', 'info');
                window.chatApp.api.checkConnection();
            }
        });

        document.addEventListener('keydown', (e) => this.handleKeyboardShortcuts(e));

        window.addEventListener('online', () => {
            if (window.chatApp && window.chatApp.api) {
                window.chatApp.api.addLog('Restored internet connection', 'success');
                window.chatApp.api.checkConnection();
            }
        });
        window.addEventListener('offline', () => {
            if (window.chatApp && window.chatApp.api) {
                window.chatApp.api.isConnected = false;
                this.statusIndicator.className = 'status-indicator disconnected';
                this.statusIndicator.innerHTML = '<div class="status-dot"></div><span>No internet</span>';
                window.chatApp.api.addLog('Lost internet connection', 'error');
            }
        });
    }

    // Export chat to a .txt file
    exportChat() {
        const chatText = this.messageHistory.map(msg => 
            `[${msg.time.toLocaleTimeString()}] ${msg.sender.toUpperCase()}: ${msg.text}`
        ).join('\n');
        
        const blob = new Blob([chatText], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `chat_export_${new Date().toISOString().split('T')[0]}.txt`;
        a.click();
        URL.revokeObjectURL(url);
        
        this.showToast('Chat exported successfully', 'success');
        if (window.chatApp && window.chatApp.api) {
            window.chatApp.api.addLog('chat exported to file', 'info');
        }
    }

    // Keyboard shortcuts (Ctrl/Cmd + K, Ctrl/Cmd + /, Escape)
    handleKeyboardShortcuts(e) {
        if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
            e.preventDefault();
            const textInput = document.getElementById('textInput');
            if (textInput) textInput.focus();
        }
        if ((e.ctrlKey || e.metaKey) && e.key === '/') {
            e.preventDefault();
            this.sidebar.classList.toggle('collapsed');
        }
        if (e.key === 'Escape') {
            const textInput = document.getElementById('textInput');
            if (document.activeElement === textInput && textInput) {
                textInput.value = '';
            }
        }
    }

    // Health Check button
    async setupHealthButton(api) {
        this.healthBtn.addEventListener('click', async () => {
            try {
                this.healthBtn.disabled = true;
                this.healthBtn.innerHTML = '<div class="loading-spinner"></div>Health Check';
                
                this.showToast('Checking system status...', 'info');
                api.addLog('Health check requested', 'info');
                
                const health = await api.getHealth();
                
                if (health.is_ready && health.status === 'healthy') {
                    this.statusIndicator.className = 'status-indicator connected';
                    this.statusIndicator.innerHTML = '<div class="status-dot"></div><span>Connected</span>';
                    this.showToast('System working properly', 'success');
                    api.addLog(`Health check: ${health.status} - Uptime: ${(health.uptime/3600).toFixed(1)}h`, 'success');
                    
                    if (health.rag_info || health.llm_info) {
                        api.addLog(`RAG: ${health.rag_info ? 'OK' : 'N/A'}, LLM: ${health.llm_info ? 'OK' : 'N/A'}`, 'info');
                    }
                } else {
                    this.statusIndicator.className = 'status-indicator disconnected';
                    this.statusIndicator.innerHTML = '<div class="status-dot"></div><span>Error</span>';
                    this.showToast(`System with problems: ${health.status}`, 'warning');
                    api.addLog(`Health check: ${health.status}`, 'warning');
                }
                
            } catch (error) {
                this.statusIndicator.className = 'status-indicator disconnected';
                this.statusIndicator.innerHTML = '<div class="status-dot"></div><span>Error</span>';
                this.showToast(`Error in health check: ${error.message}`, 'error');
                api.addLog(`Health check error: ${error.message}`, 'error');
            } finally {
                this.healthBtn.disabled = false;
                this.healthBtn.innerHTML = '<i class="fas fa-heart-pulse"></i>Health Check';
            }
        });
    }

    // Reinitialize button
    async setupReinitializeButton(api, chatManager) {
        this.reinitializeBtn.addEventListener('click', async () => {
            if (confirm('Are you sure you want to restart the system?')) {
                try {
                    this.reinitializeBtn.disabled = true;
                    this.reinitializeBtn.innerHTML = '<div class="loading-spinner"></div>Restarting...';
                    
                    this.statusIndicator.className = 'status-indicator loading';
                    this.statusIndicator.innerHTML = '<div class="loading-spinner"></div><span>Restarting...</span>';
                    this.showToast('Restarting system...', 'info');
                    api.addLog('Restarting started', 'warning');
                    
                    await api.reinitialize();
                    
                    chatManager.clearChat();
                    this.messageHistory = [];
                    
                    this.statusIndicator.className = 'status-indicator connected';
                    this.statusIndicator.innerHTML = '<div class="status-dot"></div><span>Connected</span>';
                    this.showToast('System restarted successfully', 'success');
                    api.addLog('Reset completed', 'success');
                    
                    chatManager.addMessage('System restarted! I am your chatbot assistance. You can upload documents and ask me about them.', 'bot');
                    
                } catch (error) {
                    this.showToast(`Error in reset: ${error.message}`, 'error');
                    api.addLog(`Error in reset: ${error.message}`, 'error');
                } finally {
                    this.reinitializeBtn.disabled = false;
                    this.reinitializeBtn.innerHTML = '<i class="fas fa-redo"></i>Reinitialize';
                }
            }
        });
    }

    // Show toast notifications
    showToast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.innerHTML = `
            <i class="fas fa-${type === 'success' ? 'check' : type === 'error' ? 'exclamation-triangle' : 'info-circle'}"></i>
            ${message}
        `;
        document.body.appendChild(toast);
        setTimeout(() => toast.classList.add('show'), 100);
        
        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => document.body.removeChild(toast), 300);
        }, 5000);
    }

    // Add message to history
    addToMessageHistory(text, sender, time) {
        this.messageHistory.push({ text, sender, time });
    }

    // Create a test connection button
    createTestConnectionButton() {
        const testConnectionBtn = document.createElement('button');
        testConnectionBtn.className = 'topBar-btn';
        testConnectionBtn.innerHTML = '<i class="fas fa-wifi"></i>Test';
        testConnectionBtn.onclick = () => {
            if (window.chatApp && window.chatApp.api) {
                window.chatApp.api.checkConnection();
                this.showToast('Testing connection...', 'info');
            }
        };
        const topBarRight = document.querySelector('.topBar-right');
        if (topBarRight) {
            topBarRight.insertBefore(testConnectionBtn, this.exportBtn);
        }
    }
}