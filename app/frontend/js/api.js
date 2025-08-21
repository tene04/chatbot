// class that works as a translator between the interface and the backend

export class ChatBotAPI {
	constructor() {
		this.isConnected = false;
		this.apiUrl = 'http://localhost:8000';
	}

	// Makes an API call to the backend with error handling
	async makeApiCall(endpoint, options = {}) {
		try {
			const url = `${this.apiUrl}${endpoint}`;
			const defaultOptions = {
				headers: {
					'Content-Type': 'application/json',
				}
			};
			
			const response = await fetch(url, { ...defaultOptions, ...options });
			
			if (!response.ok) {
				const errorText = await response.text();
				let errorData;
				try {
					errorData = JSON.parse(errorText);
				} catch {
					errorData = { detail: errorText || `HTTP ${response.status}` };
				}
				throw new Error(errorData.detail || errorData.message || `HTTP ${response.status}`);
			}
			
			return await response.json();
		} catch (error) {
			console.error('API Error:', error);
			throw error;
		}
	}

	// Checks if the backend connection is alive
	async checkConnection() {
		try {
			await this.makeApiCall('/health');
			if (!this.isConnected) {
				this.isConnected = true;
				this.updateConnectionStatus(true);
			}
			return true;
		} catch (error) {
			if (this.isConnected) {
				this.isConnected = false;
				this.updateConnectionStatus(false, error.message);
			}
			return false;
		}
	}

	// Updates the UI with connection status and logs it
	updateConnectionStatus(connected, errorMessage = '') {
		const statusIndicator = document.getElementById('statusIndicator');
		if (connected) {
			statusIndicator.className = 'status-indicator connected';
			statusIndicator.innerHTML = '<div class="status-dot"></div><span>Conectado</span>';
			this.addLog('Connection established with the backend', 'success');
		} else {
			statusIndicator.className = 'status-indicator disconnected';
			statusIndicator.innerHTML = '<div class="status-dot"></div><span>Desconectado</span>';
			this.addLog(`Connection error: ${errorMessage}`, 'error');
		}
	}

	// Sends a user query to the backend
	async sendQuery(query) {
		return await this.makeApiCall('/ask', {
			method: 'POST',
			body: JSON.stringify({ query })
		});
	}

	// Sends a document and save it on the server
	async uploadDocument(file) {
		const formData = new FormData();
		formData.append('file', file);
		const response = await fetch('/upload_document', {
			method: 'POST',
			body: formData 
		});
		return await response.json();
	}

	// Sends a document path to the backend for indexing
	async addDocument(filePath) {
		return await this.makeApiCall('/add_document', {
			method: 'POST',
			body: JSON.stringify({ file_path: filePath })
		});
	}

	// Requests the backend health check
	async getHealth() {
		return await this.makeApiCall('/health');
	}

	// Requests backend reinitialization
	async reinitialize() {
		return await this.makeApiCall('/reinitialize', {
			method: 'POST'
		});
	}

	// Adds a log entry to the system log UI
	addLog(text, type = 'info') {
		const logList = document.getElementById('logList');
		const logItem = document.createElement('div');
		logItem.className = `log-item ${type}`;
		logItem.innerHTML = `
			<strong>[${this.formatTime()}]</strong> ${text}
		`;
		logList.appendChild(logItem);
		logList.scrollTop = logList.scrollHeight;

		if (logList.children.length > 100) {
			logList.removeChild(logList.firstChild);
		}
	}

	// Formats current time as HH:MM:SS for logs
	formatTime() {
		return new Date().toLocaleTimeString('es-ES', {
			hour: '2-digit',
			minute: '2-digit',
			second: '2-digit'
		});
	}
}
