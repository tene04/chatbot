import { ChatBotAPI } from './api.js';
import { UIManager } from './ui.js';
import { FileManager } from './fileManager.js';
import { ChatManager } from './chatManager.js';

// Manages the overall application state and component interactions
class ChatBotApp {
	constructor() {
		this.api = null;
		this.ui = null;
		this.fileManager = null;
		this.chatManager = null;
	}

	// Initializes the application asynchronously
	async init() {
		try {
			console.log('Initializing ChatBot application...');

			this.api = new ChatBotAPI();
			this.ui = new UIManager();
			this.fileManager = new FileManager(this.api, this.ui);
			this.chatManager = new ChatManager(this.api, this.ui);

			await this.setupComponentInteractions();

			window.chatApp = this;

			this.startInitialChecks();

			console.log('Application initialized successfully');
			
		} catch (error) {
			console.error('Error initializing application:', error);
			this.ui?.showToast(`Error initializing application: ${error.message}`, 'error');
		}
	}

	// Sets up cross-component interactions
	async setupComponentInteractions() {
		await this.ui.setupHealthButton(this.api);
		await this.ui.setupReinitializeButton(this.api, this.chatManager);
		
		this.ui.createTestConnectionButton();
	}

	// Starts initial checks and periodic tasks
	startInitialChecks() {
		setTimeout(() => {
			this.api.addLog('Frontend started', 'success');
			this.api.addLog('Attempting connection to backend...', 'info');
			this.api.checkConnection();
		}, 500);

		setInterval(() => {
			this.api.checkConnection();
		}, 30000);
	}

	// Restarts the application
	async restart() {
		try {
			this.api.addLog('Restarting application...', 'warning');
			
			this.chatManager.clearChat();
			this.ui.messageHistory = [];
			
			await this.api.checkConnection();
			
			this.chatManager.addMessage('Application restarted! I am your RAG+LLM assistant ready to help you.', 'bot');
			
			this.api.addLog('Application restarted successfully', 'success');
			
		} catch (error) {
			this.api.addLog(`Error restarting application: ${error.message}`, 'error');
			this.ui.showToast(`Error restarting application: ${error.message}`, 'error');
		}
	}

	// Retrieves application statistics
	getStats() {
		return {
			totalMessages: this.ui.messageHistory.length,
			filesUploaded: this.fileManager.files.length,
			isConnected: this.api.isConnected,
			uptime: Date.now() - (window.appStartTime || Date.now())
		};
	}
}

document.addEventListener('DOMContentLoaded', async () => {
	window.appStartTime = Date.now();
	
	const app = new ChatBotApp();
	await app.init();

});

export default ChatBotApp;