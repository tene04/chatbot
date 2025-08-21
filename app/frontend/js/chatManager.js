// Manage chat functionality

export class ChatManager {
	constructor(api, ui) {
		this.api = api;
		this.ui = ui;
		this.textInput = document.getElementById('textInput');
		this.sendBtn = document.getElementById('sendBtn');
		this.chatDisplay = document.getElementById('chatDisplay');
		this.isTyping = false;
		this.setupEventListeners();
		this.setupInputValidation();
	}

	// Sets up event listeners for send button, enter key, and right-click copy
	setupEventListeners() {
		this.sendBtn.addEventListener('click', () => this.sendMessage());
		this.textInput.addEventListener('keydown', (e) => {
			if (e.key === 'Enter' && !e.shiftKey) {
				e.preventDefault();
				this.sendMessage();
			}
		});

		this.chatDisplay.addEventListener('contextmenu', (e) => {
			if (e.target.closest('.message')) {
				e.preventDefault();
				const messageText = e.target.closest('.message').textContent;
				navigator.clipboard.writeText(messageText).then(() => {
					this.ui.showToast('Message copied', 'info');
				}).catch(() => {
					this.ui.showToast('Error while copying', 'error');
				});
			}
		});
	}

	// Sets up input validation and warns when near character limit
	setupInputValidation() {
		this.textInput.addEventListener('input', (e) => {
			const maxLength = this.textInput.getAttribute('maxlength');
			const currentLength = e.target.value.length;
			
			if (currentLength > maxLength * 0.9) {
				this.textInput.style.borderColor = '#ffc107';
			} else {
				this.textInput.style.borderColor = '#333';
			}
		});
	}

	// Returns formatted time string for message timestamps
	formatTime() {
		return new Date().toLocaleTimeString('es-ES', {
			hour: '2-digit',
			minute: '2-digit',
			second: '2-digit'
		});
	}

	// Adds a message (user or bot) to the chat display
	addMessage(text, sender, showTime = true) {
		const container = document.createElement('div');
		container.className = 'messageContainer ' + (sender === 'user' ? 'userMsgContainer' : 'botMsgContainer');
		const message = document.createElement('div');
		message.className = 'message ' + (sender === 'user' ? 'userMsg' : 'botMsg');
		
		let timeHtml = '';
		if (showTime) {
			timeHtml = `<div class="message-time"><i class="fas fa-clock"></i> ${this.formatTime()}</div>`;
		}

		message.innerHTML = text + timeHtml;
		container.appendChild(message);
		this.chatDisplay.appendChild(container);
		this.chatDisplay.scrollTop = this.chatDisplay.scrollHeight;
		const time = new Date();
		this.ui.addToMessageHistory(text, sender, time);
	}

	// Shows the typing indicator in the chat
	showTypingIndicator() {
		if (this.isTyping) return;
		this.isTyping = true;

		const container = document.createElement('div');
		container.className = 'messageContainer botMsgContainer';
		container.id = 'typingIndicator';
		const indicator = document.createElement('div');
		indicator.className = 'typing-indicator';
		indicator.innerHTML = `
			<div class="typing-dots">
				<div class="typing-dot"></div>
				<div class="typing-dot"></div>
				<div class="typing-dot"></div>
			</div>
		`;

		container.appendChild(indicator);
		this.chatDisplay.appendChild(container);
		this.chatDisplay.scrollTop = this.chatDisplay.scrollHeight;
	}

	// Hides the typing indicator
	hideTypingIndicator() {
		const indicator = document.getElementById('typingIndicator');
		if (indicator) {
			indicator.remove();
		}
		this.isTyping = false;
	}

	// Sends a message to the backend and displays the response
	async sendMessage() {
		const text = this.textInput.value.trim();
		if (!text || this.isTyping || !this.api.isConnected) {
			if (!this.api.isConnected) {
				this.ui.showToast('There is no connection with the backend', 'error');
			}
			return;
		}

		this.addMessage(text, 'user');
		this.textInput.value = '';
		this.sendBtn.disabled = true;
		this.showTypingIndicator();

		try {
			this.api.addLog(`Sending query: "${text.substring(0, 50)}..."`, 'info');
			const response = await this.api.sendQuery(text);

			this.hideTypingIndicator();
			this.addMessage(response.response, 'bot', true);
			this.api.addLog('Response received', 'success');
			
		} catch (error) {
			this.hideTypingIndicator();
			const errorMsg = `Error: ${error.message}`;
			this.addMessage(errorMsg, 'bot');
			this.api.addLog(`Error in query: ${error.message}`, 'error');
			this.ui.showToast(`Error while processing query: ${error.message}`, 'error');
		} finally {
			this.sendBtn.disabled = false;
		}
	}

	// Clears all messages from the chat display
	clearChat() {
		this.chatDisplay.innerHTML = '';
	}

	// Highlights messages containing the search query
	searchInChat(query) {
		const messages = this.chatDisplay.querySelectorAll('.message');
		messages.forEach(msg => {
			const text = msg.textContent.toLowerCase();
			if (text.includes(query.toLowerCase())) {
				msg.style.backgroundColor = 'rgba(255, 235, 59, 0.3)';
				setTimeout(() => {
					msg.style.backgroundColor = '';
				}, 3000);
			}
		});
	}

	// Returns chat display DOM element
	getChatDisplay() {
		return this.chatDisplay;
	}

	// Enables or disables input and send button
	setInputEnabled(enabled) {
		this.textInput.disabled = !enabled;
		this.sendBtn.disabled = !enabled;
	}
}
