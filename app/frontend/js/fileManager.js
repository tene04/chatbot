// Manages file uploads and display within the UI
export class FileManager {
	constructor(api, ui) {
		this.api = api;
		this.ui = ui;
		this.files = [];
		
		this.chooseFileBtn = document.getElementById('chooseFileBtn');
		this.fileInput = document.getElementById('fileInput');
		this.fileList = document.getElementById('fileList');
		this.dragZone = document.getElementById('dragZone');
		
		this.setupEventListeners();
	}

	// Sets up all necessary event listeners for file interaction
	setupEventListeners() {
		this.chooseFileBtn.addEventListener('click', () => this.fileInput.click());
		this.dragZone.addEventListener('click', () => this.fileInput.click());

		this.fileInput.addEventListener('change', (e) => {
			Array.from(e.target.files).forEach(file => this.addFileToList(file));
			this.fileInput.value = '';
		});

		this.dragZone.addEventListener('dragover', (e) => {
			e.preventDefault();
			this.dragZone.classList.add('drag-over');
		});

		this.dragZone.addEventListener('dragleave', () => {
			this.dragZone.classList.remove('drag-over');
		});

		this.dragZone.addEventListener('drop', (e) => {
			e.preventDefault();
			this.dragZone.classList.remove('drag-over');
			Array.from(e.dataTransfer.files).forEach(file => this.addFileToList(file));
		});
	}

	// Formats a file size in bytes into a human-readable string
	formatFileSize(bytes) {
		if (bytes === 0) return '0 Bytes';
		const k = 1024;
		const sizes = ['Bytes', 'KB', 'MB', 'GB'];
		const i = Math.floor(Math.log(bytes) / Math.log(k));
		return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
	}

	// Gets the appropriate Font Awesome icon class based on the file extension
	getFileIcon(filename) {
		const ext = filename.split('.').pop().toLowerCase();
		const icons = {
			pdf: 'fas fa-file-pdf'
		};
		return icons[ext] || 'fas fa-file';
	}

	// Adds a file to the list for display and initiates the upload
	addFileToList(file) {
		const fileItem = document.createElement('div');
		fileItem.className = 'file-item uploading';
		fileItem.dataset.fileName = file.name;
		fileItem.innerHTML = `
            <i class="${this.getFileIcon(file.name)} file-icon"></i>
            <div class="file-info">
                <div class="file-name">${file.name}</div>
                <div class="file-size">${this.formatFileSize(file.size)}</div>
                <div class="file-status uploading">Uploading...</div>
                <div class="progress-bar-container">
                    <div class="progress-bar" style="width:0%"></div>
                </div>
            </div>
            <i class="fas fa-times file-remove" onclick="window.chatApp.fileManager.removeFile('${file.name}')"></i>
        `;

		this.fileList.appendChild(fileItem);
		this.files.push(file);

		this.uploadFileToBackend(file, fileItem);
		this.api.addLog(`Starting upload: ${file.name}`, 'info');
	}

	// Attempts to upload the file and processes the response
	async uploadFileToBackend(file, fileItem) {
        const statusElement = fileItem.querySelector('.file-status');
		const progressBar = fileItem.querySelector('.progress-bar');

		try {
            statusElement.textContent = 'Uploading to server...';
            progressBar.style.width = '25%';

            const uploadResult = await this.api.uploadDocument(file);

            statusElement.textContent = 'Processing document...';
            progressBar.style.width = '60%';
            const response = await this.api.addDocument(uploadResult.file_path);

            progressBar.style.width = '100%';
			statusElement.textContent = 'Processed successfully...';
            statusElement.className = 'file-status success';
            fileItem.classList.remove('uploading');

			setTimeout(() => {
				statusElement.remove(); 
			}, 1000);
			
			this.ui.showToast(`File "${file.name}" Processed successfully`, 'success');
			this.api.addLog(`Document processed: ${file.name}`, 'success');
			
		} catch (error) {
			progressBar.style.width = '100%';
            progressBar.style.background = '#dc3545';
            statusElement.textContent = `Error: ${error.message}`;
            statusElement.className = 'file-status error';
            fileItem.classList.remove('uploading');
            fileItem.classList.add('error');
            
            this.ui.showToast(`Error processing "${file.name}": ${error.message}`, 'error');
            this.api.addLog(`Error processing ${file.name}: ${error.message}`, 'error');
        }
	}

	// Removes a file from the list and the UI
	removeFile(fileName) {
		this.files = this.files.filter(f => f.name !== fileName);
		const fileItems = this.fileList.querySelectorAll('.file-item');
		for (let item of fileItems) {
			if (item.dataset.fileName === fileName) {
				this.fileList.removeChild(item);
				break;
			}
		}
		this.api.addLog(`Document eliminated: ${fileName}`, 'warning');
		this.ui.showToast(`File "${fileName}" eliminated`, 'warning');
	}

	// Returns the current list of files
	getFiles() {
		return this.files;
	}

	// Clears all files from the list and the UI
	clearFiles() {
		this.files = [];
		this.fileList.innerHTML = '';
	}
}