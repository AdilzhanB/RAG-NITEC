// ========================================
// Глобальные переменные и конфигурация
// ========================================
const CONFIG = {
    API_BASE_URL: '/api/v2',
    MAX_MESSAGE_LENGTH: 2000,
    TYPING_DELAY: 1000,
    AUTO_SAVE_INTERVAL: 30000,
    NOTIFICATION_DURATION: 5000,
    CHART_COLORS: {
        primary: '#667eea',
        secondary: '#764ba2',
        success: '#10b981',
        warning: '#f59e0b',
        error: '#ef4444'
    }
};

// Состояние приложения
const AppState = {
    currentSection: 'chat',
    isTyping: false,
    currentInteractionId: null,
    messagesHistory: [],
    settings: {
        theme: 'light',
        language: 'ru',
        autoSave: true,
        soundNotifications: true,
        showTyping: true,
        autoComplete: true,
        temperature: 0.7,
        maxLength: 1500,
        useWebSearch: true,
        saveHistory: true,
        analytics: true,
        dataRetention: 30
    },
    analytics: {
        totalQueries: 0,
        satisfactionRate: 0,
        avgResponseTime: 0,
        avgRating: 0
    },
    documents: [],
    currentFeedback: null
};

// ========================================
// Утилиты и вспомогательные функции
// ========================================
class Utils {
    static formatTime(date) {
        return new Intl.DateTimeFormat('ru-RU', {
            hour: '2-digit',
            minute: '2-digit'
        }).format(date);
    }

    static formatDate(date) {
        return new Intl.DateTimeFormat('ru-RU', {
            day: '2-digit',
            month: '2-digit',
            year: 'numeric'
        }).format(date);
    }

    static formatFileSize(bytes) {
        const sizes = ['B', 'KB', 'MB', 'GB'];
        if (bytes === 0) return '0 B';
        const i = Math.floor(Math.log(bytes) / Math.log(1024));
        return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
    }

    static generateId() {
        return Math.random().toString(36).substr(2, 9);
    }

    static debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    static sanitizeHTML(str) {
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }

    static parseMarkdown(text) {
        return text
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/`(.*?)`/g, '<code>$1</code>')
            .replace(/\n/g, '<br>');
    }
}

// ========================================
// Система уведомлений
// ========================================
class NotificationSystem {
    constructor() {
        this.container = document.getElementById('notificationsContainer');
    }

    show(title, message, type = 'info', duration = CONFIG.NOTIFICATION_DURATION) {
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.innerHTML = `
            <div class="notification-icon">
                <i class="fas ${this.getIcon(type)}"></i>
            </div>
            <div class="notification-content">
                <div class="notification-title">${title}</div>
                <div class="notification-message">${message}</div>
            </div>
            <button class="notification-close">
                <i class="fas fa-times"></i>
            </button>
        `;

        // Добавляем обработчики
        const closeBtn = notification.querySelector('.notification-close');
        closeBtn.addEventListener('click', () => this.remove(notification));

        // Автоматическое удаление
        if (duration > 0) {
            setTimeout(() => this.remove(notification), duration);
        }

        this.container.appendChild(notification);
        
        // Анимация появления
        setTimeout(() => notification.classList.add('show'), 10);

        return notification;
    }

    getIcon(type) {
        const icons = {
            success: 'fa-check',
            error: 'fa-exclamation',
            warning: 'fa-exclamation-triangle',
            info: 'fa-info'
        };
        return icons[type] || icons.info;
    }

    remove(notification) {
        notification.style.animation = 'slideOutRight 0.3s ease';
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 300);
    }

    success(title, message) {
        return this.show(title, message, 'success');
    }

    error(title, message) {
        return this.show(title, message, 'error');
    }

    warning(title, message) {
        return this.show(title, message, 'warning');
    }

    info(title, message) {
        return this.show(title, message, 'info');
    }
}

// ========================================
// API клиент
// ========================================
class APIClient {
    static async request(endpoint, options = {}) {
        const url = `${CONFIG.API_BASE_URL}${endpoint}`;
        const defaultOptions = {
            headers: {
                'Content-Type': 'application/json',
            }
        };

        try {
            const response = await fetch(url, { ...defaultOptions, ...options });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            return data;
        } catch (error) {
            console.error('API Request failed:', error);
            throw error;
        }
    }

    static async query(query, userId = 'default', sessionId = null) {
        return this.request('/query', {
            method: 'POST',
            body: JSON.stringify({
                query,
                user_id: userId,
                session_id: sessionId,
                priority: 'medium'
            })
        });
    }

    static async feedback(interactionId, feedbackData) {
        return this.request('/feedback', {
            method: 'POST',
            body: JSON.stringify({
                interaction_id: interactionId,
                ...feedbackData
            })
        });
    }

    static async search(query, maxResults = 5) {
        return this.request('/search', {
            method: 'POST',
            body: JSON.stringify({
                query,
                max_results: maxResults,
                priority: 'medium'
            })
        });
    }

    static async getAnalytics() {
        return this.request('/analytics');
    }

    static async getHealth() {
        return this.request('/health');
    }

    static async optimizeSystem() {
        return this.request('/optimize', {
            method: 'POST'
        });
    }

    static async getSuggestions(partialQuery) {
        return this.request(`/suggestions/${encodeURIComponent(partialQuery)}`);
    }

    static async getFeedbackTrends(days = 30) {
        return this.request(`/feedback/trends?days=${days}`);
    }

    static async getUserHistory(userId, limit = 50) {
        return this.request(`/user/${userId}/history?limit=${limit}`);
    }
}

// ========================================
// Система чата
// ========================================
class ChatSystem {
    constructor() {
        this.messagesArea = document.getElementById('messagesArea');
        this.messageInput = document.getElementById('messageInput');
        this.sendBtn = document.getElementById('sendBtn');
        this.typingIndicator = document.getElementById('typingIndicator');
        this.charCount = document.getElementById('charCount');
        
        this.initializeEventListeners();
        this.setupAutoResize();
    }

    initializeEventListeners() {
        // Отправка сообщения
        this.sendBtn.addEventListener('click', () => this.sendMessage());
        
        // Enter для отправки, Shift+Enter для новой строки
        this.messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // Счетчик символов
        this.messageInput.addEventListener('input', () => {
            this.updateCharCounter();
            this.toggleSendButton();
        });

        // Чипы предложений
        document.querySelectorAll('.chip').forEach(chip => {
            chip.addEventListener('click', () => {
                const query = chip.dataset.query;
                this.messageInput.value = query;
                this.updateCharCounter();
                this.toggleSendButton();
                this.sendMessage();
            });
        });
    }

    setupAutoResize() {
        this.messageInput.addEventListener('input', () => {
            this.messageInput.style.height = 'auto';
            this.messageInput.style.height = Math.min(this.messageInput.scrollHeight, 120) + 'px';
        });
    }

    updateCharCounter() {
        const length = this.messageInput.value.length;
        this.charCount.textContent = length;
        
        if (length > CONFIG.MAX_MESSAGE_LENGTH * 0.9) {
            this.charCount.style.color = 'var(--error-color)';
        } else {
            this.charCount.style.color = 'var(--text-muted)';
        }
    }

    toggleSendButton() {
        const hasText = this.messageInput.value.trim().length > 0;
        const isUnderLimit = this.messageInput.value.length <= CONFIG.MAX_MESSAGE_LENGTH;
        this.sendBtn.disabled = !hasText || !isUnderLimit || AppState.isTyping;
    }

    async sendMessage() {
        const message = this.messageInput.value.trim();
        if (!message || AppState.isTyping) return;

        // Добавляем сообщение пользователя
        this.addMessage(message, 'user');
        
        // Очищаем поле ввода
        this.messageInput.value = '';
        this.messageInput.style.height = 'auto';
        this.updateCharCounter();
        this.toggleSendButton();

        // Показываем индикатор печати
        this.showTyping();

        try {
            // Отправляем запрос
            const response = await APIClient.query(message);
            
            // Скрываем индикатор печати
            this.hideTyping();
            
            // Добавляем ответ ассистента
            this.addMessage(response.answer, 'assistant', {
                sources: response.sources,
                confidence: response.confidence,
                interactionId: response.interaction_id
            });

            // Сохраняем ID взаимодействия для обратной связи
            AppState.currentInteractionId = response.interaction_id;

            // Обновляем аналитику
            AppState.analytics.totalQueries++;
            
        } catch (error) {
            this.hideTyping();
            notificationSystem.error('Ошибка', 'Не удалось получить ответ от сервера');
            console.error('Chat error:', error);
        }
    }

    addMessage(text, sender, metadata = {}) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;
        
        const time = Utils.formatTime(new Date());
        const messageId = Utils.generateId();
        
        messageDiv.innerHTML = `
            <div class="message-avatar">
                <i class="fas ${sender === 'user' ? 'fa-user' : 'fa-robot'}"></i>
            </div>
            <div class="message-content">
                <div class="message-bubble">
                    <div class="message-text">${Utils.parseMarkdown(text)}</div>
                    ${metadata.sources ? this.renderSources(metadata.sources) : ''}
                </div>
                <div class="message-meta">
                    <span class="message-time">${time}</span>
                    ${sender === 'assistant' ? this.renderMessageActions(messageId, metadata.interactionId) : ''}
                </div>
            </div>
        `;

        // Удаляем welcome message если есть
        const welcomeMessage = this.messagesArea.querySelector('.welcome-message');
        if (welcomeMessage) {
            welcomeMessage.remove();
        }

        this.messagesArea.appendChild(messageDiv);
        this.scrollToBottom();

        // Сохраняем в историю
        AppState.messagesHistory.push({
            id: messageId,
            text,
            sender,
            timestamp: new Date(),
            metadata
        });

        // Автосохранение
        if (AppState.settings.autoSave) {
            this.saveHistory();
        }
    }

    renderSources(sources) {
        if (!sources || sources.length === 0) return '';
        
        const sourceLinks = sources.map(source => 
            `<a href="#" class="source-link" data-source="${source}">${source}</a>`
        ).join('');
        
        return `
            <div class="message-sources">
                <div class="sources-title">Источники:</div>
                ${sourceLinks}
            </div>
        `;
    }

    renderMessageActions(messageId, interactionId) {
        return `
            <div class="message-actions">
                <button class="message-action" onclick="chatSystem.copyMessage('${messageId}')" title="Копировать">
                    <i class="fas fa-copy"></i>
                </button>
                <button class="message-action" onclick="feedbackSystem.openModal('${interactionId}')" title="Оценить">
                    <i class="fas fa-thumbs-up"></i>
                </button>
                <button class="message-action" onclick="chatSystem.shareMessage('${messageId}')" title="Поделиться">
                    <i class="fas fa-share"></i>
                </button>
            </div>
        `;
    }

    showTyping() {
        AppState.isTyping = true;
        this.typingIndicator.classList.add('show');
        this.toggleSendButton();
    }

    hideTyping() {
        AppState.isTyping = false;
        this.typingIndicator.classList.remove('show');
        this.toggleSendButton();
    }

    scrollToBottom() {
        setTimeout(() => {
            this.messagesArea.scrollTop = this.messagesArea.scrollHeight;
        }, 100);
    }

    copyMessage(messageId) {
        const message = AppState.messagesHistory.find(m => m.id === messageId);
        if (message) {
            navigator.clipboard.writeText(message.text);
            notificationSystem.success('Скопировано', 'Сообщение скопировано в буфер обмена');
        }
    }

    shareMessage(messageId) {
        const message = AppState.messagesHistory.find(m => m.id === messageId);
        if (message && navigator.share) {
            navigator.share({
                title: 'Сообщение из RAG AI',
                text: message.text
            });
        }
    }

    clearChat() {
        const messages = this.messagesArea.querySelectorAll('.message');
        messages.forEach(msg => msg.remove());
        
        AppState.messagesHistory = [];
        this.showWelcomeMessage();
        
        notificationSystem.success('Очищено', 'История чата была очищена');
    }

    showWelcomeMessage() {
        const welcomeHTML = `
            <div class="welcome-message">
                <div class="welcome-icon">
                    <i class="fas fa-robot"></i>
                </div>
                <h3>Добро пожаловать в RAG AI!</h3>
                <p>Я готов помочь с вопросами о государственных услугах Казахстана. Задайте любой вопрос!</p>
                <div class="suggestion-chips">
                    <button class="chip" data-query="Как получить справку о несудимости?">
                        <i class="fas fa-file-alt"></i>
                        Справка о несудимости
                    </button>
                    <button class="chip" data-query="Регистрация ИП в Казахстане">
                        <i class="fas fa-briefcase"></i>
                        Регистрация ИП
                    </button>
                    <button class="chip" data-query="Услуги на egov.kz">
                        <i class="fas fa-globe"></i>
                        Услуги egov.kz
                    </button>
                </div>
            </div>
        `;
        
        this.messagesArea.innerHTML = welcomeHTML;
        
        // Переинициализируем обработчики для чипов
        this.messagesArea.querySelectorAll('.chip').forEach(chip => {
            chip.addEventListener('click', () => {
                const query = chip.dataset.query;
                this.messageInput.value = query;
                this.updateCharCounter();
                this.toggleSendButton();
                this.sendMessage();
            });
        });
    }

    saveHistory() {
        localStorage.setItem('rag_chat_history', JSON.stringify(AppState.messagesHistory));
    }

    loadHistory() {
        const saved = localStorage.getItem('rag_chat_history');
        if (saved) {
            AppState.messagesHistory = JSON.parse(saved);
            this.renderHistory();
        }
    }

    renderHistory() {
        AppState.messagesHistory.forEach(message => {
            this.addMessage(message.text, message.sender, message.metadata);
        });
    }

    exportChat() {
        const chatData = {
            timestamp: new Date().toISOString(),
            messages: AppState.messagesHistory
        };
        
        const blob = new Blob([JSON.stringify(chatData, null, 2)], {
            type: 'application/json'
        });
        
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `rag_chat_${Utils.formatDate(new Date()).replace(/\./g, '-')}.json`;
        a.click();
        
        URL.revokeObjectURL(url);
        notificationSystem.success('Экспорт', 'История чата экспортирована');
    }
}

// ========================================
// Система обратной связи
// ========================================
class FeedbackSystem {
    constructor() {
        this.modal = document.getElementById('feedbackModal');
        this.currentInteractionId = null;
        this.currentRating = 0;
        this.currentType = null;
        
        this.initializeEventListeners();
    }

    initializeEventListeners() {
        // Закрытие модального окна
        document.getElementById('closeFeedbackModal').addEventListener('click', () => {
            this.closeModal();
        });

        document.getElementById('cancelFeedback').addEventListener('click', () => {
            this.closeModal();
        });

        // Оценка звездами
        document.querySelectorAll('.star').forEach(star => {
            star.addEventListener('click', (e) => {
                this.setRating(parseInt(e.target.dataset.rating));
            });
            
            star.addEventListener('mouseenter', (e) => {
                this.highlightStars(parseInt(e.target.dataset.rating));
            });
        });

        document.getElementById('starRating').addEventListener('mouseleave', () => {
            this.highlightStars(this.currentRating);
        });

        // Типы обратной связи
        document.querySelectorAll('.feedback-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.selectFeedbackType(e.target.closest('.feedback-btn').dataset.type);
            });
        });

        // Отправка обратной связи
        document.getElementById('submitFeedback').addEventListener('click', () => {
            this.submitFeedback();
        });

        // Закрытие по клику вне модального окна
        this.modal.addEventListener('click', (e) => {
            if (e.target === this.modal) {
                this.closeModal();
            }
        });
    }

    openModal(interactionId) {
        this.currentInteractionId = interactionId;
        this.resetForm();
        this.modal.classList.add('show');
        document.body.style.overflow = 'hidden';
    }

    closeModal() {
        this.modal.classList.remove('show');
        document.body.style.overflow = '';
        this.resetForm();
    }

    resetForm() {
        this.currentRating = 0;
        this.currentType = null;
        this.highlightStars(0);
        
        document.querySelectorAll('.feedback-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        
        document.getElementById('feedbackText').value = '';
        document.getElementById('feedbackTextSection').style.display = 'none';
    }

    setRating(rating) {
        this.currentRating = rating;
        this.highlightStars(rating);
    }

    highlightStars(rating) {
        document.querySelectorAll('.star').forEach((star, index) => {
            if (index < rating) {
                star.classList.add('active');
            } else {
                star.classList.remove('active');
            }
        });
    }

    selectFeedbackType(type) {
        this.currentType = type;
        
        document.querySelectorAll('.feedback-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        
        document.querySelector(`[data-type="${type}"]`).classList.add('active');
        
        // Показываем текстовое поле для определенных типов
        const showTextArea = ['dislike', 'correction', 'suggestion'].includes(type);
        document.getElementById('feedbackTextSection').style.display = showTextArea ? 'block' : 'none';
    }

    async submitFeedback() {
        if (!this.currentInteractionId) {
            notificationSystem.error('Ошибка', 'Не выбрано взаимодействие для оценки');
            return;
        }

        if (!this.currentType && this.currentRating === 0) {
            notificationSystem.error('Ошибка', 'Выберите тип обратной связи или поставьте оценку');
            return;
        }

        const feedbackData = {
            feedback_type: this.currentType,
            rating: this.currentRating || null,
            correction: null,
            suggestion: null,
            explanation: null
        };

        const feedbackText = document.getElementById('feedbackText').value.trim();
        if (feedbackText) {
            if (this.currentType === 'correction') {
                feedbackData.correction = feedbackText;
            } else if (this.currentType === 'suggestion') {
                feedbackData.suggestion = feedbackText;
            } else {
                feedbackData.explanation = feedbackText;
            }
        }

        try {
            await APIClient.feedback(this.currentInteractionId, feedbackData);
            notificationSystem.success('Спасибо!', 'Ваша обратная связь отправлена');
            this.closeModal();
            
            // Обновляем аналитику
            if (this.currentRating > 0) {
                AppState.analytics.avgRating = (AppState.analytics.avgRating + this.currentRating) / 2;
            }
            
        } catch (error) {
            notificationSystem.error('Ошибка', 'Не удалось отправить обратную связь');
            console.error('Feedback error:', error);
        }
    }
}

// ========================================
// Система управления документами
// ========================================
class DocumentSystem {
    constructor() {
        this.modal = document.getElementById('uploadModal');
        this.uploadArea = document.getElementById('uploadArea');
        this.fileInput = document.getElementById('fileInput');
        this.docsGrid = document.getElementById('docsGrid');
        this.uploadProgress = document.getElementById('uploadProgress');
        this.progressFill = document.getElementById('progressFill');
        this.progressText = document.getElementById('progressText');
        
        this.initializeEventListeners();
        this.loadDocuments();
    }

    initializeEventListeners() {
        // Открытие модального окна загрузки
        document.getElementById('uploadDocBtn').addEventListener('click', () => {
            this.openUploadModal();
        });

        // Закрытие модального окна
        document.getElementById('closeUploadModal').addEventListener('click', () => {
            this.closeUploadModal();
        });

        // Drag & Drop
        this.uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.uploadArea.classList.add('drag-over');
        });

        this.uploadArea.addEventListener('dragleave', () => {
            this.uploadArea.classList.remove('drag-over');
        });

        this.uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            this.uploadArea.classList.remove('drag-over');
            this.handleFiles(e.dataTransfer.files);
        });

        // Клик по области загрузки
        this.uploadArea.addEventListener('click', () => {
            this.fileInput.click();
        });

        // Выбор файлов
        this.fileInput.addEventListener('change', (e) => {
            this.handleFiles(e.target.files);
        });

        // Поиск и фильтрация документов
        document.getElementById('docsSearch').addEventListener('input', 
            Utils.debounce((e) => this.filterDocuments(e.target.value), 300)
        );

        document.getElementById('docsFilter').addEventListener('change', (e) => {
            this.filterDocuments(null, e.target.value);
        });
    }

    openUploadModal() {
        this.modal.classList.add('show');
        document.body.style.overflow = 'hidden';
    }

    closeUploadModal() {
        this.modal.classList.remove('show');
        document.body.style.overflow = '';
        this.resetUpload();
    }

    resetUpload() {
        this.fileInput.value = '';
        this.uploadProgress.style.display = 'none';
        this.progressFill.style.width = '0%';
        this.uploadArea.classList.remove('drag-over');
    }

    async handleFiles(files) {
        const validFiles = Array.from(files).filter(file => {
            const validTypes = ['application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'text/plain'];
            const maxSize = 10 * 1024 * 1024; // 10MB
            
            if (!validTypes.includes(file.type)) {
                notificationSystem.error('Ошибка', `Неподдерживаемый тип файла: ${file.name}`);
                return false;
            }
            
            if (file.size > maxSize) {
                notificationSystem.error('Ошибка', `Файл слишком большой: ${file.name}`);
                return false;
            }
            
            return true;
        });

        if (validFiles.length === 0) return;

        this.uploadProgress.style.display = 'block';
        
        for (let i = 0; i < validFiles.length; i++) {
            const file = validFiles[i];
            await this.uploadFile(file, i + 1, validFiles.length);
        }

        this.loadDocuments();
        notificationSystem.success('Успешно', `Загружено ${validFiles.length} файлов`);
        this.closeUploadModal();
    }

    async uploadFile(file, current, total) {
        return new Promise((resolve, reject) => {
            const formData = new FormData();
            formData.append('file', file);

            const xhr = new XMLHttpRequest();

            xhr.upload.onprogress = (e) => {
                if (e.lengthComputable) {
                    const progress = (e.loaded / e.total) * 100;
                    this.updateProgress(progress, `Загрузка ${file.name} (${current}/${total})`);
                }
            };

            xhr.onload = () => {
                if (xhr.status === 200) {
                    resolve(JSON.parse(xhr.responseText));
                } else {
                    reject(new Error(`Upload failed: ${xhr.statusText}`));
                }
            };

            xhr.onerror = () => reject(new Error('Upload failed'));

            // Имитируем загрузку (в реальном приложении здесь будет API endpoint)
            setTimeout(() => {
                this.updateProgress(100, `Файл ${file.name} загружен`);
                resolve({ success: true, filename: file.name });
            }, 1000 + Math.random() * 2000);
        });
    }

    updateProgress(progress, text) {
        this.progressFill.style.width = `${progress}%`;
        this.progressText.textContent = text;
    }

    loadDocuments() {
        // Имитация загрузки документов
        const mockDocuments = [
            {
                id: '1',
                name: 'Справка о несудимости.pdf',
                type: 'pdf',
                size: 245760,
                uploadDate: new Date('2024-01-15'),
                status: 'indexed'
            },
            {
                id: '2',
                name: 'Регистрация ИП.docx',
                type: 'docx',
                size: 102400,
                uploadDate: new Date('2024-01-10'),
                status: 'processing'
            },
            {
                id: '3',
                name: 'Услуги egov.txt',
                type: 'txt',
                size: 51200,
                uploadDate: new Date('2024-01-05'),
                status: 'indexed'
            }
        ];

        AppState.documents = mockDocuments;
        this.renderDocuments(mockDocuments);
        this.updateDocumentStats();
    }

    renderDocuments(documents) {
        this.docsGrid.innerHTML = '';

        documents.forEach(doc => {
            const docCard = document.createElement('div');
            docCard.className = 'doc-card';
            docCard.innerHTML = `
                <div class="doc-icon ${doc.type}">
                    <i class="fas ${this.getFileIcon(doc.type)}"></i>
                </div>
                <div class="doc-title" title="${doc.name}">${doc.name}</div>
                <div class="doc-meta">
                    <span class="doc-size">${Utils.formatFileSize(doc.size)}</span>
                    <span class="doc-date">${Utils.formatDate(doc.uploadDate)}</span>
                </div>
                <div class="doc-actions">
                    <button class="doc-action" onclick="documentSystem.viewDocument('${doc.id}')" title="Просмотр">
                        <i class="fas fa-eye"></i>
                    </button>
                    <button class="doc-action" onclick="documentSystem.downloadDocument('${doc.id}')" title="Скачать">
                        <i class="fas fa-download"></i>
                    </button>
                    <button class="doc-action delete" onclick="documentSystem.deleteDocument('${doc.id}')" title="Удалить">
                        <i class="fas fa-trash"></i>
                    </button>
                </div>
            `;

            this.docsGrid.appendChild(docCard);
        });
    }

    getFileIcon(type) {
        const icons = {
            pdf: 'fa-file-pdf',
            docx: 'fa-file-word',
            txt: 'fa-file-alt'
        };
        return icons[type] || 'fa-file';
    }

    filterDocuments(searchTerm, type = 'all') {
        let filtered = AppState.documents;

        if (searchTerm) {
            filtered = filtered.filter(doc => 
                doc.name.toLowerCase().includes(searchTerm.toLowerCase())
            );
        }

        if (type !== 'all') {
            filtered = filtered.filter(doc => doc.type === type);
        }

        this.renderDocuments(filtered);
    }

    updateDocumentStats() {
        const totalDocs = AppState.documents.length;
        const indexedDocs = AppState.documents.filter(doc => doc.status === 'indexed').length;
        const totalSize = AppState.documents.reduce((sum, doc) => sum + doc.size, 0);

        document.getElementById('totalDocs').textContent = totalDocs;
        document.getElementById('indexedDocs').textContent = indexedDocs;
        document.getElementById('docsSize').textContent = Utils.formatFileSize(totalSize);
    }

    viewDocument(docId) {
        const doc = AppState.documents.find(d => d.id === docId);
        if (doc) {
            notificationSystem.info('Просмотр', `Открытие документа: ${doc.name}`);
        }
    }

    downloadDocument(docId) {
        const doc = AppState.documents.find(d => d.id === docId);
        if (doc) {
            notificationSystem.success('Скачивание', `Началось скачивание: ${doc.name}`);
        }
    }

    deleteDocument(docId) {
        if (confirm('Вы уверены, что хотите удалить этот документ?')) {
            AppState.documents = AppState.documents.filter(d => d.id !== docId);
            this.renderDocuments(AppState.documents);
            this.updateDocumentStats();
            notificationSystem.success('Удалено', 'Документ был удален');
        }
    }
}

// ========================================
// Система аналитики
// ========================================
class AnalyticsSystem {
    constructor() {
        this.charts = {};
        this.initializeEventListeners();
    }

    initializeEventListeners() {
        document.getElementById('refreshAnalytics').addEventListener('click', () => {
            this.loadAnalytics();
        });

        document.getElementById('analyticsRange').addEventListener('change', (e) => {
            this.loadAnalytics(e.target.value);
        });
    }

    async loadAnalytics(days = 30) {
        try {
            // const data = await APIClient.getAnalytics();
            // Имитация данных аналитики
            const mockData = {
                totalQueries: 1247,
                satisfactionRate: 89,
                avgResponseTime: 1.2,
                avgRating: 4.3,
                queriesOverTime: this.generateTimeSeriesData(days),
                intentDistribution: {
                    'Справки': 35,
                    'Регистрация': 28,
                    'Платежи': 22,
                    'Консультации': 15
                },
                recentQueries: this.generateRecentQueries()
            };

            this.updateMetrics(mockData);
            this.updateCharts(mockData);
            this.updateRecentQueries(mockData.recentQueries);

        } catch (error) {
            notificationSystem.error('Ошибка', 'Не удалось загрузить аналитику');
            console.error('Analytics error:', error);
        }
    }

    updateMetrics(data) {
        document.getElementById('totalQueries').textContent = data.totalQueries.toLocaleString();
        document.getElementById('satisfactionRate').textContent = `${data.satisfactionRate}%`;
        document.getElementById('avgResponseTime').textContent = `${data.avgResponseTime}с`;
        document.getElementById('avgRating').textContent = data.avgRating.toFixed(1);

        // Обновляем глобальное состояние
        Object.assign(AppState.analytics, data);
    }

    updateCharts(data) {
        this.createQueriesChart(data.queriesOverTime);
        this.createIntentChart(data.intentDistribution);
    }

    createQueriesChart(data) {
        const ctx = document.getElementById('queriesChart').getContext('2d');
        
        if (this.charts.queries) {
            this.charts.queries.destroy();
        }

        this.charts.queries = new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.labels,
                datasets: [{
                    label: 'Запросы',
                    data: data.values,
                    borderColor: CONFIG.CHART_COLORS.primary,
                    backgroundColor: CONFIG.CHART_COLORS.primary + '20',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: {
                            color: 'var(--border-color)'
                        }
                    },
                    x: {
                        grid: {
                            color: 'var(--border-color)'
                        }
                    }
                }
            }
        });
    }

    createIntentChart(data) {
        const ctx = document.getElementById('intentChart').getContext('2d');
        
        if (this.charts.intent) {
            this.charts.intent.destroy();
        }

        this.charts.intent = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: Object.keys(data),
                datasets: [{
                    data: Object.values(data),
                    backgroundColor: [
                        CONFIG.CHART_COLORS.primary,
                        CONFIG.CHART_COLORS.secondary,
                        CONFIG.CHART_COLORS.success,
                        CONFIG.CHART_COLORS.warning
                    ]
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
    }

    updateRecentQueries(queries) {
        const tbody = document.querySelector('#recentQueriesTable tbody');
        tbody.innerHTML = '';

        queries.forEach(query => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${Utils.formatTime(query.timestamp)}</td>
                <td>${query.query}</td>
                <td>${query.intent}</td>
                <td>${query.rating ? '★'.repeat(query.rating) : '-'}</td>
                <td><span class="status-badge ${query.status}">${this.getStatusText(query.status)}</span></td>
            `;
            tbody.appendChild(row);
        });
    }

    getStatusText(status) {
        const texts = {
            success: 'Выполнено',
            warning: 'Частично',
            error: 'Ошибка'
        };
        return texts[status] || status;
    }

    generateTimeSeriesData(days) {
        const labels = [];
        const values = [];
        
        for (let i = days - 1; i >= 0; i--) {
            const date = new Date();
            date.setDate(date.getDate() - i);
            labels.push(Utils.formatDate(date));
            values.push(Math.floor(Math.random() * 50) + 10);
        }
        
        return { labels, values };
    }

    generateRecentQueries() {
        const queries = [
            'Как получить справку о несудимости?',
            'Регистрация ИП в Казахстане',
            'Услуги на egov.kz',
            'Получение паспорта',
            'Замена водительских прав'
        ];

        const intents = ['Справки', 'Регистрация', 'Консультации'];
        const statuses = ['success', 'warning', 'error'];

        return Array.from({ length: 10 }, (_, i) => ({
            timestamp: new Date(Date.now() - i * 3600000),
            query: queries[Math.floor(Math.random() * queries.length)],
            intent: intents[Math.floor(Math.random() * intents.length)],
            rating: Math.random() > 0.3 ? Math.floor(Math.random() * 5) + 1 : null,
            status: statuses[Math.floor(Math.random() * statuses.length)]
        }));
    }
}

// ========================================
// Система настроек
// ========================================
class SettingsSystem {
    constructor() {
        this.initializeEventListeners();
        this.loadSettings();
    }

    initializeEventListeners() {
        // Сохранение настроек
        document.getElementById('saveSettingsBtn').addEventListener('click', () => {
            this.saveSettings();
        });

        // Сброс настроек
        document.getElementById('resetSettingsBtn').addEventListener('click', () => {
            this.resetSettings();
        });

        // Переключение темы
        document.getElementById('themeToggle').addEventListener('click', () => {
            this.toggleTheme();
        });

        document.getElementById('themeSelect').addEventListener('change', (e) => {
            this.changeTheme(e.target.value);
        });

        // Обновление значений range
        document.querySelectorAll('input[type="range"]').forEach(range => {
            range.addEventListener('input', (e) => {
                const valueSpan = e.target.parentNode.querySelector('.range-value');
                if (valueSpan) {
                    let value = e.target.value;
                    if (e.target.id === 'temperature') {
                        value = (value / 100).toFixed(1);
                    }
                    valueSpan.textContent = value;
                }
            });
        });
    }

    loadSettings() {
        const saved = localStorage.getItem('rag_settings');
        if (saved) {
            Object.assign(AppState.settings, JSON.parse(saved));
        }

        this.applySettings();
    }

    applySettings() {
        // Применяем тему
        this.changeTheme(AppState.settings.theme);

        // Заполняем форму настроек
        Object.keys(AppState.settings).forEach(key => {
            const element = document.getElementById(key);
            if (element) {
                if (element.type === 'checkbox') {
                    element.checked = AppState.settings[key];
                } else if (element.type === 'range') {
                    if (key === 'temperature') {
                        element.value = AppState.settings[key] * 100;
                        element.parentNode.querySelector('.range-value').textContent = AppState.settings[key];
                    } else {
                        element.value = AppState.settings[key];
                        element.parentNode.querySelector('.range-value').textContent = AppState.settings[key];
                    }
                } else {
                    element.value = AppState.settings[key];
                }
            }
        });
    }

    saveSettings() {
        // Собираем данные из формы
        const formElements = document.querySelectorAll('#settingsSection input, #settingsSection select');
        
        formElements.forEach(element => {
            const key = element.id;
            if (key && AppState.settings.hasOwnProperty(key)) {
                if (element.type === 'checkbox') {
                    AppState.settings[key] = element.checked;
                } else if (element.type === 'range') {
                    let value = parseFloat(element.value);
                    if (key === 'temperature') {
                        value = value / 100;
                    }
                    AppState.settings[key] = value;
                } else if (element.type === 'number') {
                    AppState.settings[key] = parseInt(element.value);
                } else {
                    AppState.settings[key] = element.value;
                }
            }
        });

        // Сохраняем в localStorage
        localStorage.setItem('rag_settings', JSON.stringify(AppState.settings));
        
        // Применяем настройки
        this.applySettings();
        
        notificationSystem.success('Сохранено', 'Настройки успешно сохранены');
    }

    resetSettings() {
        if (confirm('Вы уверены, что хотите сбросить все настройки?')) {
            localStorage.removeItem('rag_settings');
            
            // Восстанавливаем настройки по умолчанию
            AppState.settings = {
                theme: 'light',
                language: 'ru',
                autoSave: true,
                soundNotifications: true,
                showTyping: true,
                autoComplete: true,
                temperature: 0.7,
                maxLength: 1500,
                useWebSearch: true,
                saveHistory: true,
                analytics: true,
                dataRetention: 30
            };
            
            this.applySettings();
            notificationSystem.success('Сброшено', 'Настройки сброшены к значениям по умолчанию');
        }
    }

    toggleTheme() {
        const currentTheme = AppState.settings.theme;
        const newTheme = currentTheme === 'light' ? 'dark' : 'light';
        this.changeTheme(newTheme);
        AppState.settings.theme = newTheme;
        this.saveSettings();
    }

    changeTheme(theme) {
        document.documentElement.setAttribute('data-theme', theme);
        
        // Обновляем иконку переключателя темы
        const themeIcon = document.querySelector('#themeToggle i');
        if (themeIcon) {
            themeIcon.className = theme === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
        }

        // Обновляем select
        const themeSelect = document.getElementById('themeSelect');
        if (themeSelect) {
            themeSelect.value = theme;
        }
    }
}

// ========================================
// Основная система навигации
// ========================================
class NavigationSystem {
    constructor() {
        this.currentSection = 'chat';
        this.initializeEventListeners();
    }

    initializeEventListeners() {
        // Навигация в боковой панели
        document.querySelectorAll('.nav-item').forEach(item => {
            item.addEventListener('click', () => {
                const section = item.dataset.section;
                this.navigateToSection(section);
            });
        });

        // Переключение боковой панели на мобильных
        document.getElementById('menuToggle').addEventListener('click', () => {
            this.toggleSidebar();
        });

        document.getElementById('sidebarToggle').addEventListener('click', () => {
            this.toggleSidebar();
        });

        // Быстрые действия
        document.getElementById('clearChatBtn').addEventListener('click', () => {
            chatSystem.clearChat();
        });

        document.getElementById('exportChatBtn').addEventListener('click', () => {
            chatSystem.exportChat();
        });

        // Кнопка настроек в заголовке
        document.getElementById('settingsBtn').addEventListener('click', () => {
            this.navigateToSection('settings');
        });

        // Закрытие боковой панели при клике вне ее на мобильных
        document.addEventListener('click', (e) => {
            const sidebar = document.getElementById('sidebar');
            const menuToggle = document.getElementById('menuToggle');
            
            if (window.innerWidth <= 768 && 
                !sidebar.contains(e.target) && 
                !menuToggle.contains(e.target) &&
                sidebar.classList.contains('show')) {
                this.toggleSidebar();
            }
        });
    }

    navigateToSection(sectionName) {
        // Убираем активный класс со всех секций и навигационных элементов
        document.querySelectorAll('.content-section').forEach(section => {
            section.classList.remove('active');
        });

        document.querySelectorAll('.nav-item').forEach(item => {
            item.classList.remove('active');
        });

        // Активируем нужную секцию
        const targetSection = document.getElementById(`${sectionName}Section`);
        const targetNavItem = document.querySelector(`[data-section="${sectionName}"]`);

        if (targetSection && targetNavItem) {
            targetSection.classList.add('active');
            targetNavItem.classList.add('active');
            
            this.currentSection = sectionName;
            this.updatePageTitle(sectionName);

            // Загружаем данные для секции
            this.loadSectionData(sectionName);

            // Закрываем боковую панель на мобильных
            if (window.innerWidth <= 768) {
                this.toggleSidebar(false);
            }
        }
    }

    updatePageTitle(sectionName) {
        const titles = {
            chat: { title: 'AI Ассистент', subtitle: 'Умный помощник с системой обратной связи' },
            analytics: { title: 'Аналитика', subtitle: 'Статистика и метрики системы' },
            documents: { title: 'Документы', subtitle: 'Управление базой знаний' },
            settings: { title: 'Настройки', subtitle: 'Конфигурация системы' }
        };

        const titleInfo = titles[sectionName];
        if (titleInfo) {
            document.getElementById('sectionTitle').textContent = titleInfo.title;
            document.getElementById('sectionSubtitle').textContent = titleInfo.subtitle;
        }
    }

    loadSectionData(sectionName) {
        switch (sectionName) {
            case 'analytics':
                if (typeof analyticsSystem !== 'undefined') {
                    analyticsSystem.loadAnalytics();
                }
                break;
            case 'documents':
                if (typeof documentSystem !== 'undefined') {
                    documentSystem.loadDocuments();
                }
                break;
        }
    }

    toggleSidebar(force = null) {
        const sidebar = document.getElementById('sidebar');
        
        if (force !== null) {
            sidebar.classList.toggle('show', force);
        } else {
            sidebar.classList.toggle('show');
        }
    }
}

// ========================================
// Система мониторинга состояния
// ========================================
class StatusSystem {
    constructor() {
        this.connectionStatus = document.getElementById('connectionStatus');
        this.currentTimeElement = document.getElementById('currentTime');
        
        this.startStatusMonitoring();
        this.startTimeUpdater();
    }

    startStatusMonitoring() {
        setInterval(async () => {
            try {
                await APIClient.getHealth();
                this.updateConnectionStatus(true);
            } catch (error) {
                this.updateConnectionStatus(false);
            }
        }, 30000); // Проверяем каждые 30 секунд
    }

    updateConnectionStatus(isConnected) {
        const statusDot = this.connectionStatus.querySelector('.status-dot');
        const statusText = this.connectionStatus.querySelector('span');
        
        if (isConnected) {
            statusDot.style.background = '#10b981';
            statusText.textContent = 'Подключено';
            this.connectionStatus.style.color = 'var(--text-secondary)';
        } else {
            statusDot.style.background = '#ef4444';
            statusText.textContent = 'Нет связи';
            this.connectionStatus.style.color = '#ef4444';
        }
    }

    startTimeUpdater() {
        const updateTime = () => {
            this.currentTimeElement.textContent = Utils.formatTime(new Date());
        };
        
        updateTime();
        setInterval(updateTime, 1000);
    }
}

// ========================================
// Система автосохранения и восстановления
// ========================================
class AutoSaveSystem {
    constructor() {
        this.startAutoSave();
        this.restoreData();
    }

    startAutoSave() {
        setInterval(() => {
            if (AppState.settings.autoSave) {
                this.saveAllData();
            }
        }, CONFIG.AUTO_SAVE_INTERVAL);

        // Сохранение при закрытии страницы
        window.addEventListener('beforeunload', () => {
            this.saveAllData();
        });
    }

    saveAllData() {
        try {
            localStorage.setItem('rag_app_state', JSON.stringify({
                currentSection: AppState.currentSection,
                messagesHistory: AppState.messagesHistory,
                settings: AppState.settings,
                analytics: AppState.analytics,
                documents: AppState.documents,
                timestamp: new Date().toISOString()
            }));
        } catch (error) {
            console.error('Auto-save failed:', error);
        }
    }

    restoreData() {
        try {
            const saved = localStorage.getItem('rag_app_state');
            if (saved) {
                const data = JSON.parse(saved);
                
                // Проверяем, не слишком ли старые данные
                const saveTime = new Date(data.timestamp);
                const now = new Date();
                const hoursDiff = (now - saveTime) / (1000 * 60 * 60);
                
                if (hoursDiff < 24) { // Восстанавливаем данные только если они не старше суток
                    Object.assign(AppState, data);
                }
            }
        } catch (error) {
            console.error('Data restoration failed:', error);
        }
    }
}

// ========================================
// Инициализация приложения
// ========================================
class App {
    constructor() {
        this.components = {};
        this.initialize();
    }

    async initialize() {
        try {
            // Инициализируем системы уведомлений первой
            this.components.notifications = new NotificationSystem();
            
            // Инициализируем остальные системы
            this.components.navigation = new NavigationSystem();
            this.components.chat = new ChatSystem();
            this.components.feedback = new FeedbackSystem();
            this.components.documents = new DocumentSystem();
            this.components.analytics = new AnalyticsSystem();
            this.components.settings = new SettingsSystem();
            this.components.status = new StatusSystem();
            this.components.autoSave = new AutoSaveSystem();

            // Делаем системы глобально доступными
            window.notificationSystem = this.components.notifications;
            window.chatSystem = this.components.chat;
            window.feedbackSystem = this.components.feedback;
            window.documentSystem = this.components.documents;
            window.analyticsSystem = this.components.analytics;
            window.settingsSystem = this.components.settings;

            // Загружаем первоначальные данные
            await this.loadInitialData();

            // Показываем приветственное сообщение
            this.components.notifications.success(
                'Система готова!', 
                'RAG AI успешно инициализирован и готов к работе'
            );

            console.log('🚀 RAG AI Application initialized successfully');

        } catch (error) {
            console.error('❌ Application initialization failed:', error);
            
            // Показываем ошибку инициализации
            if (this.components.notifications) {
                this.components.notifications.error(
                    'Ошибка инициализации', 
                    'Произошла ошибка при запуске приложения'
                );
            }
        }
    }

    async loadInitialData() {
        try {
            // Загружаем аналитику
            if (AppState.currentSection === 'analytics') {
                await this.components.analytics.loadAnalytics();
            }

            // Восстанавливаем историю чата если включено автосохранение
            if (AppState.settings.autoSave && AppState.settings.saveHistory) {
                this.components.chat.loadHistory();
            }

            // Если история пуста, показываем приветственное сообщение
            if (AppState.messagesHistory.length === 0) {
                this.components.chat.showWelcomeMessage();
            }

        } catch (error) {
            console.error('Failed to load initial data:', error);
        }
    }

    // Метод для ручной очистки данных (для отладки)
    clearAllData() {
        if (confirm('Вы уверены, что хотите очистить все данные приложения?')) {
            localStorage.clear();
            location.reload();
        }
    }

    // Метод для экспорта всех данных
    exportAllData() {
        const exportData = {
            timestamp: new Date().toISOString(),
            version: '2.0.0',
            appState: AppState,
            settings: AppState.settings,
            messagesHistory: AppState.messagesHistory,
            documents: AppState.documents
        };

        const blob = new Blob([JSON.stringify(exportData, null, 2)], {
            type: 'application/json'
        });

        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `rag_ai_export_${new Date().toISOString().split('T')[0]}.json`;
        a.click();

        URL.revokeObjectURL(url);
        this.components.notifications.success('Экспорт', 'Все данные приложения экспортированы');
    }
}

// ========================================
// Запуск приложения
// ========================================
document.addEventListener('DOMContentLoaded', () => {
    // Инициализируем приложение
    window.app = new App();

    // Добавляем глобальные обработчики
    window.addEventListener('error', (event) => {
        console.error('Global error:', event.error);
        if (window.notificationSystem) {
            window.notificationSystem.error('Ошибка', 'Произошла неожиданная ошибка');
        }
    });

    // Обработчик для неперехваченных промисов
    window.addEventListener('unhandledrejection', (event) => {
        console.error('Unhandled promise rejection:', event.reason);
        if (window.notificationSystem) {
            window.notificationSystem.error('Ошибка', 'Произошла ошибка при выполнении операции');
        }
    });

    // Делаем полезные функции доступными в консоли для отладки
    window.DEBUG = {
        clearData: () => window.app.clearAllData(),
        exportData: () => window.app.exportAllData(),
        state: () => AppState,
        config: () => CONFIG
    };
});

// Экспортируем основные классы для возможного использования в других модулях
window.RAG = {
    Utils,
    APIClient,
    NotificationSystem,
    ChatSystem,
    FeedbackSystem,
    DocumentSystem,
    AnalyticsSystem,
    SettingsSystem,
    NavigationSystem,
    StatusSystem,
    AutoSaveSystem,
    App
};