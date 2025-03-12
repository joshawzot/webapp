// Initialize on document load
document.addEventListener('DOMContentLoaded', function() {
    // Initialize file explorer
    FileExplorer.init();
});

// File Explorer module
const FileExplorer = (function() {
    // State
    const state = {
        currentPath: '',
        selectedItem: null,
        openFiles: {},  // Map of file paths to their content
        activeFile: null,
        editor: null,
        fileModified: false
    };

    // DOM elements
    let elements = {};

    // Initialize
    function init() {
        // Get DOM elements
        elements = {
            fileTree: document.getElementById('file-tree'),
            currentPath: document.getElementById('current-path'),
            goPathBtn: document.getElementById('go-path-btn'),
            editorTabs: document.getElementById('editor-tabs'),
            editorContent: document.getElementById('editor-content'),
            fileInfo: document.getElementById('file-info'),
            saveFileBtn: document.getElementById('save-file-btn'),
            newFileBtn: document.getElementById('new-file-btn'),
            newDirectoryBtn: document.getElementById('new-directory-btn'),
            refreshExplorerBtn: document.getElementById('refresh-explorer-btn'),
            homeDirBtn: document.getElementById('home-dir-btn'),
            createFileBtn: document.getElementById('create-file-btn'),
            createDirectoryBtn: document.getElementById('create-directory-btn'),
            newFileDirectory: document.getElementById('new-file-directory'),
            newFileName: document.getElementById('new-file-name'),
            newDirectoryParent: document.getElementById('new-directory-parent'),
            newDirectoryName: document.getElementById('new-directory-name')
        };

        // Setup event listeners
        setupEventListeners();

        // Initialize Monaco Editor
        initMonacoEditor();

        // Load home directory
        loadHomeDirectory();
    }

    // Set up event listeners
    function setupEventListeners() {
        // Path navigation
        elements.goPathBtn.addEventListener('click', function() {
            navigateToPath(elements.currentPath.value);
        });
        
        elements.currentPath.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                navigateToPath(elements.currentPath.value);
            }
        });

        // Button actions
        elements.saveFileBtn.addEventListener('click', saveCurrentFile);
        elements.newFileBtn.addEventListener('click', showNewFileModal);
        elements.newDirectoryBtn.addEventListener('click', showNewDirectoryModal);
        elements.refreshExplorerBtn.addEventListener('click', refreshExplorer);
        elements.homeDirBtn.addEventListener('click', loadHomeDirectory);
        
        // Modal actions
        elements.createFileBtn.addEventListener('click', createNewFile);
        elements.createDirectoryBtn.addEventListener('click', createNewDirectory);
    }

    // Initialize Monaco Editor
    function initMonacoEditor() {
        // Configure Monaco loader
        require.config({ paths: { 'vs': 'https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.37.1/min/vs' } });
        
        // Load Monaco
        require(['vs/editor/editor.main'], function() {
            // Create editor container
            const editorContainer = document.createElement('div');
            editorContainer.id = 'editor-container';
            elements.editorContent.innerHTML = '';
            elements.editorContent.appendChild(editorContainer);
            
            // Create editor
            state.editor = monaco.editor.create(editorContainer, {
                language: 'plaintext',
                theme: 'vs',
                automaticLayout: true,
                minimap: { enabled: false }
            });
            
            // Listen for content changes
            state.editor.onDidChangeModelContent(function() {
                if (state.activeFile) {
                    setFileModified(true);
                }
            });
        });
    }

    // Navigate to path
    function navigateToPath(path) {
        fetchDirectoryContents(path);
    }

    // Load home directory
    function loadHomeDirectory() {
        // Load user's home directory
        fetchDirectoryContents('~');
    }

    // Refresh current explorer view
    function refreshExplorer() {
        fetchDirectoryContents(state.currentPath);
    }

    // Fetch directory contents from server
    function fetchDirectoryContents(path) {
        const fileTree = elements.fileTree;
        fileTree.innerHTML = '<div class="loading">Loading...</div>';
        
        fetch('/api/list-directory', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ path: path })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                state.currentPath = data.current_path;
                elements.currentPath.value = data.current_path;
                
                renderFileTree(data.items);
            } else {
                fileTree.innerHTML = `<div class="error">${data.message}</div>`;
            }
        })
        .catch(error => {
            fileTree.innerHTML = `<div class="error">Error: ${error.message}</div>`;
        });
    }

    // Render file tree from items
    function renderFileTree(items) {
        const fileTree = elements.fileTree;
        fileTree.innerHTML = '';
        
        // Sort items (directories first, then files)
        items.sort((a, b) => {
            if (a.type !== b.type) {
                return a.type === 'directory' ? -1 : 1;
            }
            return a.name.localeCompare(b.name);
        });
        
        // Add parent directory entry if not at root
        if (state.currentPath !== '/') {
            const parentItem = document.createElement('div');
            parentItem.className = 'file-item directory-item';
            parentItem.innerHTML = `
                <span class="icon"><i class="fas fa-arrow-up"></i></span>
                <span class="file-name">..</span>
            `;
            parentItem.addEventListener('click', function() {
                const parentPath = state.currentPath.split('/').slice(0, -1).join('/') || '/';
                navigateToPath(parentPath);
            });
            fileTree.appendChild(parentItem);
        }
        
        // Add items
        items.forEach(item => {
            const itemElement = document.createElement('div');
            
            if (item.type === 'directory') {
                // Directory item
                itemElement.className = 'file-item directory-item';
                itemElement.innerHTML = `
                    <span class="icon"><i class="fas fa-folder"></i></span>
                    <span class="file-name">${item.name}</span>
                `;
                itemElement.addEventListener('click', function() {
                    navigateToPath(item.path);
                });
            } else {
                // File item
                itemElement.className = 'file-item';
                itemElement.dataset.path = item.path;
                
                // Choose icon based on file type
                let iconClass = 'fas fa-file';
                if (item.icon) {
                    if (item.icon === 'python') {
                        iconClass = 'fab fa-python';
                    } else if (item.icon === 'javascript') {
                        iconClass = 'fab fa-js';
                    } else if (item.icon === 'html') {
                        iconClass = 'fab fa-html5';
                    } else if (item.icon === 'css') {
                        iconClass = 'fab fa-css3-alt';
                    } else if (item.icon === 'image') {
                        iconClass = 'far fa-image';
                    } else if (item.icon === 'pdf') {
                        iconClass = 'far fa-file-pdf';
                    }
                }
                
                itemElement.innerHTML = `
                    <span class="icon"><i class="${iconClass}"></i></span>
                    <span class="file-name">${item.name}</span>
                `;
                
                itemElement.addEventListener('click', function() {
                    openFile(item.path);
                });
            }
            
            fileTree.appendChild(itemElement);
        });
    }

    // Open file
    function openFile(path) {
        // Check if file is already open
        if (state.openFiles[path]) {
            setActiveFile(path);
            return;
        }
        
        // Fetch file content
        fetch('/api/read-file', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ path: path })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Add file to open files
                state.openFiles[path] = {
                    content: data.content,
                    type: data.file_type,
                    name: data.file_name
                };
                
                // Create tab
                createTab(path);
                
                // Set as active file
                setActiveFile(path);
                
                // Update file info
                elements.fileInfo.textContent = `${path} | ${formatFileSize(data.size)} | ${data.modified}`;
            } else {
                alert(data.message);
            }
        })
        .catch(error => {
            alert(`Error opening file: ${error.message}`);
        });
    }

    // Create a new tab for an open file
    function createTab(path) {
        const fileData = state.openFiles[path];
        
        // Check if tab already exists
        if (document.querySelector(`.vs-tab[data-path="${path}"]`)) {
            return;
        }
        
        // Remove placeholder if present
        const placeholder = elements.editorTabs.querySelector('.vs-tab-placeholder');
        if (placeholder) {
            placeholder.remove();
        }
        
        // Create tab element
        const tab = document.createElement('div');
        tab.className = 'vs-tab';
        tab.dataset.path = path;
        tab.innerHTML = `
            <span class="vs-tab-name">${fileData.name}</span>
            <span class="vs-tab-close"><i class="fas fa-times"></i></span>
        `;
        
        // Add click handler to select tab
        tab.addEventListener('click', function(e) {
            if (!e.target.closest('.vs-tab-close')) {
                setActiveFile(path);
            }
        });
        
        // Add close handler
        tab.querySelector('.vs-tab-close').addEventListener('click', function(e) {
            e.stopPropagation();
            closeFile(path);
        });
        
        // Add tab to tabs container
        elements.editorTabs.appendChild(tab);
    }

    // Set active file
    function setActiveFile(path) {
        const fileData = state.openFiles[path];
        
        // Update state
        state.activeFile = path;
        
        // Set active tab
        const tabs = elements.editorTabs.querySelectorAll('.vs-tab');
        tabs.forEach(tab => tab.classList.remove('active'));
        const activeTab = elements.editorTabs.querySelector(`.vs-tab[data-path="${path}"]`);
        if (activeTab) {
            activeTab.classList.add('active');
        }
        
        // Set editor content and language
        if (state.editor) {
            const model = monaco.editor.createModel(
                fileData.content,
                fileData.type
            );
            state.editor.setModel(model);
            setFileModified(false);
        }
    }

    // Close file
    function closeFile(path) {
        // Check if file is modified
        if (path === state.activeFile && state.fileModified) {
            if (!confirm('This file has unsaved changes. Close anyway?')) {
                return;
            }
        }
        
        // Remove from state
        delete state.openFiles[path];
        
        // Remove tab
        const tab = elements.editorTabs.querySelector(`.vs-tab[data-path="${path}"]`);
        if (tab) {
            tab.remove();
        }
        
        // Handle active file
        if (path === state.activeFile) {
            state.activeFile = null;
            
            // Set new active file if there are still open files
            const openPaths = Object.keys(state.openFiles);
            if (openPaths.length > 0) {
                setActiveFile(openPaths[0]);
            } else {
                // No open files, clear editor
                if (state.editor) {
                    state.editor.setModel(null);
                }
                
                // Reset save button
                elements.saveFileBtn.disabled = true;
                
                // Clear file info
                elements.fileInfo.textContent = '';
                
                // Add placeholder
                const placeholder = document.createElement('div');
                placeholder.className = 'vs-tab-placeholder';
                placeholder.textContent = 'No files open';
                elements.editorTabs.appendChild(placeholder);
            }
        }
    }

    // Save current file
    function saveCurrentFile() {
        if (!state.activeFile || !state.editor) {
            return;
        }
        
        const content = state.editor.getValue();
        
        fetch('/api/save-file', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                path: state.activeFile,
                content: content
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Update state
                state.openFiles[state.activeFile].content = content;
                setFileModified(false);
                
                // Show success message
                showMessage('File saved successfully', 'success');
            } else {
                showMessage(`Error saving file: ${data.message}`, 'error');
            }
        })
        .catch(error => {
            showMessage(`Error saving file: ${error.message}`, 'error');
        });
    }

    // Set file modified state
    function setFileModified(modified) {
        state.fileModified = modified;
        elements.saveFileBtn.disabled = !modified;
        
        // Mark tab as modified
        if (state.activeFile) {
            const tab = elements.editorTabs.querySelector(`.vs-tab[data-path="${state.activeFile}"]`);
            if (tab) {
                const tabName = tab.querySelector('.vs-tab-name');
                if (tabName) {
                    const fileName = state.openFiles[state.activeFile].name;
                    tabName.textContent = modified ? `${fileName} *` : fileName;
                }
            }
        }
    }

    // Show new file modal
    function showNewFileModal() {
        elements.newFileDirectory.value = state.currentPath;
        elements.newFileName.value = '';
        
        $('#new-file-modal').modal('show');
        setTimeout(() => elements.newFileName.focus(), 500);
    }

    // Show new directory modal
    function showNewDirectoryModal() {
        elements.newDirectoryParent.value = state.currentPath;
        elements.newDirectoryName.value = '';
        
        $('#new-directory-modal').modal('show');
        setTimeout(() => elements.newDirectoryName.focus(), 500);
    }

    // Create new file
    function createNewFile() {
        const directory = elements.newFileDirectory.value;
        const filename = elements.newFileName.value.trim();
        
        if (!filename) {
            alert('Please enter a file name');
            return;
        }
        
        fetch('/api/create-file', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                directory: directory,
                filename: filename
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Close modal
                $('#new-file-modal').modal('hide');
                
                // Refresh explorer
                refreshExplorer();
                
                // Open the new file
                setTimeout(() => openFile(data.file_path), 500);
            } else {
                alert(data.message);
            }
        })
        .catch(error => {
            alert(`Error creating file: ${error.message}`);
        });
    }

    // Create new directory
    function createNewDirectory() {
        const parentDir = elements.newDirectoryParent.value;
        const dirname = elements.newDirectoryName.value.trim();
        
        if (!dirname) {
            alert('Please enter a directory name');
            return;
        }
        
        fetch('/api/create-directory', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                parent_directory: parentDir,
                dirname: dirname
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Close modal
                $('#new-directory-modal').modal('hide');
                
                // Refresh explorer
                refreshExplorer();
                
                // Navigate to the new directory
                setTimeout(() => navigateToPath(data.directory_path), 500);
            } else {
                alert(data.message);
            }
        })
        .catch(error => {
            alert(`Error creating directory: ${error.message}`);
        });
    }

    // Helper: Format file size
    function formatFileSize(bytes) {
        if (bytes < 1024) {
            return bytes + ' B';
        } else if (bytes < 1024 * 1024) {
            return (bytes / 1024).toFixed(1) + ' KB';
        } else if (bytes < 1024 * 1024 * 1024) {
            return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
        } else {
            return (bytes / (1024 * 1024 * 1024)).toFixed(1) + ' GB';
        }
    }

    // Helper: Show message
    function showMessage(message, type) {
        // Create message element
        const messageElement = document.createElement('div');
        messageElement.className = `alert alert-${type === 'error' ? 'danger' : 'success'} file-explorer-message`;
        messageElement.textContent = message;
        
        // Append to body
        document.body.appendChild(messageElement);
        
        // Remove after 3 seconds
        setTimeout(() => {
            messageElement.remove();
        }, 3000);
    }

    // Public API
    return {
        init: init
    };
})(); 