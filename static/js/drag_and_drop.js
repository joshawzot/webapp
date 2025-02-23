document.addEventListener('DOMContentLoaded', function() {
    const dropArea = document.getElementById('drop_area');
    const fileInput = document.getElementById('files');
    const uploadForm = document.getElementById('uploadForm');
    let accumulatedFiles = []; // Track selected files for submission

    function setupEventListeners() {
        dropArea.addEventListener('click', function() {
            fileInput.click(); // Trigger file input dialog on drop area click
        });

        fileInput.addEventListener('change', function() {
            handleFiles(this.files); // Add files selected through input to the list
        });

        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropArea.addEventListener(eventName, preventDefaults, false); // Handle drag events
        });

        ['dragenter', 'dragover'].forEach(eventName => {
            dropArea.addEventListener(eventName, () => dropArea.classList.add('highlight'), false); // Highlight drop area
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropArea.addEventListener(eventName, () => dropArea.classList.remove('highlight'), false); // Unhighlight drop area
        });

        dropArea.addEventListener('drop', function(e) {
            handleFiles(e.dataTransfer.files); // Handle files dropped into the area
        });

        uploadForm.addEventListener('submit', handleSubmit); // Handle form submission
    }

    function handleFiles(files) {
        accumulatedFiles.push(...files); // Accumulate files from input and drop
        updateFileList();
    }

    function updateFileList() {
        const fileListElement = document.getElementById('file_list');
        fileListElement.innerHTML = accumulatedFiles.map((file, index) => `<li>${index + 1}: ${file.name}</li>`).join('') || '<p>No files selected</p>';
    }

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    async function handleSubmit(e) {
        e.preventDefault(); // Prevent the default form submission behavior
        console.log('Handling form submission...');

        const formData = new FormData(uploadForm);
        accumulatedFiles.forEach(file => {
            formData.append('files[]', file); // Append each file to FormData
        });

        try {
            console.log('Attempting to send request to:', uploadForm.action);
            const response = await fetch(uploadForm.action, {
                method: 'POST',
                body: formData,
            });

            console.log('Received response:', response);

            if (!response.ok) {
                throw new Error('Network response was not ok. Status: ' + response.status);
            }

            const contentType = response.headers.get("content-type");
            if (contentType && contentType.includes("application/json")) {
                const data = await response.json();
                console.log('Success:', data);
                alert('Files uploaded successfully.');
            } else {
                const text = await response.text();  // Assuming the response might be text/plain or HTML
                console.log('Non-JSON response:', text);
                alert('Response received, but it was not in JSON format: ' + text);
            }

            // Reset the accumulated files and update the UI accordingly
            accumulatedFiles = [];
            updateFileList();

            // Refresh the page to update the list of tables
            window.location.reload();
        } catch (error) {
            console.error('Error:', error);
            alert('Failed to upload files. Error: ' + error.message);
        }
    }

    setupEventListeners();
});