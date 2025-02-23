document.addEventListener('DOMContentLoaded', function() {
    // Function to handle 'Select All' button click
    document.getElementById('selectAll').addEventListener('click', function() {
        document.querySelectorAll('.table-checkbox').forEach(checkbox => {
            checkbox.checked = true;
        });
    });
    // Function to handle 'Deselect All' button click
    document.getElementById('deselectAll').addEventListener('click', function() {
        document.querySelectorAll('.table-checkbox').forEach(checkbox => {
            checkbox.checked = false;
        });
    });
});

// Function to handle search and filter on the table
function tableSelected() {
    let searchValues = [];
    for (let i = 1; i <= 9; i++) {
        searchValues.push(document.getElementById(`search-section-${i}`).value.toLowerCase());
    }

    let tableListItems = document.querySelectorAll('#table-list li');
    tableListItems.forEach(item => {
        let tableName = item.getAttribute('data-table-name');
        let tableSections = tableName.toLowerCase().split('_');
        
        let match = searchValues.every((searchValue, index) => {
            return searchValue === "" || (tableSections[index] && tableSections[index].startsWith(searchValue));
        });

        if (match) {
            item.style.display = 'block';
        } else {
            item.style.display = 'none';
        }
    });
}

function deleteRecord(database, tableName) {
    if(confirm('Are you sure you want to delete this record? This action cannot be undone.')) {
        fetch(`/delete-record/${database}/${tableName}`, {
            method: 'DELETE'
        })
        .then(response => response.text())
        .then(data => {
            alert(data);
            location.reload();
        })
        .catch(error => console.error('Error:', error));
    }
}

function deleteSelectedRecords(database) {
    let selectedTables = [];
    document.querySelectorAll('.table-checkbox:checked').forEach(checkbox => {
        let tableName = checkbox.parentElement.getAttribute('data-table-name');
        selectedTables.push(tableName);
    });

    if (selectedTables.length === 0) {
        alert("No tables selected.");
        return;
    }

    if(confirm('Are you sure you want to delete the selected records? This action cannot be undone.')) {
        fetch(`/delete-records/${database}`, {
            method: 'DELETE',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ tables: selectedTables })
        })
        .then(response => response.text())
        .then(data => {
            alert(data);
            // Uncheck and remove the deleted table items from the list
            selectedTables.forEach(tableName => {
                let checkbox = document.querySelector(`input[data-table-name="${tableName}"]`);
                if (checkbox) {
                    checkbox.checked = false; // Uncheck the checkbox
                    let listItem = checkbox.parentElement;
                    if (listItem) {
                        listItem.remove(); // Remove the list item
                    }
                }
            });
        })
        .catch(error => console.error('Error:', error));
    }
}

function renameTable(database, oldTableName, newTableName) {
    newTableName = newTableName.trim();
    oldTableName = oldTableName.trim();

    if (newTableName !== oldTableName) {
        var url = `/rename-table/${encodeURIComponent(database)}/${encodeURIComponent(oldTableName)}/${encodeURIComponent(newTableName)}`;

        fetch(url, {
            method: 'PUT',
        })
        .then(response => {
            if(response.ok) {
                console.log("Table renamed successfully.");
                // Update the UI dynamically to reflect the new table name
                updateUIAfterRename(oldTableName, newTableName);
            } else {
                response.text().then(text => console.error("Failed to rename the table:", text));
            }
        })
        .catch(error => {
            console.error('There was a problem with the fetch operation:', error);
        });
    } else {
        console.log("The table name is the same. No action taken.");
    }
}

function updateUIAfterRename(oldName, newName) {
    // Implement the logic to find and update the table name in your UI
    // This is a placeholder function and needs to be tailored to your specific UI structure
    let tables = document.querySelectorAll('[data-table-name]');
    tables.forEach(table => {
        if (table.getAttribute('data-table-name') === oldName) {
            table.setAttribute('data-table-name', newName); // Update the attribute
            let span = table.querySelector('span'); // Assuming this is where the name is displayed
            if (span) {
                span.innerText = newName; // Update the display text
            }
        }
    });
}

//<button class="btn btn-primary mt-2" onclick="downloadAllPlots('{{ plot }}')">Download Plot</button>
function downloadAllPlots() {
    let images = document.querySelectorAll("img");
    images.forEach((img, index) => {
        let a = document.createElement("a");
        a.href = img.src;
        a.download = "plot_" + (index + 1) + ".png"; // This will save as plot_1.png, plot_2.png, etc.
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    });
}

function downloadPlot(base64Data, index) {
    // Create an anchor element (`a`) for downloading
    let a = document.createElement("a");
    // Set the href to the base64 data URL
    a.href = 'data:image/png;base64,' + base64Data;
    // Set the download attribute to name the file
    a.download = 'plot_' + index + '.png';
    // Append the anchor to the document
    document.body.appendChild(a);
    // Trigger a click event on the anchor
    a.click();
    // Remove the anchor from the document
    document.body.removeChild(a);
}

/*function updateImage() {
    var selectedDatabase = document.getElementById("databaseSelect").value;
    var imageUrl = document.getElementById(selectedDatabase).value;
    document.getElementById("databaseImage").src = imageUrl;
}*/

// Define the updateImage function at the global scope
function updateImage() {
    var selectedDatabase = document.getElementById("databaseSelect").value;
    var imageUrl = document.getElementById(selectedDatabase).value;

    console.log("Selected Database: ", selectedDatabase);
    console.log("Image URL: ", imageUrl);

    document.getElementById("databaseImage").src = imageUrl;
}

// Use the pageshow event to call updateImage when the page is loaded or shown
window.addEventListener('pageshow', function (event) {
    updateImage();
});

function saveContent(database, tableName) {
    var content = document.getElementById('txt-content').innerText;
    var url = `/save-txt-content/${database}/${tableName}`;  // Store the URL in a variable
    console.log(url);  // Log the URL to the console
    fetch(url, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ content: content })
    })
    .then(response => response.text())
    .then(data => {
        alert(data);
    })
    .catch(error => console.error('Error:', error));
}

function filterTablesByName() {
    var input, filter, ul, li, i, txtValue;
    input = document.getElementById('search-table-name');
    filter = input.value.toUpperCase().trim();
    ul = document.getElementById("table-list");
    li = ul.getElementsByClassName('table-item');

    for (i = 0; i < li.length; i++) {
        txtValue = li[i].getAttribute('data-table-name') || "";
        // Change this line to check if the txtValue starts with the filter
        if (txtValue.toUpperCase().startsWith(filter)) {
            li[i].style.display = "";
        } else {
            li[i].style.display = "none";
        }
    }
}

