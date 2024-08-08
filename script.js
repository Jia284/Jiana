document.getElementById("spam-form").addEventListener('submit', function(event){
    event.preventDefault();
    const inputtext = document.getElementById('inputtext').value;
    const method = document.getElementById('method').value; // Get the selected method
    const url='/detect-spam';
    
    // Show loading gif
    document.getElementById('loading').style.display = 'block';
    
    fetch(url, {
    method: 'POST',
    headers: {
    'Content-Type': 'application/json'
    },
    body: JSON.stringify({"text": inputtext,"method":method})
    })
    .then(response => response.json())
    .then(data => {
    document.getElementById('result').textContent = data.message;
    // Add justification to the info icon
    document.getElementById('info-icon').title = data.justification;
    // Hide loading gif
    document.getElementById('loading').style.display = 'none';
    })
    .catch(error => {
    console.error('Error:', error);
    document.getElementById('result').textContent = 'An Error Occurred!';
    // Hide loading gif
    document.getElementById('loading').style.display = 'none';
    });
    });