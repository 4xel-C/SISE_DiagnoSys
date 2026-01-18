const main = document.querySelector('main');






async function processRAG(data) {
    const response = await fetch('ajax/process_rag', {
        method: 'POST',
        body: data
    })
    return await response.json();
}

function updateContext(content) {
    // Update context text content
    console.log('Updating context');
}

function updateDiagnostics(content) {
    // Update diagnostics list
    console.log('Updating diagnostics');
}

function updateDocuments(content) {
    // Update close documents
    console.log('Updating documents');
}

function updateCases(content) {
    // Update similar cases
    console.log('Updating cases');
}





// On diagnostics loaded
document.addEventListener('diagnosticsLoaded', (e) => {
    const patientId = e.detail.patientId;
    const diagnostics = main.querySelector('.diagnostics');
    const contextForm = diagnostics.querySelector('form#context-editor');

    // On context edited
    contextForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        // Request context processing
        const data = new FormData(contextForm);
        data.append('patientId', patientId);
        const content = await processRAG(data);
        // Update UI
        if ('diagnostics' in content) {
            updateDiagnostics(content['diagnostics']);
        }
        if ('documents' in content) {
            updateDocuments(content['documents']);
        }
        if ('cases' in content) {
            updateCases(content['cases']);
        }
    });
});