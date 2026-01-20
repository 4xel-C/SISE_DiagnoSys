import { loadContext, loadResults } from './modules/loader.js';

const main = document.querySelector('main');
let contextEditor;




async function saveContext(patientId, context) {
    console.log(context);
    const response = await fetch(`ajax/update_context/${patientId}`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({context: context})
    })
    return response
}

async function processRAG(patientId) {
    const data = new FormData();
    data.append('patientId', patientId);
    const response = await fetch('ajax/process_rag', {
        method: 'POST'
    })
    return await response.json();
}

async function renderContext(context) {
    // Convert markdown to Editor.js blocks
    const blocks = await MDtoBlocks(context);
    contextEditor.isReady.then(() => {
        // Clear editor if not empty
        if (contextEditor.blocks.getBlocksCount() > 1) {
            contextEditor.clear()
        }
        // Render with new blocks
        contextEditor.render({blocks: blocks});
    });
}

function renderDiagnostics(diagnostics) {
    // Update diagnostics list
    console.log('Updating diagnostics');
}

function renderDocuments(documents) {
    // Update close documents
    console.log('Updating documents');
}

function renderCases(cases) {
    // Update similar cases
    console.log('Updating cases');
}







// On diagnostics loaded
document.addEventListener('patientRendered', (e) => {
    const patientId = e.detail.patientId;
    const patientContainer = main.querySelector('.patient');
    const contextForm = patientContainer.querySelector('form#context-editor');

    // Create context editor
    contextEditor = new EditorJS({
        holder: 'context',
        placeholder: 'Ajoutez ou modifiez des informations',
        tools: {
            header: Header
        }
    });

    // On context saved
    contextForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        console.log('submitted');
        // Save editor and send content
        if (contextEditor.blocks.getBlocksCount() <= 1) {
            console.log(contextEditor.blocks);
            return
        }
        const output = await contextEditor.save();
        const context = await MDfromBlocks(output.blocks);
        // Request context update
        const response = await saveContext(patientId, context);
        if (!response.ok) {
            console.error('Failed to update context in database');
            return
        }
        // Request context processing
        const content = await processRAG(patientId, context);
        // Update results
        renderDiagnostics(content['diagnostics']);
        renderDocuments(content['documents']);
        renderCases(content['cases']);
    });

    // Load and render context
    loadContext(patientId).then((context) => {
        renderContext(context);
    })
    // Load and render results
    loadResults(patientId).then((content) => {
        renderDiagnostics(content['diagnostics']);
        renderDocuments(content['documents']);
        renderCases(content['cases']);
    });
});