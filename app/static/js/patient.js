import { loadContext, loadResults, renderPatient } from './modules/loader.js';
// import { parseToBlocks, parseToMarkdown } from './modules/editorjs-markdown-parser/bundle.js';

const main = document.querySelector('main');
let diagnosticViewer;
let contextEditor;




async function saveContext(patientId, context) {
    const response = await fetch(`ajax/update_context/${patientId}`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({context: context})
    })
    return response
}

async function processRAG(patientId) {
    const response = await fetch(`ajax/process_rag/${patientId}`, {
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

async function renderDiagnostics(diagnostics) {
     // Convert markdown to Editor.js blocks
    const blocks = await MDtoBlocks(diagnostics);
    diagnosticViewer.isReady.then(() => {
        // Clear editor if not empty
        if (diagnosticViewer.blocks.getBlocksCount() > 1) {
            diagnosticViewer.clear()
        }
        // Render with new blocks
        diagnosticViewer.render({blocks: blocks});
    });
}

function renderDocuments(documents) {
    // Update close documents
    const documentsList = main.querySelector('.frame.documents ul');
    documentsList.innerHTML = '';
    documents.forEach(html => {
        documentsList.insertAdjacentHTML('beforeend', html);
        const document = documentsList.lastElementChild;
        // Bind click -> open diagnostics
        document.addEventListener('click', () => {
            window.open(document.dataset.url, '_blank').focus();
        })
    });
}

function renderCases(cases) {
    // Update similar cases
    const casesList = main.querySelector('.frame.cases ul');
    casesList.innerHTML = '';
    cases.forEach(html => {
        casesList.insertAdjacentHTML('beforeend', html);
        const patient = casesList.lastElementChild;
        // Bind click -> open diagnostics
        patient.addEventListener('click', () => {
            renderPatient(patient.dataset.patientId);
        })
    });
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
            header: Header,
            list: { 
                class: NestedList,
                inlineToolbar: true,
                config: {
                    defaultStyle: 'unordered'
                },
            },
        },
        onChange: async (api, event) => {
            if (event.type == 'block-changed') {
                const output = await contextEditor.save();
                console.log(output.blocks);
                contextForm.classList.add('edited');
                main.classList.add('unsaved');
            }
        }
    });

    // Create diagnostic viewer
    diagnosticViewer = new EditorJS({
        holder: 'diagnostic',
        readOnly: true
    });

    // On context saved
    contextForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        console.log('submitted');
        const output = await contextEditor.save();
        if (output.blocks.length == 0) {
            return
        }
        // Export editor blocks to md
        const context = await MDfromBlocks(output.blocks);
        // Request context update
        contextForm.querySelector('fieldset').disabled = true;
        const response = await saveContext(patientId, context);
        contextForm.querySelector('fieldset').disabled = false;
        if (!response.ok) {
            console.error('Failed to update context in database');
            return
        }
        contextForm.classList.remove('edited');
        main.classList.remove('unsaved');
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