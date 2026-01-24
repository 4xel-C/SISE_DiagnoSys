import { fromDelta, toDelta } from 'https://cdn.jsdelivr.net/npm/@slite/quill-delta-markdown@0.0.8/+esm'
import { loadContext, loadResults, renderPatient } from './modules/loader.js';


const main = document.querySelector('main');
let frames = {
    context: null,
    diagnostic: null,
    documents: null,
    cases: null
};

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
    const frame = main.querySelector('.frame.context');
    // Convert markdown to quill delta
    const content = toDelta(context);
    // Overwrite editor
    contextEditor.setContents(content);
    // Update UI
    frame.classList.remove('waiting')
}

async function renderDiagnostics(diagnostics) {
    const frame = main.querySelector('.frame.diagnostic');
    // Convert markdown to quill delta
    const content = toDelta(diagnostics);
    // Overwrite editor
    diagnosticViewer.setContents(content);
    // Update UI
    frame.classList.remove('waiting');
}

function renderDocuments(documents) {
    // Update close documents
    const frame = main.querySelector('.frame.documents');
    const documentsList = frame.querySelector('ul');
    documentsList.innerHTML = '';
    documents.forEach(html => {
        documentsList.insertAdjacentHTML('beforeend', html);
        const document = documentsList.lastElementChild;
        // Bind click -> open diagnostics
        document.addEventListener('click', () => {
            window.open(document.dataset.url, '_blank').focus();
        })
    });
    // Update UI
    frame.classList.remove('waiting');
}

function renderCases(cases) {
    // Update similar cases
    const frame = main.querySelector('.frame.cases');
    const casesList = frame.querySelector('ul');
    casesList.innerHTML = '';
    cases.forEach(html => {
        casesList.insertAdjacentHTML('beforeend', html);
        const patient = casesList.lastElementChild;
        // Bind click -> open diagnostics
        patient.addEventListener('click', () => {
            renderPatient(patient.dataset.patientId);
        })
    });
    // Update UI
    frame.classList.remove('waiting');
}







// On diagnostics loaded
document.addEventListener('patientRendered', (e) => {
    const patientId = e.detail.patientId;
    const patientContainer = main.querySelector('.patient');
    const contextForm = patientContainer.querySelector('form#context-editor');
    frames = {
        context: patientContainer.querySelector('.frame.context'),
        diagnostic: patientContainer.querySelector('.frame.diagnostic'),
        documents: patientContainer.querySelector('.frame.documents'),
        cases: patientContainer.querySelector('.frame.cases')
    }

    // Create context editor
    contextEditor = new Quill('#context', {
        placeholder: 'Ajoutez ou modifiez des informations',
        theme: 'bubble'
    });

    contextEditor.on('text-change', (delta, oldDelta, source) => {
        if (source == 'user') {
            contextForm.classList.add('edited');
            main.classList.add('unsaved');
        }
    });

    // Create diagnostic viewer
    diagnosticViewer = new Quill('#diagnostic', {
        theme: 'bubble',
        readOnly: true
    });

    // On context saved
    contextForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        if (contextEditor.getText().trim().length === 0) {
            return
        }
        const context = fromDelta(contextEditor.getContents());
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
        frames.diagnostic.classList.add('waiting');
        frames.cases.classList.add('waiting');
        frames.documents.classList.add('waiting');
        const content = await processRAG(patientId, context);
        // Update results
        renderDiagnostics(content['diagnostics']);
        renderDocuments(content['documents']);
        renderCases(content['cases']);
    });

    Object.values(frames).forEach(frame => {
        frame.classList.add('waiting');
    })

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

