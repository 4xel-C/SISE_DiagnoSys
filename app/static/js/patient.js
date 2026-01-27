import { fromDelta, toDelta } from 'https://cdn.jsdelivr.net/npm/@slite/quill-delta-markdown@0.0.8/+esm';
import { renderPatient, openChat } from './modules/loader.js';


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
    await fetch(`ajax/process_rag/${patientId}`, {
        method: 'POST'
    })
}


function switchTab(frame, li) {
    // Unselect previous
    const previousLi = frame.querySelector('li.selected');
    previousLi.classList.remove('selected');
    const previousTab = frame.querySelector(`.tab[data-tab-id="${previousLi.dataset.tabId}"]`);
    previousTab.classList.remove('active');
    // Select current
    li.classList.add('selected');
    const tab = frame.querySelector(`.tab[data-tab-id="${li.dataset.tabId}"]`);
    tab.classList.add('active');
}



async function renderContext(patientId) {
    frames.context.classList.add('waiting');
    // Request diagnostic
    const response = await fetch(`ajax/get_context/${patientId}`);
    const content = await response.json();
    // Convert markdown to quill delta
    const mdContent = toDelta(content.context);
    // Overwrite editor
    frames.context.classList.remove('waiting');
    contextEditor.setContents(mdContent);
}

async function renderDiagnostics(patientId) {
    frames.diagnostic.classList.add('waiting');
    // Request diagnostic
    const response = await fetch(`ajax/get_diagnostic/${patientId}`);
    const content = await response.json();
    // Convert markdown to quill delta
    const mdContent = toDelta(content.diagnostic);
    // Overwrite editor
    frames.diagnostic.classList.remove('waiting');
    diagnosticViewer.setContents(mdContent);
}

async function renderDocuments(patientId) {
    frames.documents.classList.add('waiting');
    // Request diagnostic
    const response = await fetch(`ajax/get_related_documents/${patientId}`);
    const content = await response.json();
    // Update close documents
    frames.documents.classList.remove('waiting');
    const documentsList = frames.documents.querySelector('ul');
    documentsList.innerHTML = '';
    content.documents.forEach(html => {
        documentsList.insertAdjacentHTML('beforeend', html);
        const document = documentsList.lastElementChild;
        // Bind click -> open diagnostics
        document.addEventListener('click', () => {
            window.open(document.dataset.url, '_blank').focus();
        })
    });    
}

async function renderCases(patientId) {
    frames.cases.classList.add('waiting');
    // Request diagnostic
    const response = await fetch(`ajax/get_related_cases/${patientId}`);
    const content = await response.json();
    // Update similar cases
    frames.cases.classList.remove('waiting');
    const casesList = frames.cases.querySelector('ul');
    casesList.innerHTML = '';
    content.cases.forEach(html => {
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
    const chatButton = patientContainer.querySelector('button#start-chat');
    const contextForm = patientContainer.querySelector('form#context-editor');
    frames = {
        context: patientContainer.querySelector('.frame.context-profile'),
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

    // On tab switch
    Object.values(frames).forEach(frame => {
        const nav = frame.querySelector('ul.nav');
        if (nav) {
            nav.querySelectorAll('li').forEach(li => {
                li.addEventListener('click', () => {
                    switchTab(frame, li);
                })
            })
        }
    });

    // On context saved
    contextForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        if (contextEditor.getText().trim().length === 0) {
            return
        }
        const context = fromDelta(contextEditor.getContents().ops);
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
        frames.documents.classList.add('waiting');
        processRAG(patientId).then(() => {
            // Update results
            renderDiagnostics(patientId);
            renderDocuments(patientId);
            renderCases(patientId);
        })
    });

    // On start-chat button clicked
    chatButton.addEventListener('click', () => {
        openChat(patientId);
    })

    // Load and render patient content
    renderContext(patientId);
    renderDiagnostics(patientId);
    renderCases(patientId);
    renderDocuments(patientId);
});


// On recording stopped (and chatbot responded)
['audioRecordStoped', 'assistantResponded'].forEach(eventName => {
    document.addEventListener(eventName, () => {
        Object.values(frames).forEach(frame => {
            frame.classList.add('waiting');
        });
    });
});

// On audio process compleded (and chatbot simulation)
['audioProcessCompleted', 'assistantConversationProcessed'].forEach(eventName => {
    document.addEventListener(eventName, (e) => {
        const patientId = e.detail.patientId;
        // Update context
        renderContext(patientId);
        // Process RAG
        processRAG(patientId).then(() => {
            // Then update results
            renderDiagnostics(patientId);
            renderDocuments(patientId);
            renderCases(patientId);
        })
    })
});