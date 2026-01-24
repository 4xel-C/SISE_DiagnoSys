import { socket } from './streamer.js';

const main = document.querySelector('main');
const content = main.querySelector('.content');





async function createUnsavedPopup(patientId, contextForm) {
    // Request popup HTML
    const params = new URLSearchParams({
        title: "Modifications non sauvegardées",
        description: "Voulez-vous sauvegarder les modification apportez a la fiche patient ?",
        second: "Supprimer",
        main: "Enregistrer"
    })
    const response = await fetch(`ajax/custom_popup?${params}`);
    const html = await response.text();
    // Render popup
    const popup = renderPopup(html);
    const popupForm = popup.querySelector('form');
    popupForm.addEventListener('submit', (e) => {
        e.preventDefault();
        contextForm.dispatchEvent(new Event('submit', {
            bubbles: true,
            cancelable: true
        }))
        popup.remove();
        renderPatient(patientId, true);
    })
    popupForm.addEventListener('reset', () => {
        popup.remove();
        renderPatient(patientId, true);
    })
}




export async function renderPatient(patientId, force=false) {
    const audioRecord = main.querySelector('.audio-record');

    // Check for unsaved data
    if (main.classList.contains('unsaved') && !force) {
        const contextForm = main.querySelector('form#context-editor');
        if (contextForm) {
            createUnsavedPopup(patientId, contextForm);
            return;
        }
    }
    // Reset saved class
    main.classList.remove('unsaved');

    // Request diagnostic HTML
    const response = await fetch(`ajax/render_patient/${patientId}`);
    const html = await response.text();

    // Display diagnostics
    content.innerHTML = html;
    // Audio record activation logic
    audioRecord.classList.remove('active');
    if (socket && socket.readyState === WebSocket.OPEN) {
        if (main.dataset.recordPatientId === patientId) {
            audioRecord.classList.add('active');
        }
    } else {
        audioRecord.classList.add('active');
    }

    // Dispatch event
    document.dispatchEvent(
        new CustomEvent('patientRendered', {
            detail: { patientId }
        })
    );
}

export async function loadContext(patientId) {
    // request patient context
    // return md formatted string
    const response = await fetch(`ajax/get_context/${patientId}`);
    const content = await response.json();
    return content['context'];
}

export async function loadResults(patientId) {
    // request patient diagnostics, documents and similar cases
    // return dict
    const response = await fetch(`ajax/get_results/${patientId}`);
    const content = await response.json();
    return content
}