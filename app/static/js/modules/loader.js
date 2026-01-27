import { isRecording } from './streamer.js';

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
    const patientContainer = content.querySelector('.patient-container');
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
    patientContainer.innerHTML = html;
    // Audio record activation logic
    audioRecord.classList.remove('active');
    if (isRecording()) {
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

export async function openChat(patientId) {
    // Request chatbot HTML
    const response = await fetch('ajax/render_chat');
    const html = await response.text();
    // Render chatbot
    main.insertAdjacentHTML('beforeend', html);
    main.classList.add('simulate');
    // Get patient name
    const chatName = main.querySelector('form#chatbot h2')
    fetch(`ajax/get_profile/${patientId}`).then(async (response) => {
        const content = await response.json();
        chatName.textContent = content['prenom'] + ' ' + content['nom'];
    })
    // Dispatch event
    document.dispatchEvent(
        new CustomEvent('chatbotOpened', {
            detail: { 
                patientId            
            }
        })
    );
}