import { isRecording } from './streamer.js';

const main = document.querySelector('main');
const content = main.querySelector('.content');





async function createUnsavedPopup(contextForm, continueFunction) {
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
    // On save
    popupForm.addEventListener('submit', (e) => {
        e.preventDefault();
        contextForm.dispatchEvent(new Event('submit', {
            bubbles: true,
            cancelable: true
        }))
        popup.remove();
        continueFunction();
    })
    // On delete
    popupForm.addEventListener('reset', () => {
        popup.remove();
        continueFunction();
    })
}



export async function renderPage(pageName, force=false) {
    const pageContainer = content.querySelector('.page-container');

    // Check for unsaved data
    if (main.classList.contains('unsaved') && !force) {
        const contextForm = main.querySelector('form#context-editor');
        if (contextForm) {
            createUnsavedPopup(contextForm, () => renderPage(pageName, true));
            return;
        }
    }
    // Reset saved class
    main.classList.remove('unsaved');

    // Request page HTML
    const response = await fetch(`ajax/render_page/${pageName}`);
    const html = await response.text();

    // Render html
    content.classList.remove('patient-content');
    content.classList.add('internal-content');
    pageContainer.innerHTML = html;

    // Dispatch event
    document.dispatchEvent(
        new CustomEvent('internalRendered', {
            detail: { pageName }
        })
    );
}

export async function renderPatient(patientId, force=false) {
    const pageContainer = content.querySelector('.page-container');
    const audioRecord = main.querySelector('.audio-record');

    // Check for unsaved data
    if (main.classList.contains('unsaved') && !force) {
        const contextForm = main.querySelector('form#context-editor');
        if (contextForm) {
            createUnsavedPopup(contextForm, () => renderPatient(patientId, true));
            return;
        }
    }
    // Reset saved class
    main.classList.remove('unsaved');
    // Patient class
    content.classList.remove('internal-content');
    content.classList.add('patient-content');

    // Request patient HTML
    const response = await fetch(`ajax/render_patient/${patientId}`);
    const html = await response.text();

    // Display patient
    pageContainer.innerHTML = html;
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