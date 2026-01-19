import { socket } from './streamer.js';


export async function renderPatient(patientId) {
    const main = document.querySelector('main');
    const content = main.querySelector('.content');
    const audioRecord = main.querySelector('.audio-record');

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