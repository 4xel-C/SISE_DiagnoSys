import { socket } from './streamer.js';


export async function loadDiagnostics(patientId) {
    const main = document.querySelector('main');
    const content = main.querySelector('.content');
    const audioRecord = main.querySelector('.audio-record');
    // Request diagnostic HTML
    const response = await fetch(`ajax/render_diagnostics/${patientId}`);
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
}