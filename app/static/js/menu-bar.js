import { loadDiagnostics } from './modules/loader.js';

const section = document.querySelector('#menu-bar');
const internalList = section.querySelector('ul.internal');
const patientList = section.querySelector('ul.patients');





async function loadDiagnostic(patientId) {
    // Load and render patient diagnostic
}





// On internal clicked
internalList.querySelectorAll('li').forEach(internal => {
    internal.addEventListener('click', () => {
        // Unselect previous and mark new li as selected
        const selectedTab = section.querySelector('li.selected');
        if (selectedTab) {
            selectedTab.classList.remove('selected');
        }
        internal.classList.add('selected');
    })
});

// On patient clicked
patientList.querySelectorAll('li').forEach(patient => {
    patient.addEventListener('click', () => {
        // Unselect previous and mark new li as selected
        const selectedTab = section.querySelector('li.selected');
        if (selectedTab) {
            selectedTab.classList.remove('selected');
        }
        patient.classList.add('selected');
        loadDiagnostics(patient.dataset.patientId);
    })
});

// On audio recording started
document.addEventListener('audioRecordStarted', (e) => {
    const patientId = e.detail.patientId;
    const recordedPatient = patientList.querySelector(`li[data-patient-id="${patientId}"]`);
    const profileContainer = recordedPatient.querySelector('.profile');
    const imgContainer = document.createElement('div');
    imgContainer.classList.add('recording');
    const img = document.createElement('img');
    img.src = '/static/images/mic.svg';
    imgContainer.appendChild(img);
    profileContainer.appendChild(imgContainer);
});

// On audio recording stoped
document.addEventListener('audioRecordStoped', (e) => {
    const patientId = e.detail.patientId;
    const recordedPatient = patientList.querySelector(`li[data-patient-id="${patientId}"]`);
    const imgContainer = recordedPatient.querySelector('.profile .recording');
    imgContainer.remove();
});