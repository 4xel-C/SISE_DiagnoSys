import { loadDiagnostics } from './modules/loader.js';

const menu = document.querySelector('#menu-bar');
const main = document.querySelector('main');
const internalList = menu.querySelector('ul.internal');
const patientList = menu.querySelector('ul.patients');
const topbar = main.querySelector('.top-bar');




async function searchPatients(query='') {
    // Build URL
    const url = new URL('ajax/search_patients', window.location.origin);
    if (query) {
        url.searchParams.set('query', query);
    }
    // Request
    const response = await fetch(url);

    // Remove previous patient results
    patientList.innerHTML = ''; 
    // Handle errors
    if (!response.ok) {
        console.error('Failed to search patient, python bad response');
        return
    }

    // Load and render results
    const patients = await response.json();
    patients.forEach(html => {
        patientList.insertAdjacentHTML('beforeend', html)
    });
}

function selectElement(element) {
    // Unselect previous
    const selectedElement = menu.querySelector('li.selected');
    if (selectedElement) {
        selectedElement.classList.remove('selected');
    }
    // Select new
    element.classList.add('selected');
}





// On internal clicked
internalList.querySelectorAll('li').forEach(internal => {
    internal.addEventListener('click', () => {
        selectElement(internal);
    })
});

// On patient clicked
patientList.querySelectorAll('li').forEach(patient => {
    patient.addEventListener('click', () => {
        selectElement(patient);
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

// On top bar button clicked
topbar.querySelector('button.see').addEventListener('click', () => {
    const patientId = main.dataset.recordPatientId;
    const patient = patientList.querySelector(`li[data-patient-id="${patientId}"]`);
    selectElement(patient);
    loadDiagnostics(patientId);
});



// Init patients (get all)
searchPatients()