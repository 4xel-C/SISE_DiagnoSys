import { renderPage, renderPatient } from './modules/loader.js';

const menu = document.querySelector('#menu-bar');
const main = document.querySelector('main');
const searchForm = menu.querySelector('form.search');
const internalList = menu.querySelector('ul.internal');
const patientList = menu.querySelector('ul.patients');
const addPatientButton = menu.querySelector('.create-patient button');
const settingsButton = menu.querySelector('.settings button');
const topbar = main.querySelector('.top-bar');

const shortcuts = new Map([
    ['Ctrl+Slash', focusSearch],
    ['Escape', unFocus]
]);




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
        patientList.insertAdjacentHTML('beforeend', html);
        const patient = patientList.lastElementChild;
        // Bind click -> open diagnostics
        patient.addEventListener('click', () => {
            renderPatient(patient.dataset.patientId);
        })
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

async function createPatient() {
    // Render popup
    const response = await fetch('ajax/create_patient_popup');
    const html = await response.text();
    const popup = renderPopup(html);

    const popupForm = popup.querySelector('form');
    const popupFieldset = popupForm.querySelector('fieldset');
    // On cancel
    popupForm.addEventListener('reset', () => {
        popup.remove();
    });
    // On submit
    popupForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const data = new FormData(popupForm);
        popupFieldset.disabled = true;
        fetch('ajax/create_patient', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(Object.fromEntries(data))
        }).then(async (response) => {
            const content = await response.json();

            if (!response.ok) {
                popupFieldset.disabled = false;
                showError(content.error);
                return;
            }

            popup.remove();
            await searchPatients();
            renderPatient(content.patient_id);
        })
    });
}

async function openSettings() {
    // Render popup
    const response = await fetch('ajax/settings_popup');
    const html = await response.text();
    const popup = renderPopup(html);

    const popupForm = popup.querySelector('form');
    // On cancel
    popupForm.addEventListener('reset', () => {
        popup.remove();
    });
    // On submit
    popupForm.addEventListener('submit', () => {
        e.preventDefault();
        const data = new FormData(popupForm);
        fetch('ajax/update_settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(Object.fromEntries(data))
        }).then(async (response) => {
            const content = await response.json();

            if (!response.ok) {
                popupFieldset.disabled = false;
                showError(content.error);
                return;
            }

            popup.remove();
            await searchPatients();
            renderPatient(content.patient_id);
        })
    });
}


// Key shortcut handler
function focusSearch() {
    searchForm.elements.query.focus();
}

function unFocus() {
    document.activeElement?.blur();

}



// On patient searched
searchForm.addEventListener('submit', (e) => {
    e.preventDefault();
    searchPatients(searchForm.elements.query.value);
});

// On internal clicked => select it
internalList.querySelectorAll('li').forEach(internal => {
    internal.addEventListener('click', () => {
        selectElement(internal);
        renderPage(internal.dataset.pageName);
    })
});

// On patient rendered => select it
document.addEventListener('patientRendered', (e) => {
    const element = menu.querySelector(`li.patient-result[data-patient-id="${e.detail.patientId}"]`);
    selectElement(element)
});

// On audio recording started
document.addEventListener('audioRecordStarted', (e) => {
    const patientId = e.detail.patientId;
    const recordedPatient = patientList.querySelector(`li[data-patient-id="${patientId}"]`);
    const profileContainer = recordedPatient.querySelector('.pp');
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
    const imgContainer = recordedPatient.querySelector('.pp .recording');
    imgContainer.remove();
});

// On top bar button clicked
topbar.addEventListener('click', () => {
    const patientId = main.dataset.recordPatientId;
    const patient = patientList.querySelector(`li[data-patient-id="${patientId}"]`);
    selectElement(patient);
    renderPatient(patientId);
});


// On Ctrl+/ focus search bar
window.addEventListener('keydown', (event) => {
    const combo = normalizeCombo(event);
    const handler = shortcuts.get(combo);
    if (handler) {
        event.preventDefault();
        handler(event);
    }
});

// On create patient 
addPatientButton.addEventListener('click', () => {
    createPatient();
});

// On settings open
settingsButton.addEventListener('click', () => {
    openSettings();
});



// Init patients (get all)
searchPatients()