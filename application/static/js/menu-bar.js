const section = document.querySelector('#menu-bar');
const internalList = section.querySelector('ul.internal');
const patientList = section.querySelector('ul.patients');





// On internal clicked
internalList.querySelectorAll('li').forEach(internal => {
    internal.addEventListener('click', () => {
        // Unselect previous and mark new li as selected
        section.querySelector('li.selected').classList.remove('selected');
        internal.classList.add('selected');
    })
});

// On patient clicked
patientList.querySelectorAll('li').forEach(patient => {
    patient.addEventListener('click', () => {
        // Unselect previous and mark new li as selected
        section.querySelector('li.selected').classList.remove('selected');
        patient.classList.add('selected');
    })
});