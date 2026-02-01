import { renderPatient, openChat } from './modules/loader.js';

const tuto = document.querySelector('.tutorial-overlay .tuto');
const title = tuto.querySelector('h2');
const description = tuto.querySelector('.description');
let step = 1;


function step1() {
    title.textContent = "Navigation";
    description.textContent = "Utilisez la barre latérale pour rechercher et naviguer parmi les différentes fiches patient";

    const navBar = document.querySelector('nav#menu-bar');
    navBar.classList.add('tutorial-step');
}

function step2() {
    title.textContent = "Fiche patient";
    description.textContent = "Une fiche patient comporte quatre éléments : le contexte, le diagnostic, les cas similaires et les documents liés au contexte";

    renderPatient(1).then(() => {
        const patient = document.querySelector('.page-container .patient');
        patient.classList.add('tutorial-step');
    })
}

function step3() {
    title.textContent = "Lancer une conversation";
    description.textContent = "Une fois sur une fiche patient, vous pouvez enregistrer une conversation en cliquant sur ce bloc. L'application utilisera ensuite cette conversation pour mettre à jour le contexte et proposer des diagnostics.";

    const audioRecord = document.querySelector('.content .audio-record');
    audioRecord.classList.add('tutorial-step');
}

function step4() {
    tuto.style.bottom = "50%";
    tuto.style.right = "35rem";
    tuto.style.left = "";
    title.textContent = "Simuler une conversation";
    description.textContent = "Il est possible de simuler une conversation avec le patient en utilisant le chatbot intégré à sa fiche.";

    openChat(1).then(() => {
        const chatForm = document.querySelector('form#chatbot');
        chatForm.classList.add('tutorial-step');
    })
}




function renderStep(index) {
    // Remove previous tutorial steps
    tuto.dataset.step = index;
    document.querySelectorAll('.tutorial-step').forEach(element => {
        element.classList.remove('tutorial-step');
    })
    // Render current step
    switch (index)  {
        case 1:
            step1();
            break
        case 2:
            step2();
            break
        case 3:
            step3();
            break
        case 4:
            step4();
            break
        case 5:
            window.location.reload();
    }
}

document.addEventListener('click', (e) => {
    e.preventDefault();
    step += 1;
    renderStep(step);
})

renderStep(step);