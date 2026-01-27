const main = document.querySelector('main');






//On chatboat opened
document.addEventListener('chatbotOpened', (e) => {
    const chatFrom = main.querySelector('form#chatbot');
    const patientId = e.detail.patientId;

    // On chatbot close
    chatFrom.addEventListener('reset', () => {
        chatFrom.remove();
        main.classList.remove('simulate');
    })

    chatFrom.querySelector('input[name="query"]').focus();
})