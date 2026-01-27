const main = document.querySelector('main');



function addMessage(chatFrom, message, user) {
    const messageList = chatFrom.querySelector('ul.messages');
    // Create element
    const messageElement = document.createElement('li');
    messageElement.classList.add(user);
    messageElement.textContent = message;
    // Render element
    messageList.appendChild(messageElement);
    // Scroll down to new message
    messageList.scrollTop = messageList.scrollHeight;
}

async function addTypingBubbles(chatFrom) {
    const messageList = chatFrom.querySelector('ul.messages');
    // Request bubbles HTML
    const response = await fetch('ajax/render_typing_bubbles');
    const html = await response.text();
    // Render HTML
    const bubblesElement = document.createElement('li');
    bubblesElement.classList.add('assistant');
    bubblesElement.innerHTML = html;
    messageList.appendChild(bubblesElement);
    return bubblesElement
}

async function queryAgent(chatFrom, query, patientId) {
    // Create typing bubbles
    const typingBubbles = await addTypingBubbles(chatFrom);
    // Request LLM agent
    const response = await fetch(`ajax/query_agent/${patientId}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({query})
    })
    const content = await response.json();
    // Update message
    typingBubbles.remove();
    addMessage(chatFrom, content['message'], 'assistant');
}

async function loadAgent(chatFrom, patientId) {
    // Create empty message
    const typingBubbles = await addTypingBubbles(chatFrom);
    // Request LLM agent
    const response = await fetch(`ajax/load_agent/${patientId}`, {
        method: 'POST'
    })
    const content = await response.json();
    // Update message
    typingBubbles.remove();
    // Render history
    content['history'].forEach(message => {
        addMessage(chatFrom, message['content'], message['role']);
    })
    // Display response if not empty
    if (content['message'] !== '') {
        addMessage(chatFrom, content['message'], 'assistant');
    }
}



//On chatboat opened
document.addEventListener('chatbotOpened', (e) => {
    const chatFrom = main.querySelector('form#chatbot');
    const patientId = e.detail.patientId;

    // Request greeting message
    loadAgent(chatFrom, patientId);

    // On message send
    chatFrom.addEventListener('submit', (e) => {
        e.preventDefault();
        const message = chatFrom.elements.query.value;
        chatFrom.elements.query.value = '';
        addMessage(chatFrom, message, 'user');
        queryAgent(chatFrom, message, patientId);
    });

    // On chatbot close
    chatFrom.addEventListener('reset', () => {
        chatFrom.remove();
        main.classList.remove('simulate');
    })
    // On switch patient
    document.addEventListener('patientRendered', () => {
        chatFrom.remove();
        main.classList.remove('simulate');
    })

    chatFrom.querySelector('input[name="query"]').focus();
})