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
    return messageElement;
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
    // Scroll down to new message
    messageList.scrollTop = messageList.scrollHeight;
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
    return content['message'];
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
        const messageElement = addMessage(chatFrom, message['content'], message['role']);
        messageElement.classList.add('history');
    })
    // Display response if not empty
    if (content['message'] !== '') {
        addMessage(chatFrom, content['message'], 'assistant');
    }
}

async function processConversation(patientId, message, response) {
    // Request context update from conversation
    await fetch(`ajax/process_conversation/${patientId}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            message,
            response
        })
    })
    // Dispatch completion event when server responds
    document.dispatchEvent(
        new CustomEvent('assistantConversationProcessed', {
            detail: { patientId }
        })
    );
}

// Key shortcut handler
function focusInput(chatFrom) {
    chatFrom.querySelector('input[name="query"]').focus();
}





//On chatboat opened
document.addEventListener('chatbotOpened', (e) => {
    const chatFrom = main.querySelector('form#chatbot');
    const patientId = e.detail.patientId;

    const shortcuts = new Map([
        ['Ctrl+KeyM', () => focusInput(chatFrom)]
    ]);

    // Request greeting message
    loadAgent(chatFrom, patientId);

    // On message send
    chatFrom.addEventListener('submit', (e) => {
        e.preventDefault();
        const message = chatFrom.elements.query.value;
        chatFrom.elements.query.value = '';
        addMessage(chatFrom, message, 'user');
        queryAgent(chatFrom, message, patientId).then((response) => {
            // Dispatch completion event when server responds
            document.dispatchEvent(
                new CustomEvent('assistantResponded', {
                    detail: { patientId }
                })
            );
            processConversation(patientId, message, response);
        })
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

    // On Ctrl+m focus search bar
    window.addEventListener('keydown', (event) => {
        const combo = normalizeCombo(event);
        const handler = shortcuts.get(combo);
        if (handler) {
            event.preventDefault();
            handler(event);
        }
    });

    focusInput(chatFrom);
})