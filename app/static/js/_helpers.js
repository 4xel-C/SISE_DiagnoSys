function normalizeCombo(event) {
    const keys = [];
    if (event.ctrlKey) keys.push('Ctrl');
    if (event.altKey) keys.push('Alt');
    if (event.shiftKey) keys.push('Shift');
    keys.push(event.code); // 'Space', 'KeyS', etc.
    return keys.join('+');
}

function renderPopup(html) {
    const popup = document.createElement('div');
    popup.classList.add('popup');
    popup.innerHTML = html;
    // Escape popup
    popup.addEventListener('click', (event) => {
        if (event.target === popup) {
            popup.remove();
        }
    });
    // Render popup
    document.body.appendChild(popup);
    return popup;
}

function showError(message) {
    const main = document.querySelector('main');
    main.classList.add('error');
    const errorMessage = main.querySelector('.error .label');
    errorMessage.textContent = message;

    setTimeout(() => {
        main.classList.remove('error');
    }, 5000); // 5 seconds
}