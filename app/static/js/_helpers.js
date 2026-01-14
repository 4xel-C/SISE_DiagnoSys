function normalizeCombo(event) {
    const keys = [];
    if (event.ctrlKey) keys.push('Ctrl');
    if (event.altKey) keys.push('Alt');
    if (event.shiftKey) keys.push('Shift');
    keys.push(event.code); // 'Space', 'KeyS', etc.
    return keys.join('+');
}