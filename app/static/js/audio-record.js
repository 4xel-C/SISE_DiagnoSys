const main = document.querySelector('main');
const startButton = main.querySelector('button#start');
const stopButton = main.querySelector('button#stop');
let mediaRecorder;
let socket;



startButton.addEventListener('click', async () => {
    socket = new WebSocket("ws://localhost:8000/audio_stt");
    await new Promise(resolve => socket.onopen = resolve);

    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm;codecs=opus' });

    mediaRecorder.ondataavailable = async (event) => {
        if (event.data.size > 0 && socket.readyState === WebSocket.OPEN) {
            const buffer = await event.data.arrayBuffer();
            socket.send(buffer);
        }
    };

    mediaRecorder.start(250); // send data every 250ms
    startButton.disabled = true;
    stopButton.disabled = false;
});

stopButton.addEventListener('click', () => {
    mediaRecorder.stop();
    socket.close();
    startButton.disabled = false;
    stopButton.disabled = true;
});