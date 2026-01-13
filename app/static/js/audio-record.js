const main = document.querySelector('main');
const audioContainer = main.querySelector('.audio-record');


let mediaRecorder;
let stream;
let socket;
let sendChain = Promise.resolve();

// Shortcuts
const shortcuts = new Map([
    ['Alt+Space', toggleMic]
]);




async function startAudioStream() {
    socket = new WebSocket("ws://localhost:8000/audio_stt");
    await new Promise(resolve => socket.onopen = resolve);

    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm;codecs=opus' });

    mediaRecorder.ondataavailable = (event) => {
        if (event.data.size === 0) return;

        // queue the send so we know when the last chunk is done
        sendChain = sendChain.then(async () => {
            const buffer = await event.data.arrayBuffer();
            if (socket.readyState === WebSocket.OPEN) {
                socket.send(buffer);
            }
        });
    };

    mediaRecorder.onstop = async () => {
        await sendChain;                       // wait for final chunk to finish sending
        stream.getTracks().forEach(track => track.stop());
        if (socket.readyState === WebSocket.OPEN) {
            socket.close(1000, "done");
        }
    };

    mediaRecorder.start(250); // send data every 250ms
}

async function stopAudioStream() {
    mediaRecorder.stop();
}

function toggleMic() {
    if (socket && socket.readyState === WebSocket.OPEN) {
        console.log('Stop streaming');
        stopAudioStream();
        audioContainer.classList.remove('streaming');

    } else {
        console.log("Start streaming");
        startAudioStream();
        audioContainer.classList.add('streaming');
    }
}





window.addEventListener('keydown', (event) => {
    const combo = normalizeCombo(event);
    const handler = shortcuts.get(combo);
    if (handler) {
        event.preventDefault();
        handler(event);
    }
});