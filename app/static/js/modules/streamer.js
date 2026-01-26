export let socket;

let stream;
let mediaRecorder;
let sendChain = Promise.resolve();


export async function startAudioStream(patientId) {
    // Open websocket
    socket = new WebSocket(`ws://${location.host}/audio_stt?patient_id=${patientId}`);
    await new Promise(resolve => socket.onopen = resolve);
    // Start stream
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm;codecs=opus' });

    // On data available -> send data
    mediaRecorder.ondataavailable = (event) => {
        if (event.data.size === 0) return;
        // Queue the send so we know when the last chunk is done
        sendChain = sendChain.then(async () => {
            const buffer = await event.data.arrayBuffer();
            if (socket.readyState === WebSocket.OPEN) {
                socket.send(buffer);
            }
        });
    };
    // On websocket closed -> dispatch event
    socket.onclose = () => {
        document.dispatchEvent(
            new CustomEvent('audioProcessCompleted', {
                detail: { patientId }
            })
        );
    }
    
    mediaRecorder.start(250); // send data every 250ms

    return stream
}


export async function stopAudioStream() {
    if (socket.readyState === WebSocket.OPEN) {
        await sendChain;
        mediaRecorder.stop();
        stream.getTracks().forEach(track => track.stop());
        socket.send('stop');
    }
}