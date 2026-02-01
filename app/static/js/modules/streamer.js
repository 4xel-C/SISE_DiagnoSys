let stream;
let mediaRecorder;
let audioChunks = [];
let currentPatientId = null;
let _isRecording = false;
let periodicSendInterval = null;

const PERIODIC_SEND_INTERVAL_MS = 20000; // 20 seconds

export function isRecording() {
    return _isRecording;
}

async function sendAudioChunks(patientId, chunks) {
    if (chunks.length === 0) return;

    const audioBlob = new Blob(chunks, { type: 'audio/webm;codecs=opus' });

    try {
        const response = await fetch(`/ajax/audio_stt/${patientId}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'audio/webm'
            },
            body: audioBlob
        });

        if (response.ok) {
            document.dispatchEvent(
                new CustomEvent('audioProcessCompleted', {
                    detail: { patientId }
                })
            );  
        } else {
            console.error('Failed to transcribe audio:', response.status);
            const content = await response.json();
            document.dispatchEvent(
                new CustomEvent('audioProcessError', {
                    detail: { 
                        patientId,
                        error: content?.error ?? 'Erreur interne'
                    }
                })
            );
        }

    } catch (error) {
        console.error('Error sending audio:', error);
        document.dispatchEvent(
            new CustomEvent('audioProcessError', {
                detail: { 
                    patientId,
                    error: "Impossible d'envoyer l'audio"
                }
            })
        );
    }
}

function startPeriodicSend(patientId) {
    periodicSendInterval = setInterval(() => {
        if (audioChunks.length > 0) {
            // Copy current chunks and clear buffer
            const chunksToSend = [...audioChunks];
            audioChunks = [];

            // Send accumulated audio
            sendAudioChunks(patientId, chunksToSend);
        }
    }, PERIODIC_SEND_INTERVAL_MS);
}

function stopPeriodicSend() {
    if (periodicSendInterval) {
        clearInterval(periodicSendInterval);
        periodicSendInterval = null;
    }
}

export async function startAudioStream(patientId) {
    // Reset state
    audioChunks = [];
    currentPatientId = patientId;
    _isRecording = true;

    // Start stream
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm;codecs=opus' });

    // On data available -> accumulate chunks
    mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
            audioChunks.push(event.data);
        }
    };

    mediaRecorder.start(250); // collect data every 250ms

    // Start periodic context update every 20 seconds
    startPeriodicSend(patientId);

    return stream;
}


export async function stopAudioStream() {
    if (!_isRecording) return;

    _isRecording = false;
    const patientId = currentPatientId;

    // Stop periodic send
    stopPeriodicSend();

    // Stop recording and wait for final data
    await new Promise(resolve => {
        mediaRecorder.onstop = resolve;
        mediaRecorder.stop();
    });

    // Stop audio tracks
    stream.getTracks().forEach(track => track.stop());

    // Send remaining audio chunks
    if (audioChunks.length > 0) {
        sendAudioChunks(patientId, audioChunks);
        audioChunks = [];
    }
}