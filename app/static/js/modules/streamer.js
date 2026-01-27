let stream;
let mediaRecorder;
let audioChunks = [];
let currentPatientId = null;
let _isRecording = false;

export function isRecording() {
    return _isRecording;
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

    return stream;
}


export async function stopAudioStream() {
    if (!_isRecording) return;

    _isRecording = false;
    const patientId = currentPatientId;

    // Stop recording and wait for final data
    await new Promise(resolve => {
        mediaRecorder.onstop = resolve;
        mediaRecorder.stop();
    });

    // Stop audio tracks
    stream.getTracks().forEach(track => track.stop());

    // Send complete audio via POST if we have data
    if (audioChunks.length > 0) {
        const audioBlob = new Blob(audioChunks, { type: 'audio/webm;codecs=opus' });

        try {
            const response = await fetch(`/ajax/audio_stt/${patientId}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'audio/webm'
                },
                body: audioBlob
            });

            if (!response.ok) {
                console.error('Failed to transcribe audio:', response.status);
            }
        } catch (error) {
            console.error('Error sending audio:', error);
        }
    }

    // Dispatch completion event
    document.dispatchEvent(
        new CustomEvent('audioProcessCompleted', {
            detail: { patientId }
        })
    );
}