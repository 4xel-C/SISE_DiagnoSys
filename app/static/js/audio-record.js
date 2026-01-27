import { startAudioStream, stopAudioStream, isRecording } from './modules/streamer.js';

const main = document.querySelector('main');
const audioRecord = main.querySelector('.audio-record');
const waveformCanvas = audioRecord.querySelector('canvas#waveform');

let waveform, timer;

// Shortcuts
const shortcuts = new Map([
    ['Shift+Space', toggleMic]
]);






function createWaveformDrawer(stream, canvas) {
    const audioCtx = new AudioContext();
    const analyser = audioCtx.createAnalyser();
    analyser.fftSize = 2048;
    const bufferLength = analyser.fftSize;
    const dataArray = new Uint8Array(bufferLength);

    const source = audioCtx.createMediaStreamSource(stream);
    source.connect(analyser);

    const canvasCtx = canvas.getContext('2d');

    function draw() {
        analyser.getByteTimeDomainData(dataArray);

        canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
        canvasCtx.lineWidth = 2;
        canvasCtx.strokeStyle = '#8359E8';
        canvasCtx.beginPath();

        const sliceWidth = canvas.width / bufferLength;
        let x = 0;

        for (let i = 0; i < bufferLength; i++) {
            const v = dataArray[i] / 128.0;
            const y = (v * canvas.height) / 2;

            i === 0 ? canvasCtx.moveTo(x, y) : canvasCtx.lineTo(x, y);
            x += sliceWidth;
        }
        canvasCtx.lineTo(canvas.width, canvas.height / 2);
        canvasCtx.stroke();

        requestAnimationFrame(draw);
    }

    draw();

    return audioCtx
}

function createTimerInterval(timerElement) {
    let totalSeconds = 0;

    function updateDisplay() {
        const minutes = String(Math.floor(totalSeconds / 60)).padStart(2, '0');
        const seconds = String(totalSeconds % 60).padStart(2, '0');
        timerElement.textContent = `${minutes}:${seconds}`;
    }

    const intervalId = setInterval(() => {
        totalSeconds++;
        updateDisplay();
    }, 1000);

    return intervalId;
}

async function start(patientId) {
    // Start stream
    const stream = await startAudioStream(patientId);
    // Setup waveform + interval
    const timerElement = document.querySelector('main .top-bar .timer');
    timer = createTimerInterval(timerElement);
    waveform = createWaveformDrawer(stream, waveformCanvas);
    // Save recorded patient id
    main.dataset.recordPatientId = patientId;
    // Update UI
    main.classList.add('streaming');
    // Dispatch event
    document.dispatchEvent(
        new CustomEvent('audioRecordStarted', {
            detail: { patientId }
        })
    );
}

async function stop(patientId) {
    // Stop audio stream and send complete audio
    await stopAudioStream();
    // Close audio analyser and clear interval
    waveform.close();
    clearInterval(timer);
    // Clear recorded patient id
    main.dataset.recordPatientId = null;
    // Update UI
    main.classList.remove('streaming');
    const timerElement = document.querySelector('main .top-bar .timer');
    timerElement.textContent = '00:00';
    // Dispatch event
    document.dispatchEvent(
        new CustomEvent('audioRecordStoped', {
            detail: { patientId }
        })
    );
}

function toggleMic() {
    if (isRecording()) {
        const patientId = main.dataset.recordPatientId;
        stop(patientId);
    } else {
        const patientId = main.querySelector('.patient').dataset.patientId;
        start(patientId);
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

audioRecord.addEventListener('click', () => {
    toggleMic();
})