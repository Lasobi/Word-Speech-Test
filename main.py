import whisper
import sounddevice
import wavio


def record_audio():
    frequency = 44400
    duration = 5

    print("Recording...")
    recording = sounddevice.rec(int(duration * frequency), samplerate=frequency, channels=2)
    sounddevice.wait()
    print("Recording stopped")

    wavio.write("out.wav", recording, frequency, sampwidth=2)


def run():
    record_audio()

    model = whisper.load_model("medium")

    audio = whisper.load_audio("out.wav")
    audio = whisper.pad_or_trim(audio)

    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")

    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)

    print(result.text)


if __name__ == "__main__":
    run()
