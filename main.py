import whisper
import sounddevice
import wavio
import os


# Function records audio for use by whisper
def record_audio():
    frequency = 44400
    duration = 5

    print("Recording...")
    recording = sounddevice.rec(int(duration * frequency), samplerate=frequency, channels=2)
    sounddevice.wait()
    print("Recording stopped")

    wavio.write("out.wav", recording, frequency, sampwidth=2)


# Uses whisper to convert speech to text for further use
def audio_to_text():
    model = whisper.load_model("medium")

    audio = whisper.load_audio("out.wav")
    audio = whisper.pad_or_trim(audio)

    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")

    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)

    print(result.text)

    return result.text


# Function to clean up any temporary files that are no longer used
def remove_file(file):
    os.remove(file)


def run():
    record_audio()
    audio_to_text()
    remove_file("out.wav")


# Main function, it all starts here
if __name__ == "__main__":
    run()
