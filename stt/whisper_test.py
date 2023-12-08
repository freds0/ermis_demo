from os.path import abspath, dirname, join
from whisper import SpeechToText

def main():
    stt = SpeechToText()
    audio_data = stt.load_filepath(filepath=join(abspath(dirname(__file__)), "../data/demo_pt.wav"))
    text = stt.transcribe(waveform=audio_data)
    print(text)


if __name__ == "__main__":
    main()
