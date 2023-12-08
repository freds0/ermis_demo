from os.path import abspath, dirname, join, exists
from xtts import XTTS
import numpy as np

def main():
    #tts = XTTS()
    speaker_wav_path=join(abspath(dirname(__file__)), "../data/demo_pt.wav")
    assert exists(speaker_wav_path), f"O arquivo de áudio de referência não foi encontrado em {speaker_wav_path}"
  
    text = "Under the thicket of the shrubs are the couples who give those melancholic places a more alive and human affection."
    xtts = XTTS()
    xtts.synthesize_to_file(text)

if __name__ == "__main__":
    main()