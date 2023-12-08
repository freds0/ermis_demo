import torch
from TTS.api import TTS
from tts_interface import TextToSpeechInterface

device = "cuda" if torch.cuda.is_available() else "cpu"

class XTTS(TextToSpeechInterface):

    def __init__(self):
        self.model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        
    def synthesize(self, input_text: str = "Hello World") -> torch.Tensor:        
        wav = self.model.tts(
            text=input_text, 
            speaker_wav="data/demo_pt.wav", 
            language="en"
        )
        return wav
    
    def synthesize_to_file(self, input_text: str = "Hello World", lang: str = "en", filepath: str = "result.wav") -> None:        
        self.model.tts_to_file(
            text=input_text, 
            language=lang, 
            speaker_wav="/home/fred/Projetos/ermis_demo/data/demo_pt.wav",
            file_path=filepath
        )