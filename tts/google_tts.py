import torch
from tts_interface import TextToSpeechInterface
from io import BytesIO
from gtts import gTTS

device = "cuda" if torch.cuda.is_available() else "cpu"

class googleTTS(TextToSpeechInterface):
    '''
    https://gtts.readthedocs.io/en/latest/module.html#examples
    '''
    def synthesize(self, input_text: str = "Hello World") -> torch.Tensor:
        mp3_fp = BytesIO()
        tts = gTTS(input_text, lang='en')
        tts.write_to_fp(mp3_fp)
        return mp3_fp

    def synthesize_to_file(self, input_text: str = "Hello World", lang: str = "en", filepath: str = "result.mp3") -> None:        
        tts = gTTS(input_text, lang=lang)
        tts.save(filepath)