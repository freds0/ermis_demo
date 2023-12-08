import torch
import torchaudio
import torchaudio.transforms as T
from stt_interface import SpeechToTextInterface
from transformers import WhisperProcessor, WhisperForConditionalGeneration

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SpeechToText(SpeechToTextInterface):

    def __init__(self, model_id: str = "openai/whisper-large-v2"):
        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device)
        self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(language="english", task="transcribe")
        self.sr_target = 16000

    def load_filepath(self, filepath: str = None) -> torch.Tensor:
        waveform, sample_rate = torchaudio.load(filepath)
        if sample_rate != self.sr_target:
            resampler = T.Resample(sample_rate, self.sr_target)
            waveform = resampler(waveform)
        return waveform.squeeze()

    def transcribe(self, waveform: torch.Tensor) -> str:
        #waveform = self.load_filepath(input_filepath)
        input_features = self.processor(waveform, sampling_rate=self.sr_target, return_tensors="pt").input_features 
        with torch.no_grad():
            predicted_ids = self.model.generate(input_features.to(device), forced_decoder_ids=self.forced_decoder_ids)            
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
        return transcription[0]
