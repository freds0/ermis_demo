import os
import torch
import torchaudio
import torchaudio.transforms as T
from stt_interface import SpeechToTextInterface
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from faster_whisper import WhisperModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Whisper(SpeechToTextInterface):
    def __init__(self, model_id: str = os.getenv("ASR_MODEL", "openai/whisper-base")):
        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_id).to(
            device
        )
        self.sr_target = 16000

    def load_filepath(self, filepath: str = None) -> torch.Tensor:
        waveform, sample_rate = torchaudio.load(filepath)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(0, keepdim=True)
        if sample_rate != self.sr_target:
            resampler = T.Resample(sample_rate, self.sr_target)
            waveform = resampler(waveform)
        return waveform.squeeze()

    def transcribe(self, audio_data, language="en") -> str:
        if isinstance(audio_data, str):
            audio_data = self.load_filepath(audio_data)
        input_features = self.processor(
            audio_data, sampling_rate=self.sr_target, return_tensors="pt"
        ).input_features
        with torch.no_grad():
            predicted_ids = self.model.generate(
                input_features.to(device),
                forced_decoder_ids=self.processor.get_decoder_prompt_ids(
                    language=language, task="transcribe"
                ),
            )
        transcription = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )
        return transcription[0]

    def translate(self, audio_data, target_language="en"):
        if isinstance(audio_data, str):
            audio_data = self.load_filepath(audio_data)
        input_features = self.processor(
            audio_data, sampling_rate=self.sr_target, return_tensors="pt"
        ).input_features
        with torch.no_grad():
            predicted_ids = self.model.generate(
                input_features.to(device),
                forced_decoder_ids=self.processor.get_decoder_prompt_ids(
                    language=target_language, task="translate"
                ),
            )
        translation = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )
        return translation[0]


class FasterWhisper(SpeechToTextInterface):
    def __init__(self, model_id: str = os.getenv("ASR_MODEL", "base")):
        model_path = os.getenv(
            "ASR_MODEL_PATH", os.path.join(os.path.expanduser("~"), ".cache", "whisper")
        )
        if torch.cuda.is_available():
            device = "cuda"
            model_quantization = os.getenv("ASR_QUANTIZATION", "float32")
        else:
            device = "cpu"
            model_quantization = os.getenv("ASR_QUANTIZATION", "int8")
        self.model = WhisperModel(
            model_id,
            device=device,
            compute_type=model_quantization,
            download_root=model_path,
        )
        self.sr_target = 16000

    def load_filepath(self, filepath: str = None) -> torch.Tensor:
        waveform, sample_rate = torchaudio.load(filepath)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(0, keepdim=True)
        if sample_rate != self.sr_target:
            resampler = T.Resample(sample_rate, self.sr_target)
            waveform = resampler(waveform)
        return waveform.squeeze().detach().cpu().numpy()

    def transcribe(self, audio_data, language="en") -> str:
        if isinstance(audio_data, str):
            audio_data = self.load_filepath(audio_data)
        segment_generator, info = self.model.transcribe(
            audio_data, beam_size=5, language=language, task="transcribe"
        )
        segments = []
        text = ""
        for segment in segment_generator:
            segments.append(segment)
            text = text + segment.text
        return text

    def translate(self, audio_data, target_language="en"):
        segment_generator, info = self.model.transcribe(
            audio_data, beam_size=5, language=target_language, task="translate"
        )
        segments = []
        text = ""
        for segment in segment_generator:
            segments.append(segment)
            text = text + segment.text
        return text
