from abc import ABC, abstractmethod

class SpeechToTextInterface(ABC):

    @abstractmethod
    def load_filepath(self, filepath, sr_target=16000):
        """Load audio data from filepath."""
        pass

    @abstractmethod
    def transcribe(self, audio_data, language="en"):
        """Extract text from the currently loaded file."""
        pass

    @abstractmethod
    def translate(self, audio_data, target_language="en"):
        """Translate text to another language."""
        pass