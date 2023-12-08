
class SpeechToTextInterface:

    def load_filepath(self, filepath, sr_target=16000):
        """Load audio data from filepath."""
        pass

    def transcribe(self, audio_data):
        """Extract text from the currently loaded file."""
        pass
