from flask import Flask, request, jsonify
from whisper import Whisper, FasterWhisper
import traceback

app = Flask(__name__)

# stt = Whisper()
stt = FasterWhisper()

@app.route('/transcribe', methods=['POST'])
def transcribe():
    data = request.get_json()

    if 'audio' not in data:
        return jsonify({'error': 'Audio data missing in the request body'}), 400

    audio_filepath = data['audio']
    audio_data = stt.load_filepath(audio_filepath)
    try:
        transcribed_text = stt.transcribe(audio_data)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Failed to transcribe the audio:  {e}'}), 500
    if transcribed_text:
        return jsonify({'text': transcribed_text}), 200
    else:
        return jsonify({'error': 'Failed to transcribe the audio'}), 500
    
@app.route('/translate', methods=['POST'])
def translate():
    data = request.get_json()

    if 'audio' not in data:
        return jsonify({'error': 'Audio data missing in the request body'}), 400

    audio_filepath = data['audio']
    audio_data = stt.load_filepath(audio_filepath)
    try:
        transcribed_text = stt.translate(audio_data)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Failed to transcribe the audio:  {e}'}), 500
    if transcribed_text:
        return jsonify({'text': transcribed_text}), 200
    else:
        return jsonify({'error': 'Failed to transcribe the audio'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=6061)
