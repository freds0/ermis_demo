from flask import Flask, request, jsonify
from whisper import SpeechToText

app = Flask(__name__)

@app.route('/speech-to-text', methods=['POST'])
def speech_to_text():
    data = request.get_json()

    if 'audio' not in data:
        return jsonify({'error': 'Audio data missing in the request body'}), 400

    audio_filepath = data['audio']
    stt = SpeechToText()
    audio_data = stt.load_filepath(audio_filepath)
    transcribed_text = stt.transcribe(audio_data)
    if transcribed_text:
        return jsonify({'text': transcribed_text}), 200
    else:
        return jsonify({'error': 'Failed to transcribe the audio'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=6061)
