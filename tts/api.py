from flask import Flask, request, jsonify
from google_tts import googleTTS
from xtts import XTTS

app = Flask(__name__)


def get_model(model_id):
    options = {
        'gtts': googleTTS(),
        'xtts': XTTS()
    }
    model = options.get(model_id, 'Opção inválida.')
    return model


@app.route('/text-to-speech', methods=['POST'])
def text_to_speech():
    data = request.get_json()

    if 'text' not in data:
        return jsonify({'error': 'Text missing in the request body'}), 400

    text = data['text']
    language = data.get('language', 'en')  # Default to English
    filepath = data.get('filepath')

    tts_model = get_model("xtts")
    tts_model.synthesize_to_file(
        input_text=text, 
        lang=language, 
        filepath=filepath
    )

    return jsonify({'message': 'Text to speech conversion completed'}), 200

if __name__ == '__main__':
    app.run(debug=True, port=6060)
