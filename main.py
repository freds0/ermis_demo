import requests
import json

def transcribe_audio(input_filepath):

    # Set the URL for your Flask API with the desired port
    stt_api_url = 'http://localhost:6061/speech-to-text'

    audio_data = input_filepath

    # Data for the POST request
    data = {'audio': audio_data}

    # Convert the dictionary to JSON
    json_data = json.dumps(data)

    # Set up headers
    headers = {'Content-Type': 'application/json'}

    # Make the POST request
    response = requests.post(stt_api_url, data=json_data, headers=headers)

    return response.json()


def synthesize_audio(input_text, filepath):

    # Set the URL for your Flask API
    tts_api_url = 'http://localhost:6060/text-to-speech'

    # Data for the POST request
    data = {
            'text': input_text, 
            'language': 'en',
            'filepath': filepath
        }

    # Convert the dictionary to JSON
    json_data = json.dumps(data)

    # Set up headers
    headers = {'Content-Type': 'application/json'}

    # Make the POST request
    response = requests.post(tts_api_url, data=json_data, headers=headers)

    # Display the API response
    return response.json()


if __name__ == "__main__":
    input_filepath = "/home/fred/Projetos/ermis_demo/data/demo_en.wav"
    output_filepath = "/home/fred/Projetos/ermis_demo/data/output.wav"
    transcription = transcribe_audio(input_filepath)
    r = synthesize_audio(transcription, output_filepath)
    print(r)