import requests
import json

# Set the URL for your Flask API with the desired port
api_url = 'http://localhost:6061/speech-to-text'

# Audio data to be transcribed
# Note: Replace 'your_audio_data_here' with the actual audio data in a suitable format
audio_data = '/home/fred/Projetos/ermis_demo/data/demo_pt.wav'

# Data for the POST request
data = {'audio': audio_data}

# Convert the dictionary to JSON
json_data = json.dumps(data)

# Set up headers
headers = {'Content-Type': 'application/json'}

# Make the POST request
response = requests.post(api_url, data=json_data, headers=headers)

# Display the API response
print(response.json())
