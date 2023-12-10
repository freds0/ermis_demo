import requests
import json
import sys

# Set the URL for your Flask API with the desired port
api_url = 'http://localhost:6061/translate'

# Audio data to be transcribed
# Note: Replace 'your_audio_data_here' with the actual audio data in a suitable format
audio_data = sys.argv[1]

# Data for the POST request
data = {'audio': audio_data}

# Convert the dictionary to JSON
json_data = json.dumps(data)
print(json_data)

# Set up headers
headers = {'Content-Type': 'application/json'}

# Make the POST request
response = requests.post(api_url, data=json_data, headers=headers)

if response.status_code == 200 or response.status_code == 500:
    print(response.json())
else:
    print(response.status_code)