import requests
import json

# Set the URL for your Flask API
api_url = 'http://localhost:6060/text-to-speech'

# Text you want to convert to speech
text_to_convert = "Hello, this is a test."

# Data for the POST request
data = {
        'text': text_to_convert, 
        'language': 'en',
        'filepath': 'result.wav'
       }

# Convert the dictionary to JSON
json_data = json.dumps(data)

# Set up headers
headers = {'Content-Type': 'application/json'}

# Make the POST request
response = requests.post(api_url, data=json_data, headers=headers)

# Display the API response
print(response.json())