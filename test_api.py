import requests
import json

# API URL
url = "https://ml-t4wt.onrender.com/predict"

# Sample input features
data = {
    "features": [1,30,45]  # Replace with actual feature values if needed
}

# Convert to JSON format
headers = {"Content-Type": "application/json"}
payload = json.dumps(data)

# Send POST request
response = requests.post(url, data=payload, headers=headers)

# Check the response status
if response.status_code == 200:
    print("✅ API Response:", response.json())
else:
    print(f"❌ Error: {response.status_code}")
    print("Response Text:", response.text)
