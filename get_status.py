import requests

# Replace with your deployed URL and plant_id
BASE_URL = "https://uta2025hackathon--plant-backend-fastapi-app.modal.run"
plant_id = "my_first_plant"

# Call the status endpoint
response = requests.get(f"{BASE_URL}/plant/{plant_id}/status")

# Print the result
if response.status_code == 200:
    print("Plant Status:", response.json())
else:
    print("Error:", response.status_code, response.text)