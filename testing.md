
## 🚀 Quick Start

### Deploy the API
```bash
modal deploy plant_backend.py
```

After deployment, your API is live at:
```
https://uta2025hackathon--plant-backend-fastapi-app.modal.run
```

### Test the API

#### 1. Health Check
```bash
curl https://uta2025hackathon--plant-backend-fastapi-app.modal.run/health
```

#### 2. Send Sensor Data
```bash
curl -X POST https://uta2025hackathon--plant-backend-fastapi-app.modal.run/sensor-data \
  -H "Content-Type: application/json" \
  -d '{
    "plant_id": "my_plant_001",
    "soil_moisture": 25.0,
    "light_level": 65.0,
    "temperature": 22.0,
    "humidity": 50.0
  }'
```

#### 3. Chat with Your Plant
```bash
curl -X POST https://uta2025hackathon--plant-backend-fastapi-app.modal.run/chat \
  -H "Content-Type: application/json" \
  -d '{
    "plant_id": "my_plant_001",
    "message": "How are you feeling today?"
  }'
```

### PowerShell Testing (Windows)
```powershell
# Test with PowerShell
$sensorData = @{
    plant_id = "my_first_plant"
    soil_moisture = 25.0
    light_level = 70.0
    temperature = 22.0
    humidity = 50.0
} | ConvertTo-Json

Invoke-RestMethod -Uri "https://uta2025hackathon--plant-backend-fastapi-app.modal.run/sensor-data" -Method Post -Body $sensorData -ContentType "application/json"
```

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/sensor-data` | Send sensor readings |
| POST | `/chat` | Chat with the plant |
| GET | `/plant/{plant_id}/status` | Get current plant status |
| GET | `/plant/{plant_id}/history` | Get chat history |

## 🌿 Plant Moods

Based on sensor readings, your plant can be:
- **very_happy** - Everything is perfect! 🌟
- **happy** - Content and chatty
- **okay** - Doing alright
- **sad** - Needs some attention
- **very_sad** - Struggling, needs help!
- **thirsty** - Needs water
- **drowning** - Too much water!
- **cold/hot** - Temperature issues
- **stressed** - Environmental stress

## 💡 Sensor Ranges

| Sensor | Ideal Range | Units |
|--------|-------------|-------|
| Soil Moisture | 40-60% | Percentage |
| Light Level | 40-80% | Percentage |
| Temperature | 18-25°C | Celsius |
| Humidity | 40-60% | Percentage |

## 🧪 Testing

Run the test function:
```bash
modal run plant_backend.py::test_plant_system
```

## 📝 Example Response

```json
{
  "plant_id": "my_plant_001",
  "mood": "thirsty",
  "response": "Hey there! I'm feeling pretty parched - my soil is getting dry and I could really use a drink! 💧",
  "needs": ["water"],
  "timestamp": "2025-08-27T10:30:00"
}
```

## 🛠️ Development

### Option 1:
1. Change the app name in your copy: `app = modal.App("plant-backend-yourname")`
2. Deploy your version: `modal deploy your_file.py`
3. You'll get your own URL to test

### Option 2:
1. Copy the code to your own Modal account
2. Modify as needed
3. Deploy with your own app name

### Common Changes:
- **Plant personality**: Edit the `base_personality` string in `create_plant_prompt()`
- **Mood thresholds**: Modify ranges in `PlantMoodEngine.calculate_mood()`
- **Fallback responses**: Update `create_fallback_response()` function
- **API endpoints**: Add new routes in the FastAPI section
