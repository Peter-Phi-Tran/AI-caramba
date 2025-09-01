# AI-caramba 🌱
**OpenAI Open Model Hackathon - Smart Plant Care Assistant**

An AI-powered plant monitoring system that uses ESP32 sensors and GPT-OSS models to create an interactive plant companion. Your plants can now communicate their needs and chat with you!

## Project Overview

AI-caramba transforms plant care through intelligent sensor monitoring and AI-driven personality. Using the `gpt-oss-20b` model, plants gain unique personalities and can express their needs in natural language based on real-time environmental data.

### Key Features
- **Real-time sensor monitoring** (soil moisture, light, humidity)
- **AI plant personality** powered by GPT-OSS-20B
- **Interactive chat** with your plants
- **Mood-based care recommendations**
- **REST API** for hardware integration
- **ESP32 compatible** with HTTPS support

## Quick Start

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

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/sensor-data` | Send sensor readings |
| POST | `/chat` | Chat with the plant |
| GET | `/plant/{plant_id}/status` | Get current plant status |
| GET | `/plant/{plant_id}/history` | Get chat history |

## Plant Moods

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

## Sensor Ranges

| Sensor | Ideal Range | Units |
|--------|-------------|-------|
| Soil Moisture | 40-60% | Percentage |
| Light Level | 40-80% | Percentage |
| Temperature | 18-25°C | Celsius |
| Humidity | 40-60% | Percentage |

## Testing

Run the test function:
```bash
modal run plant_backend.py::test_plant_system
```

## Example Response

```json
{
  "plant_id": "my_plant_001",
  "mood": "thirsty",
  "response": "Hey there! I'm feeling pretty parched - my soil is getting dry and I could really use a drink! 💧",
  "needs": ["water"],
  "timestamp": "2025-08-27T10:30:00"
}
```

## Development

### Option 1:
1. Change the app name in your copy: `app = modal.App("plant-backend-yourname")`
2. Deploy your version: `modal deploy your_file.py`
3. You'll get your own URL to test

### Option 2:
1. Copy the code to your own Modal account
2. Modify as needed
3. Deploy with your own app name

## Architecture

### Tech Stack
- **Backend**: FastAPI + Modal for serverless deployment
- **AI Model**: GPT-OSS-20B (20 billion parameter open model)
- **Hardware**: ESP32 with WiFi and sensor capabilities
- **Cloud**: Modal Labs for GPU inference and API hosting

### Hardware Components
- ESP32 microcontroller
- Soil moisture sensor
- Temperature sensor
- humidity sensor

## Project Timeline

**Milestone & Timeline:**
- Finalize requirements – Aug 18
- Hardware assembly complete – Aug 22
- MVP demo complete – Aug 31
- Final video & polish – Sep 8
- **Deadline: September 11th, 2025**

## Hackathon Categories

This project targets the following OpenAI Open Model Hackathon categories:

1. **For Humanity** - Making plant care accessible and educational
2. **Best Local Agent** - AI personality that adapts to local sensor conditions
3. **Wildcard** - Unexpected application of LLMs for plant communication
4. **Best in Robotics** - Hardware-software integration with ESP32

## Demo Features

- Real-time sensor data processing
- Conversational AI with plant-specific personality
- Mood-based care recommendations
- Hardware integration demonstration
- Offline capability planning (future enhancement)

## Installation & Setup

### Dependencies
```bash
pip install modal fastapi uvicorn pydantic transformers torch
```

### ESP32 Setup
1. Install Arduino IDE 
2. Add ESP32 board support
3. Install HTTPClient and WiFi libraries
4. Upload `component_data.cpp` to your ESP32
5. Configure WiFi credentials and sensor pins

### Modal Deployment
```bash
# Install Modal CLI
pip install modal

# Authenticate
modal token new

# Deploy
modal deploy plant_backend.py
```

## How It Works

1. **Sensor Data Collection**: ESP32 reads environmental sensors every 30 seconds
2. **Data Transmission**: HTTPS POST request sends JSON data to Modal API
3. **Mood Analysis**: Python backend calculates plant mood based on sensor ranges
4. **AI Response Generation**: GPT-OSS-20B generates personality-driven responses
5. **User Interaction**: Chat endpoint allows natural language conversation
6. **Care Recommendations**: System provides specific care instructions based on needs

## Troubleshooting

### Common ESP32 Issues
- **Timeout errors**: Increase `setTimeout()` to 180000ms (3 minutes) for Modal cold starts
- **SSL/TLS issues**: Use `client.setInsecure()` for testing
- **Empty responses**: Check Modal logs for internal errors

### Modal API Issues
- **Cold starts**: First request may take 60+ seconds
- **Memory limits**: Model requires GPU with sufficient VRAM
- **Rate limits**: Deploy with `min_containers=1` for consistent performance

## Resources

- [OpenAI Open Model Hackathon Rules](https://openai.devpost.com/rules)
- [OpenAI Open Model Resources](https://openai.devpost.com/resources)
- [Modal Labs Documentation](https://modal.com/docs)
- [ESP32 Arduino Documentation](https://docs.espressif.com/projects/arduino-esp32/)

## Team

- **Peter** - Research, Prompting, and Testing the Open, API
- **Chau** - Model & cloud initial cloud set up
- **Chris** - Developing AI, API
- **Nhut** - Hardware Components, Testing, API

---

*Made with 🌱 for the OpenAI Open Model Hackathon 2025*
