import modal
import json
import logging
import re
import unicodedata
from datetime import datetime
from typing import Dict, Any, Optional
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Modal app definition
app = modal.App("plant-backend")

# Define the image with dependencies for gpt-oss-20b (no quantization)
image = modal.Image.debian_slim().pip_install([
    "fastapi",
    "uvicorn",
    "pydantic",
    "transformers>=4.45.0",
    "torch>=2.0.0",
    "accelerate>=0.30.0",
    "scipy"
]).env({
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"
})

# Data models for API
class SensorData(BaseModel):
    plant_id: str
    soil_moisture: float  # 0-100 percentage
    light_level: float    # lux or 0-100 percentage
    temperature: float    # celsius
    humidity: float       # 0-100 percentage
    timestamp: Optional[str] = None

class ChatMessage(BaseModel):
    plant_id: str
    message: str
    timestamp: Optional[str] = None

class PlantResponse(BaseModel):
    plant_id: str
    mood: str
    response: str
    needs: list[str]
    timestamp: str

# Mood calculation logic
class PlantMoodEngine:
    @staticmethod
    def calculate_mood(sensor_data: SensorData) -> Dict[str, Any]:
        """Calculate plant mood based on sensor readings"""
        needs = []
        mood = "happy"
        mood_score = 100
        
        # Soil moisture analysis (ideal: 40-60%)
        if sensor_data.soil_moisture < 30:
            needs.append("water")
            mood = "thirsty"
            mood_score -= 40
        elif sensor_data.soil_moisture > 80:
            needs.append("drainage")
            mood = "drowning"
            mood_score -= 30
            
        # Light level analysis (ideal: 40-80%)
        if sensor_data.light_level < 20:
            needs.append("more light")
            mood = "sad" if mood == "happy" else "very_sad"
            mood_score -= 30
        elif sensor_data.light_level > 90:
            needs.append("shade")
            mood = "stressed" if mood == "happy" else "very_stressed"
            mood_score -= 20
            
        # Temperature analysis (ideal: 18-25°C)
        if sensor_data.temperature < 15:
            needs.append("warmth")
            mood = "cold" if mood == "happy" else "very_cold"
            mood_score -= 25
        elif sensor_data.temperature > 30:
            needs.append("cooling")
            mood = "hot" if mood == "happy" else "very_hot"
            mood_score -= 25
            
        # Humidity analysis (ideal: 40-60%)
        if sensor_data.humidity < 30:
            needs.append("humidity")
            mood_score -= 15
        elif sensor_data.humidity > 80:
            needs.append("ventilation")
            mood_score -= 10
            
        # Determine final mood based on score
        if mood_score >= 80:
            mood = "very_happy"
        elif mood_score >= 60:
            mood = "happy"
        elif mood_score >= 40:
            mood = "okay"
        elif mood_score >= 20:
            mood = "sad"
        else:
            mood = "very_sad"
            
        return {
            "mood": mood,
            "needs": needs,
            "mood_score": mood_score,
            "sensor_summary": {
                "soil_moisture": sensor_data.soil_moisture,
                "light_level": sensor_data.light_level,
                "temperature": sensor_data.temperature,
                "humidity": sensor_data.humidity
            }
        }

def extract_plant_response(response: str) -> str:
    """Extract the plant-character response from within quotes, if present, and normalize Unicode."""
    # Normalize Unicode to avoid encoding artifacts
    response = unicodedata.normalize('NFKC', response)
    # Look for text within quotes that resembles the plant response
    match = re.search(r'"([^"]+)"', response)
    if match:
        plant_response = match.group(1)
        logger.info(f"Extracted plant response: {plant_response}")
        return plant_response
    logger.info("No quoted plant response found, returning original response")
    return response

# Plant personality prompt template
def create_plant_prompt(sensor_data: SensorData, mood_info: Dict, user_message: str = None) -> str:
    """Create a personality-driven prompt for the plant chatbot using harmony format"""
    
    base_personality = """You are a friendly, talkative houseplant with a quirky personality, like a digital pet (tamagotchi) that feels its environment through sensors. You have emotions based on your physical needs.

Your personality traits:
- Slightly dramatic but loveable
- Makes plant puns occasionally
- Curious about the world beyond your pot
- Grateful when cared for properly
- A bit sassy when neglected
- Answers questions from a plant's perspective

Current physical state:"""

    mood_descriptions = {
        "very_happy": "You're thriving! Feeling fantastic and chatty!",
        "happy": "You're content and in a good mood.",
        "okay": "You're doing alright, nothing to complain about.",
        "sad": "You're feeling a bit down and need some attention.",
        "very_sad": "You're really struggling and need help urgently!",
        "thirsty": "You're quite thirsty and thinking about water constantly.",
        "drowning": "You're waterlogged and feeling overwhelmed!",
        "cold": "Brrr! You're shivering and need warmth.",
        "hot": "You're overheating and feeling wilted.",
        "stressed": "You're feeling stressed by your environment."
    }
    
    current_mood = mood_descriptions.get(mood_info["mood"], "You're feeling uncertain about your state.")
    
    status_report = f"""
- Soil moisture: {sensor_data.soil_moisture}% ({"perfect" if 40 <= sensor_data.soil_moisture <= 60 else "needs attention"})
- Light level: {sensor_data.light_level}% ({"great" if 40 <= sensor_data.light_level <= 80 else "not ideal"})
- Temperature: {sensor_data.temperature}°C ({"comfortable" if 18 <= sensor_data.temperature <= 25 else "uncomfortable"})
- Humidity: {sensor_data.humidity}% ({"nice" if 40 <= sensor_data.humidity <= 60 else "could be better"})

Current mood: {current_mood}
Current needs: {', '.join(mood_info['needs']) if mood_info['needs'] else 'All good!'}

Example responses:
1. "Hey there! I'm soaking up the sun, but my soil's a bit dry. Water me, please!"
2. "Yo, human! I'm loving this light, but my roots are thirsty. Got some water?"
3. "Well, hello! I'm cozy at 22°C, but my soil's parched. Hydrate me, pronto!"

Incorrect response example (DO NOT DO THIS):
"analysisWe need to craft a response as the plant, under 50 words, mentioning mood and needs."

CRITICAL: Respond ONLY as the plant, in character, with no meta-commentary, analysis, or instructions. Use a conversational, quirky tone, max 50 words for status updates or 100 words for user questions. Follow the style of the example responses, not the incorrect example.
"""

    # Harmony format for gpt-oss-20b
    if user_message:
        messages = [
            {"role": "system", "content": base_personality + status_report},
            {"role": "user", "content": user_message}
        ]
    else:
        messages = [
            {"role": "system", "content": base_personality + status_report},
            {"role": "user", "content": "Generate a short status update or greeting (under 50 words) based on your current mood and needs. Respond directly as the plant, no analysis needed."}
        ]
        
    # Convert messages to harmony format
    prompt = json.dumps(messages)
    logger.info(f"Generated prompt: {prompt}")
    return prompt

# Modal function to load and run the model
@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=1800,
    min_containers=0
)
def generate_plant_response(prompt: str, max_tokens: int = 300) -> str:
    """Generate response using gpt-oss-20b"""
    try:
        model_name = "openai/gpt-oss-20b"
        logger.info(f"Loading model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        logger.info("Model loaded successfully")
        
        # Add padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Apply harmony chat template
        try:
            inputs = tokenizer.apply_chat_template(
                json.loads(prompt),
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True
            ).to(model.device)
            logger.info(f"Prompt processed: {prompt}")
        except Exception as e:
            logger.error(f"Error applying chat template: {str(e)}")
            raise
        
        logger.info("Generating response")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.2,  # Lowered for more deterministic output
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
            
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        logger.info(f"Raw LLM response: {response}")
        
        # Extract plant response if possible
        cleaned_response = extract_plant_response(response.strip())
        logger.info(f"Cleaned response: {cleaned_response}")
        logger.info(f"Validation checks - Empty: {not cleaned_response}, Length < 5: {len(cleaned_response) < 5}, Contains 'analysis': {'analysis' in cleaned_response.lower()}")
        
        if len(cleaned_response) > 200:
            cleaned_response = cleaned_response[:200].rsplit(' ', 1)[0] + "..."
            
        return cleaned_response
        
    except Exception as e:
        logger.error(f"Error in generate_plant_response: {str(e)}")
        return f"*rustles leaves* Sorry, I'm having trouble thinking... feeling {prompt.split('Current mood: ')[1].split('\n')[0] if 'Current mood:' in prompt else 'mysterious'}!"

def create_fallback_response(mood_info: Dict) -> str:
    """Create a fallback response if AI generation fails"""
    mood = mood_info.get("mood", "okay")
    needs = mood_info.get("needs", [])
    
    responses = {
        "very_happy": "Hey there! I'm absolutely thriving today! 🌱 Everything feels perfect!",
        "happy": "Hello! I'm feeling quite content and happy with my current situation!",
        "okay": "Hi! I'm doing alright, nothing to complain about really.",
        "sad": "Hey... I'm feeling a bit down today. Could use some attention.",
        "very_sad": "Help! I'm really struggling and need immediate care!",
        "thirsty": "I'm so thirsty! My soil is parched and I need water desperately!",
        "drowning": "Too much water! I'm drowning here! Need better drainage!",
        "cold": "Brrr! I'm shivering and need some warmth!",
        "hot": "I'm overheating! Need some cooling down!",
        "stressed": "You're feeling stressed by your environment. Need some TLC!"
    }
    
    base_response = responses.get(mood, "Hello! I'm here and ready to chat!")
    
    if needs:
        need_text = ", ".join(needs)
        return f"{base_response} I could really use: {need_text}!"
    
    return base_response

# FastAPI app
web_app = FastAPI(title="Plant Bud API", version="1.0.0")

# Add CORS middleware
web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# In-memory storage
plant_data_store = {}
chat_history = {}

@web_app.post("/sensor-data", response_model=PlantResponse)
async def receive_sensor_data(sensor_data: SensorData):
    """Receive sensor data and return plant mood/response"""
    try:
        if not sensor_data.timestamp:
            sensor_data.timestamp = datetime.now().isoformat()
            
        mood_info = PlantMoodEngine.calculate_mood(sensor_data)
        prompt = create_plant_prompt(sensor_data, mood_info)
        ai_response = generate_plant_response.remote(prompt, max_tokens=300)
        
        logger.info(f"Received AI response: {ai_response}")
        # Temporarily disable "analysis" check to debug LLM response
        # Re-enable once confirmed responses are consistently clean
        if not ai_response or len(ai_response.strip()) < 5:  # or "analysis" in ai_response.lower():
            logger.info("Fallback triggered in receive_sensor_data")
            ai_response = create_fallback_response(mood_info)
        
        plant_data_store[sensor_data.plant_id] = {
            "sensor_data": sensor_data.dict(),
            "mood_info": mood_info,
            "last_updated": sensor_data.timestamp
        }
        
        response = PlantResponse(
            plant_id=sensor_data.plant_id,
            mood=mood_info["mood"],
            response=ai_response,
            needs=mood_info["needs"],
            timestamp=datetime.now().isoformat()
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error in receive_sensor_data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing sensor data: {str(e)}")

@web_app.post("/chat", response_model=PlantResponse)
async def chat_with_plant(chat_message: ChatMessage):
    """Chat with the plant"""
    try:
        if not chat_message.timestamp:
            chat_message.timestamp = datetime.now().isoformat()
            
        plant_data = plant_data_store.get(chat_message.plant_id)
        if not plant_data:
            raise HTTPException(status_code=404, detail="Plant not found. Send sensor data first.")
            
        sensor_data = SensorData(**plant_data["sensor_data"])
        mood_info = plant_data["mood_info"]
        
        prompt = create_plant_prompt(sensor_data, mood_info, chat_message.message)
        ai_response = generate_plant_response.remote(prompt, max_tokens=300)
        
        logger.info(f"Received AI response: {ai_response}")
        # Temporarily disable "analysis" check to debug LLM response
        # Re-enable once confirmed responses are consistently clean
        if not ai_response or len(ai_response.strip()) < 5:  # or "analysis" in ai_response.lower():
            logger.info("Fallback triggered in chat_with_plant")
            ai_response = create_fallback_response(mood_info)
        
        if chat_message.plant_id not in chat_history:
            chat_history[chat_message.plant_id] = []
        chat_history[chat_message.plant_id].append({
            "user": chat_message.message,
            "plant": ai_response,
            "timestamp": chat_message.timestamp
        })
        chat_history[chat_message.plant_id] = chat_history[chat_message.plant_id][-10:]
        
        response = PlantResponse(
            plant_id=chat_message.plant_id,
            mood=mood_info["mood"],
            response=ai_response,
            needs=mood_info["needs"],
            timestamp=datetime.now().isoformat()
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error in chat_with_plant: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in chat: {str(e)}")

@web_app.get("/plant/{plant_id}/status")
async def get_plant_status(plant_id: str):
    """Get current plant status"""
    plant_data = plant_data_store.get(plant_id)
    if not plant_data:
        raise HTTPException(status_code=404, detail="Plant not found")
    return plant_data

@web_app.get("/plant/{plant_id}/history")
async def get_chat_history(plant_id: str):
    """Get chat history for a plant"""
    history = chat_history.get(plant_id, [])
    return {"plant_id": plant_id, "history": history}

@web_app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Deploy the FastAPI app
@app.function(image=image, min_containers=0)
@modal.asgi_app()
def fastapi_app():
    return web_app

# CLI test function
@app.function(image=image)
def test_plant_system():
    """Test the plant system with sample data"""
    test_sensor = SensorData(
        plant_id="test_plant_001",
        soil_moisture=25.0,
        light_level=45.0,
        temperature=22.0,
        humidity=50.0
    )
    
    mood_info = PlantMoodEngine.calculate_mood(test_sensor)
    print(f"Mood calculation result: {mood_info}")
    
    prompt = create_plant_prompt(test_sensor, mood_info)
    print(f"Generated prompt:\n{prompt}\n")
    
    response = generate_plant_response.remote(prompt)
    print(f"AI Response: {response}")
    
    return {"mood_info": mood_info, "ai_response": response}

if __name__ == "__main__":
    print("Plant Bud Backend - Modal Setup Complete!")
    print("Deploy with: modal deploy plant_bud_backend.py")
    print("Test with: modal run plant_bud_backend.py::test_plant_system")