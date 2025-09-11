import requests
import time
import json
from datetime import datetime

# Configuration
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1415490142403297472/k-Qqg6X-aesyScdvmC6X1R-nF0_rAKOHK2gdLXOMEgpf5eh25usCZj9wdnMYEGIBggBS"
PLANT_API_URL = "https://uta2025hackathon--plant-backend-fastapi-app.modal.run"
PLANT_ID = "my_plant_001"
POLL_INTERVAL = 10  # seconds

# Global variables
last_mood = None
check_count = 0

def get_plant_status():
    """Get current plant status from API"""
    try:
        response = requests.get(f"{PLANT_API_URL}/plant/{PLANT_ID}/status", timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"   ❌ API Error: {response.status_code}")
            return None
    except Exception as e:
        print(f"   ❌ Connection Error: {e}")
        return None

def get_mood_color(mood):
    """Get Discord color for plant mood"""
    colors = {
        'very_happy': 0x4CAF50,    # Green
        'happy': 0x8BC34A,         # Light Green
        'content': 0xFFEB3B,       # Yellow
        'stressed': 0xFF9800,      # Orange
        'unhappy': 0xF44336,       # Red
        'sad': 0xF44336,           # Red
        'critical': 0x9C27B0,      # Purple
    }
    return colors.get(mood, 0x607D8B)  # Default blue-grey

def send_discord_message(plant_data, is_mood_change=False):
    """Send plant status to Discord"""
    global check_count
    
    try:
        if not plant_data:
            # Connection error message
            embed = {
                "title": "🚨 Plant Connection Lost",
                "description": "Cannot reach your plant! Check ESP32 and WiFi connection.",
                "color": 0xF44336,  # Red
                "footer": {"text": f"Check #{check_count} | Auto Monitor"},
                "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S.000Z")
            }
        else:
            # Normal plant status
            sensor_data = plant_data.get('sensor_data', {})
            mood_info = plant_data.get('mood_info', {})
            
            mood = mood_info.get('mood', 'unknown')
            needs = mood_info.get('needs', [])
            plant_message = plant_data.get('last_response', 'No recent message')
            
            # Create title based on mood change
            if is_mood_change:
                title = f"📊 Plant Mood Changed: {mood.replace('_', ' ').title()}"
            else:
                title = f"🌱 Plant Status: {mood.replace('_', ' ').title()}"
            
            embed = {
                "title": title,
                "description": f"*\"{plant_message}\"*",
                "color": get_mood_color(mood),
                "fields": [
                    {
                        "name": "💧 Soil Moisture",
                        "value": f"{sensor_data.get('soil_moisture', 0):.1f}%",
                        "inline": True
                    },
                    {
                        "name": "🌡️ Temperature",
                        "value": f"{sensor_data.get('temperature', 0):.1f}°C",
                        "inline": True
                    },
                    {
                        "name": "💨 Humidity",
                        "value": f"{sensor_data.get('humidity', 0):.1f}%",
                        "inline": True
                    },
                    {
                        "name": "📋 Status",
                        "value": "✅ All Good!" if not needs else f"⚠️ Needs: {', '.join(needs).title()}",
                        "inline": False
                    }
                ],
                "footer": {"text": f"Check #{check_count} | Auto Monitor"},
                "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S.000Z")
            }
        
        payload = {"embeds": [embed]}
        response = requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=10)
        
        if response.status_code == 204:
            print(f"   ✅ Discord notification sent!")
            return True
        else:
            print(f"   ❌ Discord failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ Discord error: {e}")
        return False

def should_send_notification(current_mood):
    """Determine if we should send a Discord notification"""
    global last_mood
    
    # Always send if mood changed
    if last_mood != current_mood:
        return True, True  # Send notification, is mood change
    
    # Send every 5th check (every 50 seconds) for regular updates
    if check_count % 5 == 0:
        return True, False  # Send notification, not mood change
    
    # Always send for critical conditions
    if current_mood in ['critical', 'unhappy', 'sad']:
        return True, False  # Send notification, not mood change
    
    return False, False

def monitor_plant():
    """Main monitoring loop"""
    global last_mood, check_count
    
    print("🤖 Discord Plant Monitor Started")
    print("=" * 40)
    print(f"📡 Polling every {POLL_INTERVAL} seconds")
    print("📱 Sending to Discord on mood changes and regular intervals")
    print("💡 Press Ctrl+C to stop")
    print("=" * 40)
    
    try:
        while True:
            check_count += 1
            current_time = datetime.now().strftime("%H:%M:%S")
            
            print(f"\n🔍 [{current_time}] Check #{check_count} - Getting plant status...")
            
            # Get plant data
            plant_data = get_plant_status()
            
            if plant_data:
                mood = plant_data.get('mood_info', {}).get('mood', 'unknown')
                soil = plant_data.get('sensor_data', {}).get('soil_moisture', 0)
                temp = plant_data.get('sensor_data', {}).get('temperature', 0)
                
                print(f"   🎭 Mood: {mood} | 💧 Soil: {soil:.1f}% | 🌡️ Temp: {temp:.1f}°C")
                
                # Check if we should send notification
                should_send, is_mood_change = should_send_notification(mood)
                
                if should_send:
                    print(f"   📱 Sending Discord notification...")
                    send_discord_message(plant_data, is_mood_change)
                    
                    if is_mood_change:
                        print(f"   🔄 Mood changed: {last_mood} → {mood}")
                    
                    last_mood = mood
                else:
                    print(f"   ⏭️ Skipping notification (no change)")
            else:
                print(f"   ❌ No plant data received")
                # Send error notification occasionally
                if check_count % 3 == 0:  # Every 30 seconds for errors
                    print(f"   📱 Sending error notification...")
                    send_discord_message(None)
            
            print(f"   ⏳ Waiting {POLL_INTERVAL} seconds...")
            time.sleep(POLL_INTERVAL)
            
    except KeyboardInterrupt:
        print(f"\n\n⏹️ Monitor stopped after {check_count} checks")
        print("👋 Discord plant monitor shut down gracefully!")

if __name__ == "__main__":
    monitor_plant()
