from PIL import Image
import pygame
import requests
import sys
import time
import threading

# --- Configuration ---
BASE_URL = "https://uta2025hackathon--plant-backend-fastapi-app.modal.run"
PLANT_ID = "my_plant_001"

mood_files = {
    "very_happy": "assets/plant_very_happy.gif",
    "happy": "assets/plant_happy.gif",
    "okay": "assets/plant_okay.gif",
    "sad": "assets/plant_sad.gif"
}

SCREEN_WIDTH, SCREEN_HEIGHT = 320, 240
FPS = 10  # Adjust frame speed

# --- Initialize Pygame ---
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Plant Mood Display")

# Font for text
pygame.font.init()
font = pygame.font.SysFont("Arial", 16)

# Global state
current_mood = "very_happy"
ai_response = "Loading plant response..."
frames = []
frame_index = 0
clock = pygame.time.Clock()

# --- Function to load GIF frames ---
def load_gif_frames(gif_path):
    gif = Image.open(gif_path)
    gif_frames = []
    try:
        while True:
            frame = gif.convert("RGBA")
            pygame_image = pygame.image.fromstring(frame.tobytes(), frame.size, frame.mode)
            gif_frames.append(pygame_image)
            gif.seek(gif.tell() + 1)
    except EOFError:
        pass
    return gif_frames

# Load initial mood frames
frames = load_gif_frames(mood_files[current_mood])
print(f"Loaded {len(frames)} frames for {current_mood}")

# --- Background thread to fetch API data ---
def update_mood_from_api():
    global current_mood, frames, ai_response
    while True:
        try:
            response = requests.get(f"{BASE_URL}/plant/{PLANT_ID}/status", timeout=5)
            if response.status_code == 200:
                data = response.json()
                mood = data.get("mood_info", {}).get("mood", "okay")
                ai_response = data.get("last_response", "No response available")

                # Update mood if changed
                if mood != current_mood:
                    current_mood = mood
                    frames[:] = load_gif_frames(mood_files.get(current_mood, mood_files["okay"]))
                    print(f"Mood updated to {current_mood}")
            else:
                ai_response = "Error: Failed to fetch plant data"
        except requests.Timeout:
            ai_response = "Error: API request timed out"
        except Exception as e:
            ai_response = f"Error: {e}"

        time.sleep(10)  # Update every 10 seconds

# Start the thread
threading.Thread(target=update_mood_from_api, daemon=True).start()

# --- Main loop ---
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((0, 0, 0))

    # Draw GIF frame
    frame = frames[frame_index]
    scaled_frame = pygame.transform.scale(frame, (128, 128))
    screen.blit(scaled_frame, ((SCREEN_WIDTH - 128) // 2, 20))

    # Draw AI response text
    lines = []
    words = ai_response.split(" ")
    current_line = ""

    # Wrap text so it fits screen width
    for word in words:
        if font.size(current_line + word)[0] < SCREEN_WIDTH - 20:
            current_line += word + " "
        else:
            lines.append(current_line)
            current_line = word + " "
    lines.append(current_line)

    y_offset = 160  # Start drawing text below GIF
    for line in lines:
        text_surface = font.render(line, True, (255, 255, 255))
        screen.blit(text_surface, (10, y_offset))
        y_offset += 18

    pygame.display.flip()

    frame_index = (frame_index + 1) % len(frames)
    clock.tick(FPS)

pygame.quit()
sys.exit()