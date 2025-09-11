from PIL import Image
import pygame
import requests
import sys
import time
import threading
import os

# --- Configuration ---
BASE_URL = "https://uta2025hackathon--plant-backend-fastapi-app.modal.run"
PLANT_ID = "my_plant_001"

mood_files = {
    "very_happy": "/home/openai/Downloads/assets/plant_very_happy.gif",
    "happy": "/home/openai/Downloads/assets/plant_happy.gif",
    "okay": "/home/openai/Downloads/assets/plant_okay.gif",
    "sad": "/home/openai/Downloads/assets/plant_sad.gif"
}

SCREEN_WIDTH, SCREEN_HEIGHT = 320, 240
FPS = 10  # Adjust frame speed
TARGET_FRAME_COUNT = 47  # Optional: pad all GIFs to match this frame count

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
    if not os.path.exists(gif_path):
        print(f"Error: GIF file not found: {gif_path}")
        return []

    try:
        gif = Image.open(gif_path)
        gif_frames = []
        while True:
            frame = gif.convert("RGBA")
            pygame_image = pygame.image.fromstring(frame.tobytes(), frame.size, frame.mode)
            gif_frames.append(pygame_image)
            gif.seek(gif.tell() + 1)
    except EOFError:
        pass
    except Exception as e:
        print(f"Error loading GIF {gif_path}: {e}")
        return []

    # Optional: pad to target frame count by repeating last frame
    if gif_frames and len(gif_frames) < TARGET_FRAME_COUNT:
        last_frame = gif_frames[-1]
        while len(gif_frames) < TARGET_FRAME_COUNT:
            gif_frames.append(last_frame.copy())

    return gif_frames

# Load initial mood frames
frames = load_gif_frames(mood_files[current_mood])
print(f"Loaded {len(frames)} frames for {current_mood}")

# --- Background thread to fetch API data ---
def update_mood_from_api():
    global current_mood, frames, frame_index, ai_response
    while True:
        try:
            print("Fetching data")
            response = requests.get(f"{BASE_URL}/plant/{PLANT_ID}/status", timeout=5)
            if response.status_code == 200:
                data = response.json()
                mood = data.get("mood_info", {}).get("mood", "okay")
                ai_response = data.get("last_response", "No response available")

                # Update mood if changed
                print(mood)
                if mood != current_mood:
                    print(f"Updating mood from {current_mood} to {mood}")
                    new_frames = load_gif_frames(mood_files.get(mood, mood_files["okay"]))
                    if new_frames:
                        current_mood = mood
                        frames[:] = new_frames
                        frame_index = 0  # âœ… Reset the frame index
                        print(f"Mood updated to {current_mood} with {len(frames)} frames")
                    else:
                        print(f"Warning: Failed to load frames for mood: {mood}")
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
    if frames and 0 <= frame_index < len(frames):
        frame = frames[frame_index]
        scaled_frame = pygame.transform.scale(frame, (128, 128))
        screen.blit(scaled_frame, ((SCREEN_WIDTH - 128) // 2, 20))
    else:
        error_text = font.render("No frames to display", True, (255, 0, 0))
        screen.blit(error_text, (10, 20))

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
        text_surface = font.render(line.strip(), True, (255, 255, 255))
        screen.blit(text_surface, (10, y_offset))
        y_offset += 18

    pygame.display.flip()

    if frames:
        frame_index = (frame_index + 1) % len(frames)

    clock.tick(FPS)

pygame.quit()
sys.exit()