from PIL import Image
import pygame
import requests
import sys
import time
import threading

# ========================
# CONFIGURATION
# ========================
BASE_URL = "https://uta2025hackathon--plant-backend-fastapi-app.modal.run"
PLANT_ID = "my_plant_001"

# Map moods to GIF files
mood_files = {
    "very_happy": "assets/plant_very_happy.gif",
    "happy": "assets/plant_happy.gif",
    "okay": "assets/plant_okay.gif",
    "sad": "assets/plant_sad.gif"
}

SCREEN_WIDTH, SCREEN_HEIGHT = 320, 240
FPS = 10  # Speed of animation
MOOD_UPDATE_INTERVAL = 15  # seconds between API calls

# ========================
# GLOBAL STATE
# ========================
current_mood = "okay"  # Default mood
latest_mood = None
mood_lock = threading.Lock()

# ========================
# FUNCTIONS
# ========================

def get_plant_mood():
    """Fetch the latest mood from the API with timeout handling."""
    try:
        response = requests.get(
            f"{BASE_URL}/plant/{PLANT_ID}/status",
            timeout=5  # fail fast if API is slow
        )
        response.raise_for_status()
        data = response.json()
        mood = data.get("mood_info", {}).get("mood", "okay") # fallback if 'mood' key missing
        return mood 
    except requests.RequestException as e:
        print(f"API call failed: {e}")
        return None


def fetch_mood_background():
    """Runs in a separate thread so API calls don't freeze the game loop."""
    global latest_mood
    new_mood = get_plant_mood()
    if new_mood:
        with mood_lock:
            latest_mood = new_mood


def load_gif_frames(path):
    """Load all frames of a GIF as Pygame images."""
    gif = Image.open(path)
    frames = []

    try:
        while True:
            frame = gif.convert("RGBA")
            pygame_image = pygame.image.fromstring(frame.tobytes(), frame.size, frame.mode)
            frames.append(pygame_image)
            gif.seek(gif.tell() + 1)
    except EOFError:
        pass  # Reached the end of the GIF

    print(f"Loaded {len(frames)} frames from {path}")
    return frames


# ========================
# INITIALIZATION
# ========================
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Plant Mood Display")

# Load initial mood frames
frames = load_gif_frames(mood_files[current_mood])

frame_index = 0
clock = pygame.time.Clock()
running = True

# Timing for mood updates
last_mood_check = 0

# ========================
# MAIN LOOP
# ========================
while running:
    # Handle quit event
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Periodically fetch mood in background
    if time.time() - last_mood_check >= MOOD_UPDATE_INTERVAL:
        threading.Thread(target=fetch_mood_background, daemon=True).start()
        last_mood_check = time.time()

    # Check if background thread updated the mood
    with mood_lock:
        if latest_mood and latest_mood != current_mood:
            print(f"Mood changed: {current_mood} -> {latest_mood}")
            current_mood = latest_mood
            frames = load_gif_frames(mood_files[current_mood])
            frame_index = 0

    # Draw current frame
    screen.fill((0, 0, 0))
    frame = frames[frame_index]

    # Optional scaling to fit display
    scaled_frame = pygame.transform.scale(frame, (128, 128))
    screen.blit(scaled_frame, ((SCREEN_WIDTH - 128) // 2, (SCREEN_HEIGHT - 128) // 2))

    pygame.display.flip()

    # Advance GIF frame
    frame_index = (frame_index + 1) % len(frames)
    clock.tick(FPS)

pygame.quit()
sys.exit()