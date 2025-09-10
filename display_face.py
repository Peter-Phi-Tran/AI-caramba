from PIL import Image
import pygame
import requests
import sys
import time
import threading

# ----------------------------
# CONFIGURATION
# ----------------------------
BASE_URL = "https://uta2025hackathon--plant-backend-fastapi-app.modal.run"
PLANT_ID = "my_plant_001"
POLL_INTERVAL = 5  # seconds between API updates

mood_files = {
    "very_happy": "assets/plant_very_happy.gif",
    "happy": "assets/plant_happy.gif",
    "okay": "assets/plant_okay.gif",
    "sad": "assets/plant_sad.gif",
}

# Global variables
current_mood = "happy"
ai_response = "Loading plant status..."
frames = []
frame_index = 0
running = True

# ----------------------------
# API POLLING FUNCTION
# ----------------------------
def fetch_plant_status():
    global current_mood, ai_response, running
    while running:
        try:
            response = requests.get(f"{BASE_URL}/plant/{PLANT_ID}/status", timeout=5)
            if response.status_code == 200:
                data = response.json()

                # Extract mood and AI response safely
                mood_info = data.get("mood_info", {})
                current_mood = mood_info.get("mood", "okay")

                ai_response = data.get("last_response", "No response from plant yet.")
                print(f"[API] Mood: {current_mood}, Response: {ai_response}")
            else:
                print(f"[API] Error: {response.status_code} - {response.text}")
                ai_response = "Error retrieving plant status."
        except requests.exceptions.Timeout:
            print("[API] Request timed out.")
            ai_response = "Connection timeout."
        except requests.exceptions.RequestException as e:
            print(f"[API] Request error: {e}")
            ai_response = "Error connecting to server."

        time.sleep(POLL_INTERVAL)

# ----------------------------
# LOAD GIF FRAMES
# ----------------------------
def load_gif_frames(path):
    gif = Image.open(path)
    frames_list = []
    try:
        while True:
            frame = gif.convert("RGBA")
            pygame_image = pygame.image.fromstring(frame.tobytes(), frame.size, frame.mode)
            frames_list.append(pygame_image)
            gif.seek(gif.tell() + 1)
    except EOFError:
        pass  # End of GIF
    return frames_list

# ----------------------------
# PYGAME INITIALIZATION
# ----------------------------
pygame.init()
pygame.mouse.set_visible(False)  # Hide mouse cursor

# Fullscreen mode
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
SCREEN_WIDTH, SCREEN_HEIGHT = screen.get_size()

pygame.display.set_caption("Plant Mood Display")
clock = pygame.time.Clock()

# Font for AI response
font = pygame.font.Font(None, 32)  # Default pygame font, size 32

# ----------------------------
# THREAD TO FETCH API DATA
# ----------------------------
api_thread = threading.Thread(target=fetch_plant_status, daemon=True)
api_thread.start()

# ----------------------------
# MAIN LOOP
# ----------------------------
while running:
    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                running = False

    # Load frames for the current mood dynamically
    gif_path = mood_files.get(current_mood, mood_files["okay"])
    if not frames or gif_path != mood_files.get(current_mood):
        frames = load_gif_frames(gif_path)
        frame_index = 0

    # Draw background
    screen.fill((0, 0, 0))

    # Draw GIF centered
    if frames:
        frame = frames[frame_index]
        scaled_frame = pygame.transform.scale(frame, (128, 128))  # Scale to fixed size
        gif_x = (SCREEN_WIDTH - 128) // 2
        gif_y = (SCREEN_HEIGHT // 3) - 64  # Position near top third of screen
        screen.blit(scaled_frame, (gif_x, gif_y))

        # Cycle through frames
        frame_index = (frame_index + 1) % len(frames)

    # Draw AI response text, centered below GIF
    max_width = SCREEN_WIDTH - 40
    words = ai_response.split(" ")
    lines = []
    current_line = ""

    for word in words:
        test_line = current_line + word + " "
        if font.size(test_line)[0] < max_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word + " "
    lines.append(current_line)

    y_offset = gif_y + 150  # Start drawing text below the GIF
    for line in lines:
        text_surface = font.render(line.strip(), True, (255, 255, 255))
        text_width = text_surface.get_width()
        screen.blit(text_surface, ((SCREEN_WIDTH - text_width) // 2, y_offset))
        y_offset += 40

    # Update display
    pygame.display.flip()
    clock.tick(10)  # Limit frame rate to 10 FPS

# ----------------------------
# CLEAN EXIT
# ----------------------------
pygame.quit()
sys.exit()