from PIL import Image
import pygame
import sys

mood_files = {
    "very_happy": "assets/plant_very_happy.gif",
    "happy": "assets/plant_happy.gif",
    "okay": "assets/plant_okay.gif",
    "sad": "assets/plant_sad.gif"
}

current_mood = "very_happy"
GIF_PATH = mood_files[current_mood]
SCREEN_WIDTH, SCREEN_HEIGHT = 320, 240
FPS = 10  # Set to your GIF's frame speed

# Initialize pygame
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Plant Mood Display")

# Load GIF and extract frames
gif = Image.open(GIF_PATH)
frames = []

try:
    while True:
        frame = gif.convert("RGBA")
        pygame_image = pygame.image.fromstring(frame.tobytes(), frame.size, frame.mode)
        frames.append(pygame_image)
        gif.seek(gif.tell() + 1)
except EOFError:
    pass  # End of GIF

print(f"Loaded {len(frames)} frames from GIF.")

# Main loop
running = True
frame_index = 0
clock = pygame.time.Clock()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Draw frame
    screen.fill((0, 0, 0))
    frame = frames[frame_index]
    
    # Optional scaling
    scaled_frame = pygame.transform.scale(frame, (128, 128))
    screen.blit(scaled_frame, ((SCREEN_WIDTH - 128) // 2, (SCREEN_HEIGHT - 128) // 2))
    
    pygame.display.flip()

    # Advance frame
    frame_index = (frame_index + 1) % len(frames)
    clock.tick(FPS)

pygame.quit()
sys.exit()