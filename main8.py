import pygame
import sys
import math
import random
import noise
import numpy as np

# Configuration
SCREEN_WIDTH = 1500
SCREEN_HEIGHT = 1000
OVERWRITE_SCREEN_DIMENSIONS = True
NUM_BOIDS = 100

# Boid behavior parameters
START_SPEED = 2.25
MAX_SPEED = 5
PERCEPTION_RADIUS = 30
BOUNDARY_PERCEPTION = PERCEPTION_RADIUS
SEPARATION_RADIUS = 40
COHESION_WEIGHT = 0.4
ALIGNMENT_WEIGHT = 0.2
SEPARATION_WEIGHT = 3
RANDOMNESS_WEIGHT = 0.225
MASS_EQUIVALENT = 2      # (was 1.5)
TURN_FACTOR = 0.0325
SPEED_ADJUSTMENT_FACTOR = 0.3

# Attractors
ATTRACTOR_FORCE = 10
ATTRACTOR_FORCE_MAX = 20
ATTRACTOR_FORCE_DISTANCE_EXP = -1
ATTRACTOR_INIT = True
ATTRACTOR_INIT_NUM = 10
ATTRACTOR_INIT_SEED = 42
ATTRACTOR_INIT_MARGIN = 100

# Color and drawing
ADD_TRAILS = True
# TRAIL_ALPHA now used in a multiply blend (values near 255 mean little fading)
TRAIL_ALPHA = 220       # 240/255 ≈ 0.94 multiplier (i.e. 6% fade per frame)
PROXIMITY_NORMALIZATION = 5
ACCELERATION_UPPER_EXPECTED = MAX_SPEED / 10
BACKGROUND_COLOR = (0, 0, 0)
BASE_COLOR = (50, 50, 50)
ATTRACTOR_COLOR = (50, 50, 0)
ATTRACTOR_COLOR_FACTOR = 2

# Overwrite screen dimensions using current monitor info, if enabled.
pygame.init()
if OVERWRITE_SCREEN_DIMENSIONS:
    info = pygame.display.Info()
    SCREEN_WIDTH = info.current_w
    SCREEN_HEIGHT = int(info.current_h * 0.95)  # leave some room for taskbar

def clamp_color(color):
    return max(0, min(255, color))

def clamp_colors(r, g, b):
    return (clamp_color(r), clamp_color(g), clamp_color(b))

def get_color(base_value, factor):
    return base_value + int((255 - base_value) * factor)

def initialize_attractors():
    assert ATTRACTOR_INIT == True
    random.seed(ATTRACTOR_INIT_SEED)
    attractors = [
        pygame.Vector2(
            random.uniform(ATTRACTOR_INIT_MARGIN, SCREEN_WIDTH - ATTRACTOR_INIT_MARGIN),
            random.uniform(ATTRACTOR_INIT_MARGIN, SCREEN_HEIGHT - ATTRACTOR_INIT_MARGIN)
        )
        for _ in range(ATTRACTOR_INIT_NUM)
    ]
    return attractors

# Compute background at a lower resolution for speed.
BG_SCALE = 0.25
def compute_background_surface(attractors):
    """
    Computes a background surface whose brightness is modulated by the net attractor force.
    The computation is done at low resolution (scaled by BG_SCALE) and then upscaled.
    """
    bg_width = int(SCREEN_WIDTH * BG_SCALE)
    bg_height = int(SCREEN_HEIGHT * BG_SCALE)
    xs = np.linspace(0, SCREEN_WIDTH, bg_width, dtype=np.float32)
    ys = np.linspace(0, SCREEN_HEIGHT, bg_height, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)  # shape: (bg_height, bg_width)

    net_fx = np.zeros((bg_height, bg_width), dtype=np.float32)
    net_fy = np.zeros((bg_height, bg_width), dtype=np.float32)

    for attractor in attractors:
        ax = attractor.x
        ay = attractor.y
        dx = ax - X
        dy = ay - Y
        dist = np.sqrt(dx**2 + dy**2)
        dist = np.maximum(dist, 1e-5)  # avoid division by zero
        force = ATTRACTOR_FORCE * (dist ** ATTRACTOR_FORCE_DISTANCE_EXP)
        force = np.minimum(force, ATTRACTOR_FORCE_MAX)
        nx = dx / dist
        ny = dy / dist
        net_fx += nx * force
        net_fy += ny * force

    net_force = np.sqrt(net_fx**2 + net_fy**2)
    # Map net_force to brightness: when net_force == ATTRACTOR_FORCE_MAX, brightness ≈ 255.
    brightness = net_force / ATTRACTOR_FORCE_MAX * ATTRACTOR_COLOR_FACTOR * 255.0
    brightness = np.clip(brightness, 0, 255).astype(np.uint8)
    rgb = np.stack([brightness, brightness, brightness], axis=-1)
    lowres_surface = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
    background_surface = pygame.transform.smoothscale(lowres_surface, (SCREEN_WIDTH, SCREEN_HEIGHT))
    return background_surface

class Boid:
    def __init__(self):
        self.position = pygame.Vector2(random.uniform(0, SCREEN_WIDTH),
                                       random.uniform(0, SCREEN_HEIGHT))
        angle = random.uniform(0, 2 * math.pi)
        self.velocity = pygame.Vector2(math.cos(angle), math.sin(angle)) * START_SPEED
        self.acceleration = pygame.Vector2(0, 0)
        self.color = (0, 0, 0)
        # Individual variation in behavior:
        self.cohesion_weight = COHESION_WEIGHT * random.uniform(0.8, 1.2)
        self.alignment_weight = ALIGNMENT_WEIGHT * random.uniform(0.8, 1.2)
        self.separation_weight = SEPARATION_WEIGHT * random.uniform(0.8, 1.2)
        # Noise offsets:
        self.noise_offset_x = random.random() * 1000
        self.noise_offset_y = random.random() * 1000

    def update(self, boids, attractors):
        self.acceleration = pygame.Vector2(0, 0)
        self.apply_rules(boids)
        self.avoid_edges()
        self.add_noise()
        self.apply_attractors(attractors)
        # Update velocity with inertia:
        self.velocity += self.acceleration / MASS_EQUIVALENT
        speed = self.velocity.length()
        if speed > MAX_SPEED:
            self.velocity = self.velocity.normalize() * MAX_SPEED
        if speed != 0:
            new_speed = speed + (START_SPEED - speed) * SPEED_ADJUSTMENT_FACTOR
            self.velocity = self.velocity.normalize() * new_speed
        self.position += self.velocity
        self.update_color(boids)
        # Slight damping for smoother motion.
        self.velocity = 0.95 * self.velocity

    def apply_rules(self, boids):
        separation = pygame.Vector2(0, 0)
        alignment = pygame.Vector2(0, 0)
        cohesion = pygame.Vector2(0, 0)
        total = 0
        for boid in boids:
            if boid == self:
                continue
            distance = self.position.distance_to(boid.position)
            if distance < PERCEPTION_RADIUS:
                if distance < SEPARATION_RADIUS:
                    diff = self.position - boid.position
                    separation += diff.normalize() / distance
                alignment += boid.velocity
                cohesion += boid.position
                total += 1
        if total > 0:
            self.acceleration += separation * self.separation_weight
            self.acceleration += (alignment / total) * self.alignment_weight
            self.acceleration += ((cohesion / total) - self.position).normalize() * self.cohesion_weight

    def avoid_edges(self):
        margin = BOUNDARY_PERCEPTION
        force = pygame.Vector2(0, 0)
        if self.position.x < margin:
            force.x = TURN_FACTOR * (margin - self.position.x)
        elif self.position.x > SCREEN_WIDTH - margin:
            force.x = -TURN_FACTOR * (self.position.x - (SCREEN_WIDTH - margin))
        if self.position.y < margin:
            force.y = TURN_FACTOR * (margin - self.position.y)
        elif self.position.y > SCREEN_HEIGHT - margin:
            force.y = -TURN_FACTOR * (self.position.y - (SCREEN_HEIGHT - margin))
        self.acceleration += force

    def add_noise(self):
        nx = noise.pnoise1(self.noise_offset_x)
        ny = noise.pnoise1(self.noise_offset_y)
        self.acceleration += pygame.Vector2(nx, ny) * RANDOMNESS_WEIGHT
        self.noise_offset_x += 0.01
        self.noise_offset_y += 0.01

    def apply_attractors(self, attractors):
        for attractor in attractors:
            direction = attractor - self.position
            if direction.length() != 0:
                acceleration = ATTRACTOR_FORCE * (direction.length() ** ATTRACTOR_FORCE_DISTANCE_EXP) / MASS_EQUIVALENT
                acceleration = min(acceleration, ATTRACTOR_FORCE_MAX)
                self.acceleration += direction.normalize() * acceleration

    def update_color(self, boids):
        proximity = 0
        for boid in boids:
            if boid == self:
                continue
            distance = self.position.distance_to(boid.position)
            if distance < PERCEPTION_RADIUS:
                proximity += PERCEPTION_RADIUS - distance
        proximity_factor = proximity / PERCEPTION_RADIUS / PROXIMITY_NORMALIZATION
        acceleration_factor = self.acceleration.length() / ACCELERATION_UPPER_EXPECTED / MAX_SPEED * 3
        acceleration_factor = acceleration_factor ** 3
        self.color = clamp_colors(
            get_color(BASE_COLOR[0], acceleration_factor),
            BASE_COLOR[1],
            get_color(BASE_COLOR[2], proximity_factor)
        )

    def draw(self, surface):
        angle = math.atan2(self.velocity.y, self.velocity.x)
        length = 8  # Boid size
        tip = self.position + pygame.Vector2(math.cos(angle), math.sin(angle)) * length
        left = self.position + pygame.Vector2(math.cos(angle + 2.5), math.sin(angle + 2.5)) * (length / 2)
        right = self.position + pygame.Vector2(math.cos(angle - 2.5), math.sin(angle - 2.5)) * (length / 2)
        pygame.draw.polygon(surface, self.color, [tip, left, right])

def main():
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Boids Simulation with Dynamic Background & Trails")
    clock = pygame.time.Clock()

    attractors = initialize_attractors() if ATTRACTOR_INIT else []
    background_surface = compute_background_surface(attractors)

    # Create a persistent trails surface with per-pixel alpha.
    trails = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
    trails.fill((0, 0, 0, 0))  # Fully transparent initially

    boids = [Boid() for _ in range(NUM_BOIDS)]
    attractors_changed = True

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.Vector2(event.pos)
                attractors.append(pos)
                attractors_changed = True

        if attractors_changed:
            background_surface = compute_background_surface(attractors)
            attractors_changed = False

        # Fade the trails surface by multiplying its RGBA values.
        fade_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        # Using BLEND_RGBA_MULT multiplies each pixel; a value of 240 (≈0.94) keeps most of the pixel.
        fade_surface.fill((255, 255, 255, TRAIL_ALPHA))
        trails.blit(fade_surface, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)

        # Update boids and draw them into the trails surface.
        for boid in boids:
            boid.update(boids, attractors)
            boid.draw(trails)

        # Compose the final frame by combining the attractor background and the trails.
        composite = background_surface.copy()
        composite.blit(trails, (0, 0))
        screen.blit(composite, (0, 0))

        pygame.display.flip()
        clock.tick(60)
        print(f"FPS: {clock.get_fps():.2f}")

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
