import pygame
import sys
import math
import random
import noise

# Configuration
SCREEN_WIDTH = 1500
SCREEN_HEIGHT = 1000
NUM_BOIDS = 100
EDGE_MARGIN = 30 # should be close to perception radius
ADD_TRAILS = True
TRAIL_ALPHA = 10

# Boid behavior parameters
START_SPEED = 2.25
MAX_SPEED = 5
PERCEPTION_RADIUS = 30
SEPARATION_RADIUS = 40
COHESION_WEIGHT = 0.4
ALIGNMENT_WEIGHT = 0.2
SEPARATION_WEIGHT = 3
RANDOMNESS_WEIGHT = 0.225
MASS_EQUIVALENT = 1.5
TURN_FACTOR = 0.0325
SPEED_ADJUSTMENT_FACTOR = 0.3
VELOCITY_DAMPING = 0.95

# Color
PROXIMITY_NORMALIZATION = 5
ACCELERATION_UPPER_EXPECTED = MAX_SPEED / 10
BACKGROUND_COLOR = (0, 0, 0)
BASE_COLOR = (50,50,50)

def clamp_color(color):
    return max(0, min(255, color))

def clamp_colors(r, g, b):
    return (clamp_color(r), clamp_color(g), clamp_color(b))

def get_color(base_value, factor):
    return base_value + int((255 - base_value) * factor)

class Boid:
    def __init__(self):
        self.position = pygame.Vector2(random.uniform(0, SCREEN_WIDTH), 
                                       random.uniform(0, SCREEN_HEIGHT))
        angle = random.uniform(0, 2 * math.pi)
        self.velocity = pygame.Vector2(math.cos(angle), math.sin(angle)) * START_SPEED
        self.acceleration = pygame.Vector2(0, 0)
        self.color = (0,0,0)
        
        # Individual behavior variations
        self.cohesion_weight = COHESION_WEIGHT * random.uniform(0.8, 1.2)
        self.alignment_weight = ALIGNMENT_WEIGHT * random.uniform(0.8, 1.2)
        self.separation_weight = SEPARATION_WEIGHT * random.uniform(0.8, 1.2)
        
        # Noise parameters
        self.noise_offset_x = random.random() * 1000
        self.noise_offset_y = random.random() * 1000

    def update(self, boids):
        self.acceleration = pygame.Vector2(0, 0)
        self.apply_rules(boids)
        self.avoid_edges()
        self.add_noise()
        
        # Update physics
        self.velocity += self.acceleration / MASS_EQUIVALENT
        
        # Speed management
        speed = self.velocity.length()
        if speed > MAX_SPEED:
            self.velocity = self.velocity.normalize() * MAX_SPEED
        if speed != 0:
            new_speed = speed + (START_SPEED - speed) * SPEED_ADJUSTMENT_FACTOR
            self.velocity = self.velocity.normalize() * new_speed

        self.position += self.velocity
        self.update_color(boids)
        self.velocity *= VELOCITY_DAMPING

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
            self.acceleration += (alignment/total) * self.alignment_weight
            self.acceleration += ((cohesion/total) - self.position).normalize() * self.cohesion_weight

    def avoid_edges(self):
        margin = EDGE_MARGIN
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
        # speed_ratio = self.velocity.length() / MAX_SPEED

        self.color = clamp_colors(
            get_color(BASE_COLOR[0], acceleration_factor),
            BASE_COLOR[1], 
            get_color(BASE_COLOR[2], proximity_factor), 
        )

    def draw(self, screen):
        angle = math.atan2(self.velocity.y, self.velocity.x)
        length = 8  # Boid size
        
        # Create triangle points
        tip = self.position + pygame.Vector2(math.cos(angle), math.sin(angle)) * length
        left = self.position + pygame.Vector2(math.cos(angle + 2.5), math.sin(angle + 2.5)) * (length/2)
        right = self.position + pygame.Vector2(math.cos(angle - 2.5), math.sin(angle - 2.5)) * (length/2)
        
        pygame.draw.polygon(screen, self.color, [tip, left, right])

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Boids Flocking Simulation with Trails")
    clock = pygame.time.Clock()

    # comment out if not in use
    # Trail initialization
    trail_surface = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
    trail_surface.fill((0, 0, 0, TRAIL_ALPHA))

    boids = [Boid() for _ in range(NUM_BOIDS)]

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Handle screen updates
        if ADD_TRAILS:
            screen.blit(trail_surface, (0, 0))
        else:
            screen.fill(BACKGROUND_COLOR)

        # Update and draw boids
        for boid in boids:
            boid.update(boids)
            boid.draw(screen)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()