import pygame
import sys
import math
import random
import noise

# Configuration
SCREEN_WIDTH = 1500
SCREEN_HEIGHT = 1000
NUM_BOIDS = 120
BACKGROUND_COLOR = (0, 0, 0)
EDGE_MARGIN = 5  # New margin for edge avoidance

# Boid behavior parameters
START_SPEED = 2
MAX_SPEED = 5
PERCEPTION_RADIUS = 30
SEPARATION_RADIUS = 40
COHESION_WEIGHT = 0.4
ALIGNMENT_WEIGHT = 0.2
SEPARATION_WEIGHT = 3
RANDOMNESS_WEIGHT = 0.225  # Factor for noise-based movement

TURN_FACTOR = 0.5
SPEED_ADJUSTMENT_FACTOR = 0.3

def clamp_color(color):
    return max(0, min(255, color))

def clamp_colors(r, g, b):
    return (clamp_color(r), clamp_color(g), clamp_color(b))

class Boid:
    def __init__(self):
        self.position = pygame.Vector2(random.uniform(0, SCREEN_WIDTH), 
                                       random.uniform(0, SCREEN_HEIGHT))
        angle = random.uniform(0, 2*math.pi)
        self.velocity = pygame.Vector2(math.cos(angle), math.sin(angle)) * START_SPEED
        self.color = (100, 100, 100)
        
        # Variable behavior parameters (±20% variation)
        self.cohesion_weight = COHESION_WEIGHT * random.uniform(0.8, 1.2)
        self.alignment_weight = ALIGNMENT_WEIGHT * random.uniform(0.8, 1.2)
        self.separation_weight = SEPARATION_WEIGHT * random.uniform(0.8, 1.2)
        
        # Initialize noise offsets for smooth randomness
        self.noise_offset_x = random.random() * 1000
        self.noise_offset_y = random.random() * 1000

    def update(self, boids):
        self.apply_rules(boids)
        self.avoid_edges()  # edge avoidance
        self.add_noise()    # smooth, noise-based perturbation
        self.position += self.velocity
        self.update_color(boids)  # dynamic color update

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
                # Separation: steer away from too-close boids
                if distance < SEPARATION_RADIUS:
                    diff = self.position - boid.position
                    diff = diff.normalize() / distance
                    separation += diff
                # Alignment: match velocity
                alignment += boid.velocity
                # Cohesion: steer toward the average position
                cohesion += boid.position
                total += 1

        if total > 0:
            alignment /= total
            cohesion = (cohesion / total - self.position).normalize()
            self.velocity += separation * self.separation_weight
            self.velocity += alignment * self.alignment_weight
            self.velocity += cohesion * self.cohesion_weight

        # Limit speed to MAX_SPEED
        speed = self.velocity.length()
        if speed > MAX_SPEED:
            self.velocity = self.velocity.normalize() * MAX_SPEED

        # Gradually adjust speed toward START_SPEED
        current_speed = self.velocity.length()
        if current_speed != 0:
            new_speed = current_speed + (START_SPEED - current_speed) * SPEED_ADJUSTMENT_FACTOR
            self.velocity = self.velocity.normalize() * new_speed

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
            
        self.velocity += force

    def add_noise(self):
        # Use Perlin noise for smooth randomness; scale factors adjust spatial frequency.
        scale = 0.01
        nx = noise.pnoise1(self.noise_offset_x)
        ny = noise.pnoise1(self.noise_offset_y)
        noise_vector = pygame.Vector2(nx, ny)
        self.velocity += noise_vector * RANDOMNESS_WEIGHT
        # Increment noise offsets to evolve the noise over time.
        self.noise_offset_x += 0.01
        self.noise_offset_y += 0.01

    def update_color(self, boids):
        # Color based on speed (red) and local boid density (blue)
        speed_ratio = self.velocity.length() / MAX_SPEED
        red = int(255 * speed_ratio)
        
        avg_distance = 0
        count = 0
        for boid in boids:
            if boid == self:
                continue
            distance = self.position.distance_to(boid.position)
            if distance < PERCEPTION_RADIUS:
                avg_distance += distance
                count += 1
                
        if count > 0:
            avg_distance /= count
            proximity = 1 - (avg_distance / PERCEPTION_RADIUS)
            blue = int(255 * proximity)
        else:
            blue = 0
            
        self.color = clamp_colors(red, 100, blue)

    def draw(self, screen):
        angle = math.atan2(self.velocity.y, self.velocity.x)
        length = 8  # Size of the boid
        
        tip = self.position + pygame.Vector2(math.cos(angle), math.sin(angle)) * length
        left = self.position + pygame.Vector2(math.cos(angle + 2.5), math.sin(angle + 2.5)) * (length / 2)
        right = self.position + pygame.Vector2(math.cos(angle - 2.5), math.sin(angle - 2.5)) * (length / 2)
        
        pygame.draw.polygon(screen, self.color, [tip, left, right])

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Boids Flocking Simulation")
    clock = pygame.time.Clock()

    boids = [Boid() for _ in range(NUM_BOIDS)]

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill(BACKGROUND_COLOR)

        for boid in boids:
            boid.update(boids)
            boid.draw(screen)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
