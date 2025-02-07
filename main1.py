import pygame
import sys
import math
import random

# Configuration
SCREEN_WIDTH = 1500
SCREEN_HEIGHT = 1000
NUM_BOIDS = 120
BACKGROUND_COLOR = (0, 0, 0)
BOID_COLOR = (255, 255, 255)

# Boid behavior parameters
MAX_SPEED = 6
PERCEPTION_RADIUS = 30
SEPARATION_RADIUS = 40
COHESION_WEIGHT = 0.1
ALIGNMENT_WEIGHT = 0.05
SEPARATION_WEIGHT = 3

class Boid:
    def __init__(self):
        self.position = pygame.Vector2(random.uniform(0, SCREEN_WIDTH), 
                                      random.uniform(0, SCREEN_HEIGHT))
        angle = random.uniform(0, 2*math.pi)
        self.velocity = pygame.Vector2(math.cos(angle), math.sin(angle)) * MAX_SPEED

    def update(self, boids):
        self.apply_rules(boids)
        self.position += self.velocity
        self.edges()

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
                # Separation
                if distance < SEPARATION_RADIUS:
                    diff = self.position - boid.position
                    diff = diff.normalize() / distance
                    separation += diff
                
                # Alignment
                alignment += boid.velocity
                
                # Cohesion
                cohesion += boid.position
                total += 1

        if total > 0:
            # Calculate averages
            alignment /= total
            cohesion = (cohesion / total - self.position).normalize()
            
            # Apply weights
            self.velocity += separation * SEPARATION_WEIGHT
            self.velocity += alignment * ALIGNMENT_WEIGHT
            self.velocity += cohesion * COHESION_WEIGHT

        # Limit speed
        if self.velocity.length() > MAX_SPEED:
            self.velocity = self.velocity.normalize() * MAX_SPEED

    def edges(self):
        if self.position.x > SCREEN_WIDTH:
            self.position.x = 0
        elif self.position.x < 0:
            self.position.x = SCREEN_WIDTH
        if self.position.y > SCREEN_HEIGHT:
            self.position.y = 0
        elif self.position.y < 0:
            self.position.y = SCREEN_HEIGHT

    def draw(self, screen):
        angle = math.atan2(self.velocity.y, self.velocity.x)
        length = 15
        
        # Create triangle points
        tip = self.position + pygame.Vector2(math.cos(angle), math.sin(angle)) * length
        left = self.position + pygame.Vector2(math.cos(angle + 2.5), math.sin(angle + 2.5)) * length/2
        right = self.position + pygame.Vector2(math.cos(angle - 2.5), math.sin(angle - 2.5)) * length/2
        
        pygame.draw.polygon(screen, BOID_COLOR, [tip, left, right])

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