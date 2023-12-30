import multiprocessing
import time

import neat
import numpy as np
import pygame
import tensorflow as tf
from labyrinth import Labyrinth

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 600, 800
TILE_SIZE = WIDTH // 5
BACKGROUND_COLOR_RUNNABLE = (255, 255, 255)
BACKGROUND_COLOR_OBSTACLE = (0, 0, 0)
SEEKER_COLOR = (255, 0, 0)  # Red
RUNNER_COLOR = (0, 0, 255)  # Blue

# Setup the display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Labyrinth Game")

# Function to draw the grid
def draw_grid(grid):
  for y in range(grid.shape[0]):
    for x in range(grid.shape[1]):
      rect = pygame.Rect(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE)
      if grid[y][x] == 0:
        pygame.draw.rect(screen, BACKGROUND_COLOR_OBSTACLE, rect)
      else:
        pygame.draw.rect(screen, BACKGROUND_COLOR_RUNNABLE, rect, 1)

# Function to draw the seekers and runner
def draw_players(seekers, runner):
    for seeker in seekers:
        pygame.draw.circle(screen, SEEKER_COLOR, (seeker.position[1] * TILE_SIZE + TILE_SIZE // 2, seeker.position[0] * TILE_SIZE + TILE_SIZE // 2), TILE_SIZE // 4)
    
    pygame.draw.circle(screen, RUNNER_COLOR, (runner.position[1] * TILE_SIZE + TILE_SIZE // 2, runner.position[0] * TILE_SIZE + TILE_SIZE // 2), TILE_SIZE // 4)

# Load configuration.
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward.cfg')

def main(show, net):
    labyrinth = Labyrinth()
    clock = pygame.time.Clock()

    steps = 0
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # AI moves the seekers here
        output = net.activate(labyrinth.labyrinth_with_seekers(steps).flatten())
        # first four values are for the first seeker, second four values are for the second seeker
        # 0: up, 1: down, 2: left, 3: right
        for i in range(2):
            if output[i * 4] > 0.5:
                labyrinth.seekers[i].move(labyrinth.grid, [-1, 0])
            elif output[i * 4 + 1] > 0.5:
                labyrinth.seekers[i].move(labyrinth.grid, [1, 0])
            elif output[i * 4 + 2] > 0.5:
                labyrinth.seekers[i].move(labyrinth.grid, [0, -1])
            elif output[i * 4 + 3] > 0.5:
                labyrinth.seekers[i].move(labyrinth.grid, [0, 1])
      
        if (steps % 10):
           labyrinth.runner.move(labyrinth.grid, [0, -1])

        # if (steps == 30 or labyrinth.shortest_distance_to_runner() == 0):
        if (steps == 30):
            running = False
        if show:
          screen.fill(BACKGROUND_COLOR_RUNNABLE)
          draw_grid(labyrinth.grid)
          draw_players(labyrinth.seekers, labyrinth.runner)

          pygame.display.flip()
          clock.tick(2)
        steps += 1

    return labyrinth.fitness(steps)



# Define the fitness function.
def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = main(False, net)


# Create the population, which is the top-level object for a NEAT run.
p = neat.Population(config)

# Add a stdout reporter to show progress in the terminal.
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)

# Run NEAT.
winner = p.run(eval_genomes, 50)  # Run for up to 50 generations.

# Display the winning genome.
print('\nBest genome:\n{!s}'.format(winner))

main(True, neat.nn.FeedForwardNetwork.create(winner, config))



# if __name__ == "__main__":
#     main()

time.sleep(5)