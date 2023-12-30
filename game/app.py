import multiprocessing
import os
import pickle
import time

import neat
import numpy as np
import pygame
import tensorflow as tf
from labyrinth import Labyrinth

# Constants
WIDTH, HEIGHT = 1600, 1000
TILE_SIZE = WIDTH // 16
BACKGROUND_COLOR_RUNNABLE = (255, 255, 255)
BACKGROUND_COLOR_OBSTACLE = (0, 0, 0)
SEEKER_COLOR = (255, 0, 0)  # Red
RUNNER_COLOR = (0, 0, 255)  # Blue

screen = None

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


def main(show, net):
    global screen
    # Initialize Pygame
    pygame.init()


    # Setup the display
    if show:
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Labyrinth Game")
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
      
        # if (steps % 10):
        #    labyrinth.runner.move(labyrinth.grid, [0, -1])

        # if (steps == 30 or labyrinth.shortest_distance_to_runner() == 0):
        if (steps == 30):
            running = False
        if show:
          screen.fill(BACKGROUND_COLOR_RUNNABLE)
          draw_grid(labyrinth.grid)
          draw_players(labyrinth.seekers, labyrinth.runner)

          pygame.display.flip()
          clock.tick(5)
        steps += 1

    return labyrinth.fitness(steps)



# Multiprocessing
# # Define the fitness function.
# def eval_genome(genome_id, genome, config):
#     net = neat.nn.FeedForwardNetwork.create(genome, config)
#     return genome_id, main(False, net)  # Assuming 'False' means no GUI display

# def eval_genomes(genomes, config):
#     pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())  # Pool of worker processes
#     results = pool.starmap(eval_genome, [(genome_id, genome, config) for genome_id, genome in genomes])
   
#     # Close the pool properly
#     pool.close()
#     pool.join() 
#     # Apply the fitness back to each genome
#     # set all genomes to have a fitness of 0
#     for genome_id, genome in genomes:
#         genome.fitness = 0
#     for genome_id, fitness in results:
#         try:
#             genomes[genome_id % 500][1].fitness = fitness
#             print(f"Genome {genome_id % 500} fitness: {fitness}")
#         except:
#             print(f"Genome {genome_id % 500} fitness: {fitness} failed")


# Single process
def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    return main(False, net)  # Assuming 'False' means no GUI display

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)
        # print(f"Genome {genome_id % 500} fitness: {genome.fitness}")


if __name__ == "__main__":
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        'config-feedforward.cfg')
    filename = 'winner.pkl'

    if os.path.exists(filename):
        with open(filename, 'rb') as input:
            winner = pickle.load(input)
            # Now you can use the 'winner' object
            print("Loaded winner successfully.")
    else:
        print(f"File {filename} does not exist.")
        # Create the population, which is the top-level object for a NEAT run.
        p = neat.Population(config)

        # Add a stdout reporter to show progress in the terminal.
        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)

        # Run NEAT.
        winner = p.run(eval_genomes, 50)  # Run for up to 50 generations.

        # Assuming 'winner' is your winning genome
        with open('winner.pkl', 'wb') as output:
            pickle.dump(winner, output, 1)



    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    main(True, neat.nn.FeedForwardNetwork.create(winner, config))

    time.sleep(5)