from collections import deque

import networkx as nx
import numpy as np
from runner import Runner
from seeker import Seeker


class Labyrinth: 
  def __init__(self) -> None:
    self.grid =np.array([
      [1,1,1,0,0,1,1,1,1,1,1,0,0,1,1,1],
      [1,0,1,1,1,1,0,1,0,0,1,1,1,1,0,1],
      [1,0,1,0,0,0,0,1,0,0,0,0,1,0,0,1],
      # [1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,1],
      # [0,0,1,0,1,1,1,0,0,1,1,1,0,0,1,0],
      # [1,1,1,1,1,0,1,0,0,1,0,1,1,1,1,1],
      # [1,0,1,0,1,0,1,1,1,1,0,1,0,0,0,1],
      # [1,1,1,1,1,0,0,1,0,0,0,1,1,1,1,1],
      # [1,0,1,0,1,1,0,1,0,0,0,1,0,1,0,1],
      # [1,1,1,0,0,1,1,1,1,1,1,1,0,1,1,1]
    ])
    self.seekers = [Seeker([1,0], 0), Seeker([2,0], 1)]
    self.runner = Runner([2,7])
    # used for library
    self.graph = self.create_graph_from_grid()
    # self.seekers = [Seeker([0,0]), Seeker([0,0])]
    # self.runner = Runner([0,0])

  def labyrinth_with_seekers(self, steps):
    labyrinth = self.grid.copy().flatten()
    labyrinth = np.concatenate((labyrinth, self.seekers[0].position), axis=0)
    labyrinth = np.concatenate((labyrinth, self.seekers[1].position), axis=0)
    labyrinth = np.concatenate((labyrinth, self.runner.position), axis=0)
    labyrinth = np.concatenate((labyrinth, [self.tiles_reachable_by_runner() / 40]), axis=0)
    labyrinth = np.concatenate((labyrinth, [steps / 25]), axis=0)
    for seeker in self.seekers:
      try:
          path = nx.astar_path(self.graph, tuple(seeker.position), tuple(self.runner.position))
          labyrinth = np.concatenate((labyrinth, [len(path) - 1]), axis=0)
      except nx.NetworkXNoPath:
          print("No path exists")
    return labyrinth

  def labyrinth_with_players(self):
    labyrinth = self.labyrinth_with_seekers()
    labyrinth[self.runner.position[0]][self.runner.position[1]] = 10
    return labyrinth
  
  def create_graph_from_grid(self):
          G = nx.grid_2d_graph(*self.grid.shape)
          for i in range(self.grid.shape[0]):
              for j in range(self.grid.shape[1]):
                  if self.grid[i, j] == 0:
                      G.remove_node((i, j))
          return G
  def tiles_reachable_by_runner(self):
      visited = set()
      queue = deque([tuple(self.runner.position)])
      reachable_tiles = 0

      # if runenr position is in seeker position, return 0
      if tuple(self.runner.position) in [tuple(seeker.position) for seeker in self.seekers]:
        return 0

      while queue:
          current = queue.popleft()
          if current not in visited:
              visited.add(current)
              reachable_tiles += 1
              for direction in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                  next_tile = (current[0] + direction[0], current[1] + direction[1])

                  # Check if the next tile is within bounds and not a wall or seeker
                  if (0 <= next_tile[0] < self.grid.shape[0] and
                          0 <= next_tile[1] < self.grid.shape[1] and
                          self.grid[next_tile[0]][next_tile[1]] != 0 and
                          next_tile not in [tuple(seeker.position) for seeker in self.seekers]):
                      queue.append(next_tile)

      return reachable_tiles - 1  # Subtract 1 to exclude the starting tile
  def shortest_distance_to_runner(self):
      total_distance = 0
      for seeker in self.seekers:
          try:
              path = nx.astar_path(self.graph, tuple(seeker.position), tuple(self.runner.position))
              total_distance += len(path) - 1  # Subtract 1 since the starting point is included
          except nx.NetworkXNoPath:
              total_distance += float('inf')  # No path exists
      return total_distance


  def fitness(self, steps):
    if self.shortest_distance_to_runner() == 0:
      return 4
    # if self.check_and_punish_bad_move():
    #   return min(1/self.shortest_distance_to_runner(), 4) / 2 + 0.1/steps
    #  min(1/self.shortest_distance_to_runner(), 0) +
    return 0.1/steps + min(4 / (self.tiles_reachable_by_runner() + 2), 4)

  def check_and_punish_bad_move(self):
    for seeker in self.seekers:
      if seeker.tried_bad_move:
        seeker.tried_bad_move = False
        return True
    return False
