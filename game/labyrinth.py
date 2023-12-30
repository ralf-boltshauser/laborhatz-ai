import numpy as np
from runner import Runner
from seeker import Seeker


class Labyrinth: 
  def __init__(self) -> None:
    self.grid =np.array([
      [1,1,1,1],
      [1,0,1,1],
      [1,0,0,1],
      [1,1,1,1],
      [1,0,1,1],
      [1,1,1,0],
    ])
    self.seekers = [Seeker([0,2], 0), Seeker([3,3], 1)]
    self.runner = Runner([5,2])
    # self.seekers = [Seeker([0,0]), Seeker([0,0])]
    # self.runner = Runner([0,0])

  def labyrinth_with_seekers(self, steps):
    labyrinth = self.grid.copy().flatten()
    labyrinth = np.concatenate((labyrinth, self.seekers[0].position), axis=0)
    labyrinth = np.concatenate((labyrinth, self.seekers[1].position), axis=0)
    labyrinth = np.concatenate((labyrinth, [steps / 25]), axis=0)
    return labyrinth

  def labyrinth_with_players(self):
    labyrinth = self.labyrinth_with_seekers()
    labyrinth[self.runner.position[0]][self.runner.position[1]] = 10
    return labyrinth
  
  def shortest_distance_to_runner(self):
    distances = [np.linalg.norm(np.array(seeker.position) - np.array(self.runner.position)) for seeker in self.seekers]
    return sum(distances)
  def fitness(self, steps):
    if self.shortest_distance_to_runner() == 0:
      return 4
    if self.check_and_punish_bad_move():
      return min(1/self.shortest_distance_to_runner(), 4) / 2 + 0.1/steps
    return min(1/self.shortest_distance_to_runner(), 4) + 0.1/steps

  def check_and_punish_bad_move(self):
    for seeker in self.seekers:
      if seeker.tried_bad_move:
        seeker.tried_bad_move = False
        return True
    return False
