class Player: 
  def __init__ (self, position=[0,0], id=0) -> None:
    self.position = position
    self.id = id
    self.tried_bad_move = False
  
  def possible_moves(self, grid):
    moves = []
    if self.position[0] > 0 and grid[self.position[0] - 1][self.position[1]] >= 1:
      moves.append([-1, 0])
    if self.position[0] < grid.shape[0] - 1 and grid[self.position[0] + 1][self.position[1]] >= 1:
      moves.append([1, 0])
    if self.position[1] > 0 and grid[self.position[0]][self.position[1] - 1] >= 1:
      moves.append([0, -1])
    if self.position[1] < grid.shape[1] - 1 and grid[self.position[0]][self.position[1] + 1] >= 1:
      moves.append([0, 1])
    return moves

  def move(self, grid, direction):
    if direction in self.possible_moves(grid):
      self.position[0] += direction[0]
      self.position[1] += direction[1]
      return True
    else: 
      self.tried_bad_move = True
    return False