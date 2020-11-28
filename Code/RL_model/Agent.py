import numpy as np
class Agent:
    def __init__(self, initial_position):
        self.action_space = np.array([(0,0), (-1,0), (1,0), (0,-1), (0,1)])
        self.action_pr = np.array([0.20,0.20,0.20,0.20,0.20])
        self.pos = np.array(initial_position)

    def set_pos(self, position):
        self.pos = position
        return self.pos

    def get_pos(self):
        return self.pos