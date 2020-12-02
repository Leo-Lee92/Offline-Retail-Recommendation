import numpy as np
class Agent:
    def __init__(self, state_info):
        self.state_space = len(state_info) 
        self.action_space = np.array([(0,0), (-1,0), (1,0), (0,-1), (0,1)]) # (제자리, 상, 하, 좌, 우) #hi
        self.policy = np.array([0.20,0.20,0.20,0.20,0.20])
        self.q_table = np.zeros([self.state_space, self.action_space.shape[0]])
        self.pos = None #hi

    def set_pos(self, position): #hi
        self.pos = position
        return self.pos

    def get_pos(self): #hi
        return self.pos

    def get_action(self, cur_state, policy): #hi
        
        #self.q_table[cur_state, :]

        if policy == "greedy":  # To try greedy policy
            pass
        elif policy == "epsilon-greedy":    # To try epsilon-greedy (SARSA, Q-learning)
            action_idx = np.argmax(self.q_table[cur_state, :])  # action_idx is either 0, 1, 2, 3 or 4 (제자리, 상, 하, 좌, 우).
            #action = self.action_space[action_idx]              # transform action_idx to action 
                                                                # e.g., if action_idx = 2 then action[action_idx] = (1, 0)
        # sample input : 4
        return 4


    def update_q_table(self):   # Update Q-table via this function.
                                # If tabluar case, updating q-table means "learning"
                                # q-table value is updated according to a learing algorithm (e.g., SARSA, Q-learning)
        pass
    