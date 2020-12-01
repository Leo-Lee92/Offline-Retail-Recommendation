import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from RL_model.Environment import Env
from RL_model.Agent import Agent
from utility.Utility import Error
import copy
import numpy as np

def main():
    num_episodes = 100  # the number of episodes to be execution.

    env = Env()
    initial_state = env.initialize_state()

    agent = Agent(env.grid_dic)

    shopping_plan_list = ["청과", "채소", "축산"]        # Type the plan list of shopping. Datatype should be list.
    reward_cell = env.set_reward(shopping_plan_list)    # reward cell is set.


    '''
    Excute episodes
    '''
    for episode in range(num_episodes):
        terminal = False    # If done = False, episode continues, otherwise episode ends. Episode ends when agent arrives at goal

        initial_state = env.initialize_state()
        cur_state = copy.deepcopy(initial_state) 

        while(terminal == False):   # continues episode until it arrives at goal

            action = agent.get_action(cur_state, "epsilon-greedy")  # agent select action according to policy
            state = env.move(agent, action)
            agent.set_pos(state)


if __name__ == "__main__":
    main()