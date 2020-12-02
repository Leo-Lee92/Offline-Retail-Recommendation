
import os
import sys
from RL_model.Environment import Env
from RL_model.Agent import Agent
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utility.Utility import Error
import pandas as pd
import copy
import numpy as np

def main():
    #import data sets
    dir = os.path.dirname(os.path.abspath(__file__)) + "/Preprocessing/Data/"
    rule_file_name = "rules.csv"
    trans_prob_file_name = "transition_probability_final.csv"

    ass_list = []                                                           # 10 plan list will be stored from 'rules.csv' data
    file = open(dir+rule_file_name, 'r')
    for line in file.readlines():
        line = line.strip()
        list_t = line.split(",")
        ass_list.append(list_t)
    
    trans_prob_data = pd.read_csv(dir+trans_prob_file_name)                 #Transition probability matrix from 'transition_probability_final.csv' data
    trans_prob_data = trans_prob_data.set_index("Unnamed: 0")

    #Initialization
    num_episodes = 1                                                        # the number of episodes to be execution.
    env = Env(trans_prob_data)                                                             # initialize Environment
    initial_state = env.initialize_state()                                  # initialize_state to (0,0)
    agent = Agent(env.grid_dic)                                             # Show Grid_world to Agent


    #형 shopping_plan_list는 매 episode 마다 바뀌어야 하니까 Excute episode 밑 for문 안에 들어가있어야하는 게 맞는거 같아요!

    #shopping_plan_list = ass_list[np.random.randint(len(ass_list))]         # Type the plan list of shopping. Datatype should be list.
    shopping_plan_list = ass_list[0]
    reward_cell = env.set_reward(shopping_plan_list)                        # reward cell is set.


    '''
    Excute episodes
    '''
    
    for episode in range(num_episodes):
        terminal = False    # If done = False, episode continues, otherwise episode ends. Episode ends when agent arrives at goal
        initial_state = env.initialize_state()  # initial environment for agent, (0,0)
        cur_state = copy.deepcopy(initial_state) 
        agent.set_pos(cur_state)# initial state for agent, (0,0)
        while(terminal == False):   # continues episode until it arrives at goal
            print("current_pos : {0}".format(agent.pos))
            cur_state = agent.get_pos()
            action = agent.get_action(cur_state, "epsilon-greedy")  # agent select action according to policy
            print("action : {0}".format(action))
            
            next_state, reward , terminal = env.move(agent, action)
            print("next_state : {0}".format(next_state))
            print("reward : {0}".format(reward))
            print("terminal : {0}".format(terminal))
            #print(next_state)
            #agent.set_pos(next_state)
    

if __name__ == "__main__":
    main()
