import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from RL_model.Environment import Env
from RL_model.Agent import Agent
from utility.Utility import Error

def main():
    agent = Agent((1,0)) 
    env = Env()
    print(env.move(agent, 3))
    agent.get_pos()

if __name__ == "__main__":
    main()