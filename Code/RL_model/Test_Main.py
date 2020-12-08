# %%
import os
import sys
import pandas as pd
import copy
import numpy as np

# %%
# 단순한 world에서 테스트 해보기 
env = TestEnv()
env.set_grid_initialize((4, 4))             # 테스트 grid 환경을 원하는 사이즈로 초기화하기 
env.make_grid_world()                       # 만들어진 grid world 확인하기
env.set_start((0, 0))                       # 출발지점 좌표 설정해주기
env.set_end((2, 3))                         # 도착지점 좌표 설정해주기
env.set_reward([(1, 1)])                    # 보상 좌표 설정해주기 (도착지점만 보상좌표로 설정하고 싶으면 보상좌표 설정 안하면 됨.)
env.grid_world                              # 그리드 월드 확인
reward_cell = copy.deepcopy(env.reward_cell)

# state_info = []
# for i, val in enumerate(env.grid_dic.values()):
#     state_info.extend(val)

# agent = Agent(state_info)
# agent.set_q_table()
# agent.q_table
agent = Agent()
agent.epsilon = 1.0
num_episodes = 1000
parameter_list = []

for epi in range(num_episodes):
    # 상태 초기화
    initial_state = env.initialize_state()

    # 보상 초기화
    env.reward_cell = copy.deepcopy(reward_cell)
    
    # 초기화된 상태를 에이전트 초기 위치로 설정
    agent.set_pos(initial_state)
    
    # 하이퍼 파라미터 초기화
    terminal = False
    complete = False
    total_reward = 0
    total_loss = 0
    mean_reward = 0
    mean_loss = 0
    len_episode = 0

    while(terminal == False):
        len_episode += 1
        # (2) 통제
        # training_sample = (cur_pos, action, reward, next_pos)
        # loss = Train(agent, training_sample, "Q-learning")
        loss, reward, next_pos, terminal = Train(agent, env, len_episode, "Q-learning")
        print('reward_cell :{}, reward : {}'.format(env.reward_cell, reward))

        if (terminal == True) and (reward > 1):
            print("Here!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print('reward_cell :{}, reward : {}'.format(env.reward_cell, reward))
            complete = True

        # (3) 상태 반영
        agent.set_pos(next_pos)                                             # Agent's next_pos is set to be the cur_pos.
        # print('agent.pos :', agent.pos)

        # (4) 파라미터 추적
        total_loss += loss
        total_reward += reward

    mean_reward = total_reward / len_episode
    mean_loss = total_loss / len_episode
    print('mean_reward :', mean_reward)
    print('mean_loss:', mean_loss)
    print(' ')

    parameter_list.append((epi, len_episode, mean_reward, mean_loss, terminal, complete))
    # if epi % 100 == 0 or epi == (num_episodes - 1):
    #     print('parameter_list :', parameter_list)


# %%
from matplotlib import pyplot as plt
pd.DataFrame(parameter_list).iloc(axis = 1)[2]
plt.plot(pd.DataFrame(parameter_list).iloc(axis = 1)[1])
plt.plot(pd.DataFrame(parameter_list).iloc(axis = 1)[2])
plt.plot(pd.DataFrame(parameter_list).iloc(axis = 1)[3])
len(np.where(pd.DataFrame(parameter_list).iloc(axis = 1)[5] == True)[0])