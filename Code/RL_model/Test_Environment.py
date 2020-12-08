# %%
import numpy as np
import pandas as pd
import copy
import re

class TestEnv:
    # Ref : action space = [(-1,0)), (1,0)), (0,-1)), (0,1)]
    
    def __init__(self):
        # self.grid_dic = {
        #     "출발점" : [(0,0)], 
        #     "empty" : [(0,1),(0,2),(0,3), (1,0),(1,1),(1,2),(1,3), 
        #     (2,0),(2,1),(2,2),(2,3), (3,0),(3,1),(3,2),(3,3), (4,0),(4,1),(4,2)], 
        #     "도착점" : [(4,3)]
        # }
        self.reward_cell = []
        self.grid_dic = self.set_grid_initialize
        self.grid_world = None
        self.init_reward_len = None
    
    # 그리드 딕셔너리, 그리드 월드 초기화
    def set_grid_initialize(self, total_size):
        self.grid_world = np.zeros(total_size)
        grid_val = list()
        grid_key = list()
        number = 0
        for i in range(self.grid_world.shape[0]):
            for j in range(self.grid_world.shape[1]):
                grid_val.append([(i, j)])
                key = "empty " + str(number)
                grid_key.append(key)
                number += 1

        self.grid_dic = dict(zip(grid_key, grid_val))

        return self.grid_dic

    # 그리드 월드 생성
    def make_grid_world(self):
        self.grid_world = pd.DataFrame(self.grid_world)
        for key in self.grid_dic.keys():
            for x,y in self.grid_dic[key]:
                self.grid_world.iloc[x,y] = key
        return self.grid_world

    # 출발지점 설정
    def set_start(self, start_point):
        for i, (key, value) in enumerate(self.grid_dic.copy().items()):
            if np.all(np.equal(start_point, value)) == True:

                # self.grid_dic[key] = "출발점"
                self.grid_dic['출발점'] = self.grid_dic.pop(key)

                self.grid_world.iloc[value[0][0], value[0][1]] = "출발점"

    # 도착지점 설정
    def set_end(self, end_point):
        for i, (key, value) in enumerate(self.grid_dic.copy().items()):
            if (isinstance(value, str) == False) and (np.all(np.equal(end_point, value)) == True):

                # self.grid_dic[key] = "도착점"
                self.grid_dic['도착점'] = self.grid_dic.pop(key)

                self.grid_world.iloc[value[0][0], value[0][1]] = "도착점"

        self.reward_cell.extend(self.grid_dic['도착점'])
        self.init_reward_len = len(self.reward_cell)

    # 보상지점 설정 (multiple 보상이 가능하게)
    def set_reward(self, reward_point):
        for r_i, r_item in enumerate(reward_point):
            for i, (key, value) in enumerate(self.grid_dic.copy().items()):
                if (isinstance(value, str) == False) and (np.all(np.equal(r_item, value)) == True):

                    # self.grid_dic[key] = "도착점"
                    reward_key = "보상 " + str(r_i)
                    self.grid_dic[reward_key] = self.grid_dic.pop(key)

                    self.grid_world.iloc[value[0][0], value[0][1]] = reward_key

        regex = re.compile('보상\s\d')
        for idx, key in enumerate(self.grid_dic):
            match = regex.findall(key)
            # print('key : {}, match : {}'.format(key, match))
        
            # if len(match) > 0 and (key == match[0]) or (key == "도착점"):
            if len(match) > 0 and (key == match[0]):
                # print('coordinate :', self.grid_dic[key])
                self.reward_cell.extend(self.grid_dic[key])
                # self.reward_cell.extend(self.grid_dic[key][0])
                # print('self.reward_cell :', self.reward_cell)

        self.init_reward_len = len(self.reward_cell)
        return self.reward_cell

    def get_reward(self, cell_pos):
        reward = 1  # 보상은 기본적으로 1점을 준다.
                    # 그 이유는 도착지점의 보상은 최소 2점이 가능하므로.
        
        # 그리드 월드를 벗어났다면
        if np.any(cell_pos < 0) or (cell_pos[0] > self.grid_world.shape[0] - 1) or (cell_pos[1] > self.grid_world.shape[1] - 1):
            reward = -1
            terminal = True
            return reward, terminal

        # 그리드 월드를 벗어나지 않았다면
        else:
            # 보상셀 = 도착좌표 (계산대) and 보상좌표

            # 보상셀이 둘 이상 남았을 때
            if len(self.reward_cell) > 1:   

                # 현재 좌표가 보상셀이라면
                if tuple(cell_pos) in self.reward_cell:

                    # 도달한 보상셀이 도착지점이라면
                    if tuple(cell_pos) == self.grid_dic['도착점'][0]:

                        # 에이전트는 보상 -1을 받고 에피소드가 끝난다.
                        reward = 0
                        terminal = True

                        return reward, terminal

                    # 도달한 보상셀이 도착지점이 아니라면
                    else:
                        # 이미 획득한 보상의 갯수만큼 보상점수를 추가하여 부여한다.
                        reward += self.init_reward_len - len(self.reward_cell)
                        cell_name = self.grid_world.iloc[cell_pos[0], cell_pos[1]]
                        self.remove_reward(cell_name)

                        terminal = False
                        
                        return reward, terminal

                # 현재 좌표가 보상셀이 아니라면
                else:
                    reward = 0
                    terminal = False
                    return reward, terminal
                
            # 보상셀이 하나남은 시점에서
            else:
                # 현재 좌표가 도착지점이라면
                if tuple(cell_pos) == self.grid_dic['도착점'][0]:
                    # 이미 획득한 보상의 갯수만큼 보상점수를 추가하여 부여한다.
                    reward += self.init_reward_len - len(self.reward_cell)
                    terminal = True

                # 현재 좌표가 도착지점이 아니라면
                else:
                    reward = 0
                    terminal = False

                return reward, terminal

    def remove_reward(self, cell_name):
        self.reward_cell.remove(self.grid_dic[cell_name][0])


    # # If the agent passes through the reward cell, the cell will no longer return the reward.
    # def get_reward(self, cell_pos):
    #     reward = 0
    #     if np.all(np.equal(cell_pos, self.reward_cell)):
    #         reward = 1
    #         terminal = True
    #         return reward, terminal

    #     elif (cell_pos[0] == -1) or (cell_pos[1] == -1) or (cell_pos[0] > 4) or (cell_pos[1] > 3):
    #         reward = -1
    #         terminal = True
    #         return reward, terminal

    #     else:
    #         reward = 0
    #         terminal = False
    #         return reward, terminal

    # Initialize the sate
    def initialize_state(self):
        self.pos = np.array([0, 0])
        return self.pos

    def move(self, agent, action):
        cur_pos = copy.deepcopy(agent.pos)  # agent의 현재 pos를 불러옮

        if action == 0:         # 상
            cur_pos[0] -= 1
            reward, terminal = self.get_reward(cur_pos)

        elif action == 1:       # 하
            cur_pos[0] += 1
            reward, terminal = self.get_reward(cur_pos)

        elif action == 2:       # 좌
            cur_pos[1] -= 1
            reward, terminal = self.get_reward(cur_pos)

        else:
            cur_pos[1] += 1     # 우
            reward, terminal = self.get_reward(cur_pos)

        return cur_pos, reward, terminal

# %%
