import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import numpy as np
import pandas as pd
from utility.Utility import Error

class Env:
    # action space = [(0,0)), (-1,0)), (1,0)), (0,-1)), (0,1)]
    
    def __init__(self):
        self.grid_dic = {"청과" : [(0,1),(0,2),(0,3),(0,4)], "곡물" : [(1,1),(1,2),(1,3),(1,4)], "입구" : [(0,0)], 
                "채소" : [(0,5),(0,6),(0,7),(1,5),(1,6),(1,7)], "닭고기" : [(0,8)], "수산" : [(0,9), (1,8), (1,9)], "축산" : [(0,10), (1,10)], 
                "델리카" : [(0,11), (1,11)], "계산대" : [(1,0),(2,0),(3,0),(4,0)], "청소욕실" : [(2,5),(2,6),(3,5),(3,6)],
                "주방용품" : [(2,7),(3,7)], "언더웨어" : [(4,5), (4,6)], "H&B" : [(2,1),(2,2),(2,3),(2,4), (3,1),(3,2),(3,3),(3,4), (4,1),(4,2),(4,3),(4,4)],
                "건강,차" : [(4,7)], "조미식품" : [(2,8), (2,9)], "인스턴트" : [(3,8), (3,9)], "과자" : [(4,8), (4,9)], 
                "냉장냉동" : [(2,10), (2,11)], "데일리" : [(3,10), (3,11)], "음료,주류" : [(4,10),(4,11)],
                "스포츠" : [(5,1)], "문구" : [(5,2)], "완구" : [(5,3)], "자동차" : [(5,4)], "애완원예용품" : [(5,5)], "인테리어" : [(5,6)],
                "수예" : [(5,7)], "세제" : [(5,8)], "위생용품" : [(5,9)], "금지구역" : [(5,10),(5,11),(5,0)]}
        self.grid_world = self.make_grid_world()
        self.reward_cell = []

    # Set Reward
    def set_reward(self, list_of_cells):
        for key in list_of_cells:
            self.reward_cell = self.reward_cell + self.grid_dic[key]

        return np.array(self.reward_cell)

    # Initialize the sate
    def initialize_state(self):
        return np.array((0, 0))

    def make_grid_world(self):
        grid_world = pd.DataFrame(np.zeros((6,12)))
        for key in self.grid_dic.keys():
            for x,y in self.grid_dic[key]:
                grid_world.iloc[x,y] = key
        return grid_world

    def move(self, agent, action):
        terminal = False
        reward = 0

        #현재 좌표
        current_pos = agent.pos

        #현재 좌표에 대응되는 현재 매대
        cur_key = self.grid_world.iloc[current_pos[0], current_pos[1]]

        #이동된 좌표
        try: next_pos = agent.pos + agent.action_space[action] #아직은 전이확률을 고려하지 않음
        except: raise Error()

        #이동된 좌표에 대응되는 다음 매대
        try: next_key = self.grid_world.iloc[next_pos[0], next_pos[1]]
        except: next_key = "금지구역"

        #매대간 이동이 존재하는대도 만약 cur_key와 next_key가 같을 시,(이동이 발생했는대도 그리드월드 설계문제로 인해 매대 이동이 발생하지 못함)
        #이 때 이동좌표를 변경시켜줌
        if action != 0:
            if cur_key == next_key:
                temp = np.array(self.grid_dic[next_key]).T
                if action == 1: next_pos[0] = min(temp[0]) - 1
                elif action == 2: next_pos[0] = max(temp[0]) + 1
                elif action == 3: next_pos[1] = min(temp[1]) - 1
                elif action == 4: next_pos[1] = max(temp[1]) + 1
                else: raise Error()

        #현재 좌표가 최종목적지인지 확인
        if self.grid_world.iloc[current_pos[0], current_pos[1]] == "계산대":
            terminal = True
            show_next_pos = agent.set_pos(agent.pos)
            #reward = 0

        #이동된 좌표가 이동 불가능한 지점인지 확인
        elif next_key == "금지구역" or tuple(next_pos) in self.grid_dic["금지구역"] or next_pos[0] < 0 or next_pos[1] < 0:
            terminal = True
            show_next_pos = agent.set_pos(agent.pos)
            #reward = 0

        #이동가능한 동선이면 '이동':
        else:
            show_next_pos = agent.set_pos(next_pos)
        
        return show_next_pos, reward, terminal