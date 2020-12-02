import numpy as np
import pandas as pd
from utility.Utility import Error
import copy
class Env:
    # Ref : action space = [(0,0)), (-1,0)), (1,0)), (0,-1)), (0,1)]
    
    def __init__(self, trans_prob):
        self.grid_dic = {"청과" : [(0,1),(0,2),(0,3),(0,4)], "곡물" : [(1,1),(1,2),(1,3),(1,4)], "입구" : [(0,0)], 
                "채소" : [(0,5),(0,6),(0,7),(1,5),(1,6),(1,7)], "닭고기" : [(0,8)], "수산" : [(0,9), (1,8), (1,9)], "축산" : [(0,10), (1,10)], 
                "델리카" : [(0,11), (1,11)], "계산대" : [(1,0),(2,0),(3,0),(4,0)], "청소욕실" : [(2,5),(2,6),(3,5),(3,6)],
                "주방용품" : [(2,7),(3,7)], "언더웨어" : [(4,5), (4,6)], "H&B" : [(2,1),(2,2),(2,3),(2,4), (3,1),(3,2),(3,3),(3,4), (4,1),(4,2),(4,3),(4,4)],
                "건강,차" : [(4,7)], "조미식품" : [(2,8), (2,9)], "인스턴트" : [(3,8), (3,9)], "과자" : [(4,8), (4,9)], 
                "냉장냉동" : [(2,10), (2,11)], "데일리" : [(3,10), (3,11)], "음료,주류" : [(4,10),(4,11)],
                "스포츠" : [(5,1)], "문구" : [(5,2)], "완구" : [(5,3)], "자동차" : [(5,4)], "애완원예용품" : [(5,5)], "인테리어" : [(5,6)],
                "수예" : [(5,7)], "세제" : [(5,8)], "위생용품" : [(5,9)], "금지구역" : [(5,10),(5,11),(5,0)]}
        self.grid_world = self.make_grid_world()
        self.trans_prob = trans_prob
        self.reward_cell = []


    # Set Reward cell
    def set_reward(self, list_of_cells):
        for key in list_of_cells:
            self.reward_cell = self.reward_cell + self.grid_dic[key]
        return np.array(self.reward_cell)

    #If the agent passes through the reward cell, the cell will no longer return the reward.
    # Remove Reward cell
    def get_reward(self, cell_pos):
        if len(self.reward_cell) != 0:
            for r_cell in self.reward_cell:
                if r_cell == tuple(cell_pos):
                    cell_name = self.grid_world.iloc[cell_pos[0], cell_pos[1]]
                    self.remove_reward(cell_name)
                    return 1
                else:
                    return 0
        else:
            return 0
        
    def remove_reward(self, cell_name):
        for cell in self.grid_dic[cell_name]:
            try: self.reward_cell.remove(cell)
            except: pass

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
        out_of_grid_world = False
        #현재 좌표
        current_pos = agent.pos

        #현재 좌표에 대응되는 현재 매대
        cur_key = self.grid_world.iloc[current_pos[0], current_pos[1]]
        next_key = None
        #전이확률을 고려하지 않을 시, temp_pos 를 next_pos로 바꿀 것
        #이동된 좌표
        try: temp_pos = np.array(agent.pos)
        #try: temp_pos = agent.pos + agent.action_space[action] 
        except: raise Error()

        #이동된 좌표에 대응되는 다음 매대
        try: temp_key = self.grid_world.iloc[temp_pos[0], temp_pos[1]]
        except: temp_key = "금지구역"

        #매대간 이동이 존재하는대도 만약 cur_key와 temp_key가 같을 시,(이동이 발생했는대도 그리드월드 설계문제로 인해 매대 이동이 발생하지 못함)
        #이 때 이동좌표를 변경시켜줌
        t_pos = 0
        if action != 0:
            if cur_key == temp_key:
                temp = np.array(self.grid_dic[temp_key]).T
                if (action == 1) or (action == 2): 
                    if action == 1:                     # when action is up
                        temp_pos[0] = min(temp[0])
                    elif action == 2:                   # when action is down
                        temp_pos[0] = max(temp[0])
                    t_pos = (temp_pos + agent.action_space[action])[0]
                    try: target_pos = self.grid_world.iloc[t_pos,:]
                    except: out_of_grid_world = True

                elif (action == 3) or (action == 4): 
                    if action == 3:                     # when action is left
                        temp_pos[1] = min(temp[1])
                    elif action == 4:                   # when action is right
                        temp_pos[1] = max(temp[1])
                    t_pos = (temp_pos + agent.action_space[action])[1]
                    try: target_pos = self.grid_world.iloc[:, t_pos]
                    except: out_of_grid_world = True

                else: 
                    t_pos = -1 #error(Unbounded local variable error)를 피하기 위해 삽입함.
                    raise Error()

            if t_pos < 0 or out_of_grid_world == True: #이동불가능한 매대임.(그리드 월드를 벗어난 경우)
                terminal = True
                print("Terminal!, out of grid world")
                print("이동을 시도한 좌표(+1) : {0}, {1}, {2}".format(temp_pos, t_pos, out_of_grid_world))
                show_next_pos = agent.set_pos(agent.pos) #원래위치 반환
                next_key = "금지구역"
                next_pos = temp_pos #error(Unbounded local variable error)를 피하기 위해 삽입함.
        
            else:
                prob_list = []
                prob_list.append(self.trans_prob.loc[cur_key,cur_key])   #자기자신으로 돌아올 확률 추가
                for target in list(target_pos):
                    try:
                        prob_list.append(self.trans_prob.loc[cur_key,target]) #전이확률 반영
                    except: #금지구역에 대응되는 전이확률은 없음으로, 고정된 확률값을 삽입함.
                        prob_list.append(1/(len(target_pos)+1))
                prob_list = np.array(prob_list) / np.sum(np.array(prob_list))
                order_index = np.random.choice(np.arange(len(target_pos)+1), 1, p = prob_list)
                if order_index[0] == 0:
                    next_key = cur_key
                else:  
                    next_key = target_pos.values[order_index[0]-1]

                if next_key == cur_key:
                    next_pos = current_pos
                else:
                    if (action == 1) or (action == 2):
                        next_pos = (t_pos, order_index[0]-1)
                    elif (action == 3) or (action == 4):
                        next_pos = (order_index[0]-1, t_pos)
                        #print(next_pos)
                    else: pass
        else:
            next_pos = current_pos
        

        #현재 좌표가 최종목적지인지 확인
        if self.grid_world.iloc[current_pos[0], current_pos[1]] == "계산대":
            if len(self.reward_cell) == 0: # 현재 좌표가 계산대이고 모든 보상 셀을 지나친 경우, terminal을 종료하고 보상을 (10)얻음. 
                print("Terminal!, 계산대진입")
                terminal = True
                show_next_pos = agent.set_pos(agent.pos)
                reward = 10
            else: #현재 좌표가 계산대이지만, 모든 보상 셀을 지나치지 않은 경우, 다시 이동함.
                pass

        if terminal == True: #이동된 좌표가 이동 불가능한 지점인지 확인
            reward = -1
        
        elif next_key == "금지구역" or tuple(next_pos) in self.grid_dic["금지구역"] :
            print("Terminal!, 금지구역진입")
            print("이동을 시도한 좌표 : {0}, {1}".format(next_key, next_pos))
            terminal = True
            show_next_pos = agent.set_pos(agent.pos)
            reward = -1

        #이동가능한 동선이면 '이동':
        else:
            show_next_pos = agent.set_pos(next_pos)
            reward = self.get_reward(show_next_pos)
        
        return show_next_pos, reward, terminal
    

