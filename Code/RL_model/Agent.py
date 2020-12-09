#%%
import numpy as np
import tensorflow as tf
class Agent(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(self, **kwargs)
        self.action_space = np.array([(-1,0), (1,0), (0,-1), (0,1)]) # (상, 하, 좌, 우) #hi
        self.state_space = None
        self.q_table = None
        self.policy = np.array([0.20,0.20,0.20,0.20,0.20])
        self.pos = None

        self.policy = "epsilon-greedy"
        self.epsilon = 1
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99

        # self.dense1 = tf.keras.layers.Dense(units = 32, activation = 'relu', kernel_initializer = tf.keras.initializers.zeros)  # build the first layer of Agent's deep q-network
        self.dense3 = tf.keras.layers.Dense(units = len(self.action_space), kernel_initializer = tf.keras.initializers.zeros)  # build the first layer of Agent's deep q-network

    def call(self, cur_state):              # Take cur_state as input
        cur_state = tf.expand_dims(cur_state, 0)
        # x = self.dense1(cur_state)          # Return q_vector of which each element represents q-value.
        q_vector = self.dense3(cur_state)           # Return q_vector of which each element represents q-value.

        return q_vector

    def set_q_table(self, state_space):
        self.state_space = copy.deepcopy(state_space)
        self.q_table = np.zeros([len(self.state_space), len(self.action_space)])

    def set_pos(self, pos):
        self.pos = pos
        return self.pos

    def get_pos(self):
        return self.pos

    '''
    get_q_vector 함수는 tableu 모드일때만 사용.
    '''
    def get_q_vector(self, env, cur_pos):

        # (1) 현재 pos가 q_table 상에서 어느 state일지 idx를 판단
        q_table_idx = np.array([])

        for i in range(len(env.grid_dic)):
            cur_state = np.all(np.equal(list(env.grid_dic.values())[i], cur_pos))
            q_table_idx = np.append(q_table_idx, cur_state)

        # q_table 상에서의 현재 state의 idx
        cur_state_idx = np.where(q_table_idx == 1)[0]  

        # (2) q-vector 뽑아오기
        q_vector = self.q_table[cur_state_idx, :]

        if len(q_vector) == 0:
            q_vector = np.zeros([1, self.q_table.shape[1]])

        return q_vector

    '''
    update_q_table 함수는 tableu 모드일때만 사용.
    '''
    def update_q_table(self, cur_pos, action, loss):
        # (1) 현재 pos가 q_table 상에서 어느 state일지 idx를 판단
        q_table_idx = np.array([])

        for i in range(len(env.grid_dic)):
            cur_state = np.all(np.equal(list(env.grid_dic.values())[i], cur_pos))
            q_table_idx = np.append(q_table_idx, cur_state)

        # q_table 상에서의 현재 state의 idx
        cur_state_idx = np.where(q_table_idx == 1)[0]   

        # (2) q_table 업데이트 하기
        self.q_table[cur_state_idx, action] += loss


    def get_action(self, q_vector):
        action = None                                             # Initailize local variable "action" as None type.
        
        try:
            # Try greedy policy
            if self.policy == "greedy":                                
                action_idx = tf.argmax(q_vector[0])                        # action_idx is either 0, 1, 2, 3 or 4 (제자리, 상, 하, 좌, 우).
                action = self.action_space[action_idx]

            # Try epsilon-greedy policy (SARSA, Q-learning)
            elif self.policy == "epsilon-greedy":

                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay

                random_prob = np.random.uniform(low=0.0, high=1.0, size= 1)[0]      # random value from normal distribution
                print('epsilon :', self.epsilon)

                # EXPLORATION : if random value is less and equal to epsilon
                if random_prob <= self.epsilon:        
                    print('random !')
                    action_vector = range(len(self.action_space))                                  
                    action = np.random.choice(action_vector, size = 1)[0]

                # EXPLOITATION : if random value is greater than epsilon
                else:
                    print('exploitation !')
                    action = tf.argmax(q_vector[0])     # if multiple actions have the same highest value, take the first index.
        except:
            print('terminal !')
            action = 0

        return action

def Train_TABLE(agent, env, len_episode, algorithm):
    gamma = 0.99
    lr = 1e-5

    # (0) 현재 상태 판단
    cur_pos = agent.get_pos()   # 현재 상태 조회

    if algorithm == "SARSA":    # SARSA 'predicts' next action one more time based on current policy and 'control' with the predicted action.

        # (1) 현재 Q-value 예측
        q_vector = agent.get_q_vector(env, cur_pos)
        action = agent.get_action(q_vector)
        action_vector = tf.one_hot(action, depth = len(agent.action_space))
        pred_qval = tf.reduce_sum(action_vector * q_vector, axis = 1)

        # (2) 다음 상태 (next_pos)를 조회
        next_pos, reward, terminal = env.move(agent, action)    

        # (3) 타겟 Q-value 예측
        next_q_vector = agent.get_q_vector(env, next_pos)
        next_action = agent.get_action(next_q_vector)
        next_action_vector = tf.one_hot(next_action, depth = len(agent.action_space))
        optimal_qval = tf.reduce_sum(next_action_vector * next_q_vector, axis = 1)            
        target_qval = reward + (1 - terminal) * gamma * optimal_qval

        # (4) Q-value 통제
        loss = lr * (target_qval - pred_qval)
        agent.update_q_table(cur_pos, action, loss)

        print("Episode : {0}, Action : {1}, current_pos : {2} -> next_pos : {3}, Reward : {4}, Terminal : {5}".format(epi, action, cur_pos, next_pos, reward, terminal))

    if algorithm == "Q-learning":   # Q-learning 'control' with the observation that has the maximum q-value. Do not predicts one more time.

        # (1) 현재 Q-value 예측
        q_vector = agent.get_q_vector(env, cur_pos)
        action = agent.get_action(q_vector)
        action_vector = tf.one_hot(action, depth = len(agent.action_space))
        pred_qval = tf.reduce_sum(action_vector * q_vector, axis = 1)

        # (2) 다음 상태 (next_pos)를 조회
        next_pos, reward, terminal = env.move(agent, action)      

        # (3) 타겟 Q-value 예측
        next_q_vector = agent.get_q_vector(env, next_pos)
        max_qval = np.amax(next_q_vector, axis = -1)
        target_qval = reward + (1 - terminal) * gamma * max_qval

        # (4) Q-value 통제
        loss = lr * (target_qval - pred_qval)
        agent.update_q_table(cur_pos, action, loss)

        print("Episode : {0}, Action : {1}, current_pos : {2} -> next_pos : {3}, Reward : {4}, Terminal : {5}".format(epi, action, cur_pos, next_pos, reward, terminal))

    return  loss, reward, next_pos, terminal

def Train_DEEP(agent, env, len_episode, algorithm):
    gamma = 0.99
    lr = 1e-20
    optimizer = tf.keras.optimizers.Adam(lr)

    (0) 현재 상태 판단
    cur_pos = agent.get_pos()   # 현재 상태 조회
    
    if algorithm == "SARSA":    # SARSA 'predicts' next action one more time based on current policy and 'control' with the predicted action.
        with tf.GradientTape() as tape:
            tape.watch(agent.trainable_variables)

            # (1) 현재 Q-value 예측
            q_vector = agent(cur_pos)
            action = agent.get_action(q_vector)
            action_vector = tf.one_hot(action, depth = len(agent.action_space))
            pred_qval = tf.reduce_sum(action_vector * q_vector, axis = 1)

            # (2) 다음 상태 (next_pos)를 조회
            next_pos, reward, terminal = env.move(agent, action)    

            # (3) 타겟 Q-value 예측
            next_q_vector = agent(next_pos)
            next_action = agent.get_action(next_q_vector)
            next_action_vector = tf.one_hot(next_action, depth = len(agent.action_space))
            optimal_qval = tf.reduce_sum(next_action_vector * next_q_vector, axis = 1)            
            target_qval = reward + (1 - terminal) * gamma * optimal_qval

            # (4) 손실 계산
            loss = tf.reduce_mean(tf.math.square(tf.stop_gradient(target_qval) - pred_qval))

        gradients = tape.gradient(loss, agent.trainable_variables)
        optimizer.apply_gradients(zip(gradients, agent.trainable_variables))

        print("Episode : {0}, Action : {1}, current_pos : {2} -> next_pos : {3}, Reward : {4}, Terminal : {5}".format(epi, action, cur_pos, next_pos, reward, terminal))

    if algorithm == "Q-learning":   # Q-learning 'control' with the observation that has the maximum q-value. Do not predicts one more time.
        with tf.GradientTape() as tape:
            tape.watch(agent.trainable_variables)

            # (1) 현재 Q-value 예측
            q_vector = agent(cur_pos)
            action = agent.get_action(q_vector)
            action_vector = tf.one_hot(action, depth = len(agent.action_space))
            pred_qval = tf.reduce_sum(action_vector * q_vector, axis = 1)

            # (2) 다음 상태 (next_pos)를 조회
            next_pos, reward, terminal = env.move(agent, action)    

            # (3) 타겟 Q-value 예측
            next_q_vector = agent(next_pos)
            max_qval = np.amax(next_q_vector, axis = -1)
            target_qval = reward + (1 - terminal) * gamma * max_qval

            # (4) 손실 계산
            loss = tf.reduce_mean(tf.math.square(tf.stop_gradient(target_qval) - pred_qval))

        gradients = tape.gradient(loss, agent.trainable_variables)
        optimizer.apply_gradients(zip(gradients, agent.trainable_variables))   

        print("Episode : {0}, Action : {1}, current_pos : {2} -> next_pos : {3}, Reward : {4}, Terminal : {5}".format(epi, action, cur_pos, next_pos, reward, terminal))

    return  loss.numpy(), reward, next_pos, terminal
    
# %%
