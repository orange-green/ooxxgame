from collections import deque
from datetime import datetime
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        # input输入是当前棋盘的状态state
        self.fc1 = nn.Linear(9, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 9)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# 将 Q 表替换为一个神经网络，用于近似 Q 值函数。这样可以处理更大的状态空间
class DQNAgent:
    def __init__(self, OOXX_index, epsilon=0.1, learning_rate=0.001, gamma=0.9, memory_size=1000, batch_size=64):
        self.index = OOXX_index
        self.current_state = np.zeros(9)
        self.previous_state = np.zeros(9)
        self.previous_action = None
        self.epsilon = epsilon
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.model = DQN()
        # 引入一个固定的目标网络，定期更新其权重以稳定训练过程
        self.target_model = DQN()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        # self.update_target_model()
        self.update_counter = 0
        self.update_frequency = 10  # 每10次replay更新一次目标模型
        self.epsilon_decay_frequency = 20  # 每20次replay减少一次探索率

    def update_target_model(self):
        # 复制训练模型的权重到目标模型
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        # 使用经验回放技术存储智能体的经历，并从中随机采样进行训练，以打破数据相关性
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            # np.where() 返回的是符合条件的元素的下标，返回结果是数组有多少维就有多少个元组组成
            # 随机取元素值为0的一个下标
            return random.choice(np.where(state == 0)[0])
        # squeeze 是用来减少维度的函数，它会去除张量中维度为1的维度，从而使张量变得更紧凑
        #  unsqueeze 则是用来增加维度的函数，它会在张量的指定位置增加一个维度
        # 这里使用unsqueeze增加一个维度是为了符合pytorch的标准，将张量（9，）转成 （1，9）
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        # detach() 方法用于返回一个新的 Tensor，这个 Tensor 和原来的 Tensor 共享相同的内存空间，但是不会被计算图所追踪，也就是说它不会参与反向传播，不会影响到原有的计算图，用来缓存中间结果
        q_values = self.model(state_tensor).detach().numpy()[0]
        available_action_locations = np.where(state == 0)[0]

        # np.argmax() 返回多维矩阵中最大值元素的下标（或坐标）
        # action指代的是在哪个位置落子
        action = available_action_locations[np.argmax(q_values[available_action_locations])]
        self.previous_state = state.copy()
        self.previous_action = action
        return action

    def replay(self):
        # 模型在以往的经验中进行小批次样本学习
        if len(self.memory) < self.batch_size:
            return

        # random.sample(k, count) 随机样本选择，在数列k中随机选取count个样本
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
                # 将下一个状态 next_state_tensor 输入到目标网络中，得到目标网络对下一个状态的Q值预测
                # np.max()和np.amax()等价函数，求得数组中的第一个最大值， 一个返回元素值的下标（np.max），一个返回元素值(np.amax)
                # self.gamma为折扣因子， 为了区分及时奖励和长期奖励的重要性
                target += self.gamma * np.amax(self.target_model(next_state_tensor).detach().numpy()[0])

            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            target_f = self.model(state_tensor).detach().numpy()
            target_f[0][action] = target
            target_tensor = torch.FloatTensor(target_f)
            self.optimizer.zero_grad()

            # 获取训练模型的预测输出q值
            output = self.model(state_tensor)
            # 计算q值和目标q值的误差，追求误差的最小化
            loss = nn.MSELoss()(output, target_tensor)
            loss.backward()
            self.optimizer.step()

        # 更新计数器
        self.update_counter += 1

        # 更新目标模型
        if self.update_counter % self.update_frequency == 0:
            self.update_target_model()

        # 减少探索率
        if self.update_counter % self.epsilon_decay_frequency == 0:
            self.reduce_epsilon()

    def reduce_epsilon(self, min_epsilon=0.01, decay=0.995):
        if self.epsilon > min_epsilon:
            self.epsilon *= decay

    def isWin(self, state):
        state = state.reshape(3, 3)
        for i in range(3):
            if (state[i, 0] == self.index and state[i, 1] == self.index and state[i, 2] == self.index) or (
                state[0, i] == self.index and state[1, i] == self.index and state[2, i] == self.index
            ):
                return True
        if (state[0, 0] == self.index and state[1, 1] == self.index and state[2, 2] == self.index) or (
            state[0, 2] == self.index and state[1, 1] == self.index and state[2, 0] == self.index
        ):
            return True

        return False

    def reset(self):
        self.current_state = np.zeros(9)
        self.previous_state = np.zeros(9)
        self.previous_action = None

    def save_model(self, model_name: str):
        # 获取当前时间
        current_time = datetime.now().strftime("%Y%m%d%H%M%S")
        # 构建保存路径
        directory = f"DQN_model/{current_time}/"
        # 创建目录
        os.makedirs(directory, exist_ok=True)
        # 保存模型
        model_path = os.path.join(directory, f"{model_name}_model.pth")
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")


def train_dqn(agent, episodes=1000):
    for e in range(episodes):
        state = np.zeros(9)
        done = False
        while not done:
            action = agent.act(state)
            next_state = state.copy()
            next_state[action] = agent.index
            reward = 0
            if agent.isWin(next_state):
                reward = 1
                done = True
            elif 0 not in next_state:
                reward = 0.5
                done = True
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            agent.replay()
        agent.reduce_epsilon()

        if e % 10 == 0:
            agent.update_target_model()

    print("Training completed")


# 两个模型的进行下棋对弈训练
def self_play_train(agent1: DQNAgent, agent2: DQNAgent, e1, e2, times=1000):
    agent1.epsilon = e1
    agent2.epsilon = e2

    competition_result = np.zeros(times)

    for i in range(times):
        print(f"Battle {i}:")
        state = np.zeros(9)

        while True:
            # Agent1 takes action
            action1 = agent1.act(state)
            next_state = state.copy()
            next_state[action1] = agent1.index
            # 胜出、平局、输掉的reward分别是1、0、-1
            if agent1.isWin(next_state):
                competition_result[i] = 1
                agent1.remember(state, action1, 1, next_state, True)
                agent2.remember(agent2.current_state, agent2.previous_action, -1, next_state, True)
                print("Agent1 Win")
                break
            elif np.all(next_state != 0):
                agent1.remember(state, action1, 0, next_state, True)
                agent2.remember(agent2.current_state, agent2.previous_action, 0, next_state, True)
                competition_result[i] = 0
                print("Draw")
                break
            else:
                agent1.remember(state, action1, -1, next_state, False)

            state = next_state

            # Agent2 takes action
            action2 = agent2.act(state)
            next_state = state.copy()
            next_state[action2] = agent2.index
            if agent2.isWin(next_state):
                competition_result[i] = -1
                agent1.remember(agent1.current_state, agent1.previous_action, -1, next_state, True)
                agent2.remember(state, action2, 1, next_state, True)
                print("Agent2 Win")
                break
            elif np.all(next_state != 0):
                agent1.remember(agent1.current_state, agent1.previous_action, 0, next_state, True)
                agent2.remember(state, action2, 0, next_state, True)
                competition_result[i] = 0
                print("Draw")
                break
            else:
                agent2.remember(state, action2, -1, next_state, False)

            state = next_state

        agent1.replay()
        agent2.replay()
        agent1.reset()
        agent2.reset()

    # 保存模型
    agent1.save_model("agent1")
    agent1.save_model("agent2")

    agent1_wins = 0
    agent2_wins = 0
    agent_draws = 0
    agent1_wins_proportion = np.zeros(times)
    agent2_wins_proportion = np.zeros(times)
    agent_draws_proportion = np.zeros(times)
    for i in range(times):
        if competition_result[i] == 1:
            agent1_wins += 1
        if competition_result[i] == -1:
            agent2_wins += 1
        if competition_result[i] == 0:
            agent_draws += 1
        agent1_wins_proportion[i] = agent1_wins / (i + 1)
        agent2_wins_proportion[i] = agent2_wins / (i + 1)
        agent_draws_proportion[i] = agent_draws / (i + 1)

    plt.figure(figsize=(15, 7))

    plt.plot(agent1_wins_proportion, label="agent1", color="blue")
    plt.plot(agent2_wins_proportion, label="agent2", color="red")
    plt.plot(agent_draws_proportion, label="agent draw", color="yellow")
    plt.legend()
    plt.title("DQN agent win rate")
    plt.xlabel("step")
    plt.ylabel("win rate")

    plt.show()


if __name__ == "__main__":
    times = 1000
    agent1 = DQNAgent(OOXX_index=1)
    agent2 = DQNAgent(OOXX_index=-1)

    self_play_train(agent1, agent2, 0.1, 0.1, times)
