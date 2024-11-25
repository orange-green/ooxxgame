from datetime import datetime
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt


# 通过分离状态值和优势函数来提高性能
class DuelingDQN(nn.Module):
    def __init__(self):
        super(DuelingDQN, self).__init__()
        # 特征提取层, 将输入的 9 个状态特征转换为 128 维的特征向量
        self.feature_layer = nn.Sequential(nn.Linear(9, 128), nn.ReLU())
        # 状态值流, 计算每个状态的基础价值, 输出是一个标量，
        # 代表当前状态的整体价值,因为它表示当前状态的整体价值，
        # 与具体的动作无关。这个标量是对当前状态的一个全局评估
        self.value_layer = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        # 优势流,计算每个动作相对于平均值的优势, 输出是一个向量，代表在当前状态下每个可能动作的优势值
        self.advantage_layer = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 9))

    def forward(self, x):
        # 提取特征
        features = self.feature_layer(x)
        # 计算状态值
        values = self.value_layer(features)
        # 计算优势
        advantages = self.advantage_layer(features)
        # 计算 Q 值, 将状态值和去均值后的优势值相加，得到最终的 Q 值
        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))

        return q_values


# 减少 Q-learning 的过估计偏差
class DoubleDQNAgent:
    def __init__(self, OOXX_index, epsilon=0.1, learning_rate=0.001, gamma=0.9, memory_size=1000, batch_size=64):
        self.index = OOXX_index
        self.epsilon = epsilon
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.model = DuelingDQN()
        self.target_model = DuelingDQN()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.update_target_model()
        self.current_state = np.zeros(9)
        self.previous_state = np.zeros(9)
        self.previous_action = None
        self.update_counter = 0
        self.update_frequency = 20
        self.epsilon_decay_frequency = 20

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        self.current_state = state
        if np.random.rand() <= self.epsilon:
            self.previous_action = random.choice(np.where(state == 0)[0])
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.model(state_tensor).detach().numpy()[0]
            available_actions = np.where(state == 0)[0]
            self.previous_action = available_actions[np.argmax(q_values[available_actions])]
            self.previous_state = state.copy()

        return self.previous_action

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)

            target = reward
            if not done:
                next_action = self.model(next_state_tensor).detach().argmax().item()
                target += self.gamma * self.target_model(next_state_tensor).detach().numpy()[0][next_action]

            target_f = self.model(state_tensor).detach().numpy()
            target_f[0][action] = target

            target_tensor = torch.FloatTensor(target_f)
            self.optimizer.zero_grad()
            output = self.model(state_tensor)
            loss = nn.MSELoss()(output, target_tensor)
            loss.backward()
            self.optimizer.step()

        self.update_counter += 1
        if self.update_counter % self.update_frequency == 0:
            self.update_target_model()
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
        self.previous_action = np.zeros(9)

    def save_model(self, agent_name):
        current_time = datetime.now().strftime("%Y%m%d%H%M%S")
        directory = os.path.join("DOUBEL_DQN_model", current_time)
        os.makedirs(directory, exist_ok=True)
        model_path = os.path.join(directory, f"{agent_name}_model.pth")
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")


def self_play_train(agent1: DoubleDQNAgent, agent2: DoubleDQNAgent, e1, e2, times=1000):
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
    agent2.save_model("agent2")
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
    plt.title(" double DQN agent win rate")
    plt.xlabel("step")
    plt.ylabel("win rate")
    plt.savefig("./pics/double_DQN_agent_win_rate.png")


if __name__ == "__main__":
    times = 4000
    agent1 = DoubleDQNAgent(OOXX_index=1)
    agent2 = DoubleDQNAgent(OOXX_index=-1)
    self_play_train(agent1, agent2, 0.2, 0.2, times)
