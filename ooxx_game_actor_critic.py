import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from datetime import datetime
import os
import matplotlib.pyplot as plt


# 一个 Actor网络用于选择动作，一个Critic网络用于评估状态值
class ActorNetwork(nn.Module):
    def __init__(self):
        super(ActorNetwork, self).__init__()
        self.fc = nn.Linear(9, 128)
        self.relu = nn.ReLU()

        self.policy = nn.Linear(128, 9)

    def forward(self, x):
        x = self.relu(self.fc(x))
        # 输出一个概率分布，表示在9个位子上每个位置落子的概率
        policy = torch.softmax(self.policy(x), dim=-1)
        return policy


# Critic负责评估当前状态的价值，具体来说是评估在当前状态下采取某个动作后，未来可能获得的回报的期望值
class CriticNetwork(nn.Module):
    def __init__(self):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(9, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # 确保输出是标量

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ActorCriticAgent:
    def __init__(self, OOXX_index, epsilon=0.1, learning_rate=0.001, gamma=0.9, memory_size=1000, batch_size=64):
        self.index = OOXX_index
        self.current_state = np.zeros(9)
        self.previous_state = np.zeros(9)
        self.previous_action = None
        self.epsilon = epsilon
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.actor = ActorNetwork()
        self.critic = CriticNetwork()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        self.update_counter = 0
        self.update_frequency = 20  # 每20次replay更新一次目标模型
        self.epsilon_decay_frequency = 20  # 每20次replay减少一次探索率

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.choice(np.where(state == 0)[0])

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        policy = self.actor(state_tensor).detach().numpy()[0]
        available_action_locations = np.where(state == 0)[0]
        action = available_action_locations[np.argmax(policy[available_action_locations])]
        self.previous_state = state.copy()
        self.previous_action = action
        return action

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
            value = self.critic(state_tensor).squeeze()
            next_value = self.critic(next_state_tensor).squeeze()
            target_value = reward + (1 - done) * self.gamma * next_value.item()
            advantage = target_value - value.item()

            # 更新 Critic 网络
            target_value_tensor = torch.FloatTensor([[target_value]])
            self.critic_optimizer.zero_grad()
            value_loss = nn.MSELoss()(value, target_value_tensor)
            value_loss.backward()
            self.critic_optimizer.step()

            # 更新 Actor 网络
            self.actor_optimizer.zero_grad()
            policy = self.actor(state_tensor)
            policy_loss = -torch.log(policy[0, action]) * advantage
            policy_loss = policy_loss.mean()  # 将policy_loss变成标量
            policy_loss.backward()
            self.actor_optimizer.step()

        self.update_counter += 1

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
        current_time = datetime.now().strftime("%Y%m%d%H%M%S")
        directory = f"ActorCritic_model/{current_time}/"
        os.makedirs(directory, exist_ok=True)
        actor_path = os.path.join(directory, f"{model_name}_actor.pth")
        critic_path = os.path.join(directory, f"{model_name}_critic.pth")
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
        print(f"Model saved to {actor_path} and {critic_path}")


def self_play_train(agent1, agent2, e1, e2, times=1000):
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
    plt.title(" Actor Critic agent win rate")
    plt.xlabel("step")
    plt.ylabel("win rate")

    plt.savefig("./pics/actor_critic_agent_win_rate.png")


if __name__ == "__main__":
    times = 4000
    agent1 = ActorCriticAgent(OOXX_index=1)
    agent2 = ActorCriticAgent(OOXX_index=-1)
    self_play_train(agent1, agent2, 0.2, 0.2, times)
