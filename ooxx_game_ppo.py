from datetime import datetime
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt


class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(9, 128)
        # actor网络
        self.fc_actor = nn.Linear(128, 9)
        # critic网络
        self.fc_critic = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        # 输出动作选择概率分布
        policy_logits = self.fc_actor(x)
        # 评估动作好坏程度
        value = self.fc_critic(x)
        return policy_logits, value


class PPOAgent:
    def __init__(self, OOXX_index, epsilon=0.1, lr=0.001, gamma=0.99, eps_clip=0.2, memory_size=1000, batch_size=64):
        self.index = OOXX_index
        self.epsilon = epsilon
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.policy = ActorCritic()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic()
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.current_state = np.zeros(9)
        self.previous_action = None

    def act(self, state):
        self.current_state = state
        action = None
        if np.random.rand() <= self.epsilon:
            available_actions = np.where(state == 0)[0]
            action = random.choice(available_actions)
            self.previous_action = action
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            policy_logits, _ = self.policy_old(state_tensor)
            policy = torch.softmax(policy_logits, dim=1)
            available_actions = np.where(state == 0)[0]
            action = available_actions[torch.argmax(policy[0, available_actions]).item()]
            self.previous_action = action

        return action

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        _, next_value = self.policy_old(torch.FloatTensor(next_states[-1]).unsqueeze(0))
        next_value = next_value.detach().item()
        returns = self.compute_gae(rewards, 1 - dones, [v.item() for v in self.policy_old(states)[1]], next_value)

        returns = torch.FloatTensor(returns)
        old_log_probs, old_values, _ = self.evaluate(states, actions)

        for _ in range(4):
            log_probs, values, dist_entropy = self.evaluate(states, actions)
            ratios = torch.exp(log_probs - old_log_probs.detach())
            advantages = returns - old_values.detach()

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * nn.MSELoss()(values, returns) - 0.01 * dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())

    def compute_gae(self, rewards, masks, values, next_value):
        values = values + [next_value]
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * masks[step] - values[step]
            gae = delta + self.gamma * 0.95 * masks[step] * gae
            returns.insert(0, gae + values[step])
        return returns

    def evaluate(self, state, action):
        policy_logits, value = self.policy(state)
        policy = torch.softmax(policy_logits, dim=1)
        action_log_probs = torch.log(policy.gather(1, action))
        dist_entropy = -(policy * torch.log(policy)).sum(dim=1)
        return action_log_probs, torch.squeeze(value), dist_entropy

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
        self.previous_action = None

    def save_model(self, agent_name):
        current_time = datetime.now().strftime("%Y%m%d%H%M%S")
        directory = os.path.join("PPO_model", current_time)
        os.makedirs(directory, exist_ok=True)
        model_path = os.path.join(directory, f"{agent_name}_ppo_model.pth")
        torch.save(self.policy.state_dict(), model_path)
        print(f"Model saved as {model_path}")


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
            # print(next_state.copy().reshape(3, 3))
            # print("-" * 15)
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
    plt.title("PPO DQN agent win rate")
    plt.xlabel("step")
    plt.ylabel("win rate")
    plt.savefig("./pics/ppo_agent_win_rate.png")


if __name__ == "__main__":
    pass
    # 使用示例
    times = 10000
    agent1 = PPOAgent(1)
    agent2 = PPOAgent(-1)
    self_play_train(agent1, agent2, 0.2, 0.2, times)
