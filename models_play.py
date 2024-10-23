import torch
import numpy as np
import matplotlib.pyplot as plt

from ooxx_game_actor_critic import ActorNetwork, CriticNetwork
from ooxx_game_double_dqn import DuelingDQN
from ooxx_game_dqn import DQN
from ooxx_game_ppo import ActorCritic


class Agent:
    def __init__(self, actor_path, critic_path=None, model_type="dqn", index=1):
        self.index = index
        self.model_type = model_type
        self.actor = self.load_model(actor_path, model_type)
        self.critic = self.load_model(critic_path, "ac") if critic_path else None

    def load_model(self, model_path, model_type):
        if model_type == "double_dqn":
            model = DuelingDQN()
        elif model_type == "dqn":
            model = DQN()
        elif model_type == "ppo":
            model = ActorCritic()
        elif model_type == "ac":
            model = ActorNetwork() if "actor" in model_path else CriticNetwork()

        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model

    def act(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        if self.model_type in ["dqn", "double_dqn"]:
            q_values = self.actor(state_tensor).detach().numpy().flatten()
        elif self.model_type in ["ppo", "ac"]:
            policy_logits, _ = self.actor(state_tensor)
            q_values = torch.softmax(policy_logits, dim=1).detach().numpy().flatten()

        # 只选择空格子上的动作
        valid_actions = np.where(state == 0)[0]
        if len(valid_actions) == 0:
            raise ValueError("No valid actions available")

        valid_q_values = q_values[valid_actions]
        action = valid_actions[np.argmax(valid_q_values)]
        return action

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


def play_game(agent1, agent2, name1, name2, times=1000):
    competition_result = np.zeros(times)

    for i in range(times):
        state = np.zeros(9)
        while True:
            # Agent1 takes action
            action1 = agent1.act(state)
            state[action1] = agent1.index
            print(state.copy().reshape(3, 3))
            print("-" * 15)
            if agent1.isWin(state):
                print("Agent1 Wins!")
                competition_result[i] = 1
                break

            elif np.all(state != 0):
                print("Draw!")
                competition_result[i] = 0
                break

            # Agent2 takes action
            action2 = agent2.act(state)
            state[action2] = agent2.index
            print(state.copy().reshape(3, 3))
            print("-" * 15)
            if agent2.isWin(state):
                print("Agent2 Wins!")
                competition_result[i] = -1
                break
            elif np.all(state != 0):
                print("Draw!")
                competition_result[i] = 0
                break

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

    plt.plot(agent1_wins_proportion, label=f"agent1_{name1}", color="blue")
    plt.plot(agent2_wins_proportion, label=f"agent2_{name2}", color="red")
    plt.plot(agent_draws_proportion, label="agent draw", color="yellow")
    plt.legend()
    plt.title("agent win rate")
    plt.xlabel("step")
    plt.ylabel("win rate")

    plt.show()
    plt.savefig(f"./pics/{name1}_vs_{name2}_agent_win_rate.png")


if __name__ == "__main__":
    # Dueling DQN for Double DQN
    agent1 = Agent("./DOUBEL_DQN_model/20241023145122/agent1_model.pth", model_type="double_dqn", index=1)
    # PPO with combined Actor-Critic
    ppo_agent2 = Agent("./PPO_model/20241022180013/agent1_ppo_model.pth", model_type="ppo", index=1)
    ppo_agent2_op = Agent("./PPO_model/20241022180013/agent1_ppo_model.pth", model_type="ppo", index=-1)
    # AC with separate Actor and Critic
    agent3 = Agent(
        "./ActorCritic_model/20241022173900/agent1_actor.pth",
        "./ActorCritic_model/20241022173900/agent1_critic.pth",
        model_type="ac",
        index=1,
    )

    # DQN
    agent4 = Agent("./DQN_model/20241021222044/agent1_model.pth", model_type="dqn", index=-1)
    times = 2000
    play_game(ppo_agent2, ppo_agent2_op, "ppo", "ppo", times)
