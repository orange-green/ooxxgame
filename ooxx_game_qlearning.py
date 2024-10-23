import numpy as np
import matplotlib.pyplot as plt


# Q-learning,是一种特定的 TD 学习方法,直接学习状态-动作对（state-action pairs）的价值，即 Q 值,目标是找到最优的策略，即使在没有模型的情况下
class Agent:
    def __init__(self, OOXX_index, Epsilon=0.1, LearningRate=0.1, DiscountFactor=0.9):
        # Q-table with state space (3^9) and action space (9)
        # q表存放着空间状态-动作状态-q值， 本质是个高维空间，用空间向量来定位找到动作状态数组，动作状态数组存放着各个动作的q值
        self.q_table = np.zeros((3, 3, 3, 3, 3, 3, 3, 3, 3, 9))
        self.currentState = np.zeros(9)
        self.previousState = np.zeros(9)
        self.previousAction = None
        self.index = OOXX_index
        self.epsilon = Epsilon
        self.alpha = LearningRate
        self.gamma = DiscountFactor

    def reset(self):
        self.currentState = np.zeros(9)
        self.previousState = np.zeros(9)
        self.previousAction = None

    def actionTake(self, State):
        state = State.copy()
        available = np.where(state == 0)[0]

        if len(available) == 0:
            return state

        if np.random.uniform(0, 1) < self.epsilon:
            # Exploration: random action
            action = np.random.choice(available)
        else:
            # Exploitation: choose the best action based on Q-table
            q_values = self.q_table[tuple(state.astype(int))]
            action = available[np.argmax(q_values[available])]

        state[action] = self.index
        self.previousState = State.copy()
        self.previousAction = action
        return state

    def valueUpdate(self, State, Reward=0):
        currentState = State.copy()
        q_current = self.q_table[tuple(self.previousState.astype(int)) + (self.previousAction,)]
        action_values = self.q_table[tuple(currentState.astype(int))]
        q_next_max = np.max(action_values)

        # Q-learning update
        self.q_table[tuple(self.previousState.astype(int)) + (self.previousAction,)] = q_current + self.alpha * (
            Reward + self.gamma * q_next_max - q_current
        )

    def isWin(self, State):
        state = State.copy().reshape([3, 3])
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


def test(agent1, agent2, e1, e2, times=500):
    times = times
    agent1.epsilon = e1
    agent2.epsilon = e2

    competition_result = np.zeros(times)

    for i in range(times):
        print(f"battle {i}:")
        # 假设agent1先手
        state = np.zeros(9)

        while True:
            state = agent1.actionTake(state)
            # 判断agent1否胜利
            if agent1.isWin(state):
                competition_result[i] = 1
                agent1.valueUpdate(state, Reward=1)
                agent2.valueUpdate(agent2.currentState, Reward=-1)
                print("Agent1 Win")
                break
            else:
                # 输或和棋
                agent1.valueUpdate(state, Reward=0)

            # 是否是和棋
            if np.where(state == 0)[0].size == 0:
                agent1.valueUpdate(state, Reward=0)
                agent2.valueUpdate(agent2.currentState, Reward=0)
                competition_result[i] = 0
                print("Draw")
                break

            # 换agent2下棋
            state = agent2.actionTake(state)
            if agent2.isWin(state):
                competition_result[i] = -1
                agent1.valueUpdate(agent1.currentState, Reward=-1)
                agent2.valueUpdate(state, Reward=1)
                print("Agent2 Win")
                break
            else:
                agent2.valueUpdate(state, Reward=0)

        agent1.reset()
        agent2.reset()

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
    plt.title("q-learning agent win rate")
    plt.xlabel("step")
    plt.ylabel("win rate")

    plt.show()
    plt.savefig("./pics/qlearning_agent_win_rate.png")


def play(agent, index=-1):
    # 玩家和agent对战
    agent.epsilon = 0  # 禁用agent探索
    agent_player = Agent(index)
    state = np.zeros(9)
    while True:
        state = agent.actionTake(state)
        print(state.reshape(3, 3))
        if agent.isWin(state):
            print("Agent Win")
            break
        if np.where(state == 0)[0].size == 0:
            print("Draw")
            break

        position = input("Please input the position: ")
        state[int(position)] = -1
        if agent_player.isWin(state):
            print("You win")
            break


if __name__ == "__main__":
    times = 40000
    agent1 = Agent(1, Epsilon=0.2, LearningRate=0.2)
    agent2 = Agent(-1, Epsilon=0.2, LearningRate=0.2)

    # 人机 vs 人机 进行学习
    test(agent1, agent2, 0.2, 0.2, times)

    # 人机对战
    play(agent1)
