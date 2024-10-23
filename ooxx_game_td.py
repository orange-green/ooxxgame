import numpy as np
import matplotlib.pyplot as plt


# 时序分差（TD）,是一种广义的强化学习方法。通过估计状态值（state values）来学习，即 V 值,可以用于策略评估和策略改进
class Agent:
    def __init__(self, OOXX_index, Epsilon=0.1, LearningRate=0.1):
        self.value = np.zeros((3, 3, 3, 3, 3, 3, 3, 3, 3))
        self.currentState = np.zeros(9)
        self.previousState = np.zeros(9)
        self.index = OOXX_index
        self.epsilon = Epsilon
        self.alpha = LearningRate

    def reset(self):
        self.currentState = np.zeros(9)
        self.previousState = np.zeros(9)

    def actionTake(self, State) -> None:
        state = State.copy()
        available = np.where(state == 0)[0]
        length = len(available)

        if length == 0:
            return state
        else:
            random = np.random.uniform(0, 1)
            # 进行随机动作探索
            if random < self.epsilon:
                choose = np.random.randint(length)
                state[available[choose]] = self.index

            else:
                tempValue = np.zeros(length)
                for i in range(length):
                    tempState = state.copy()
                    tempState[available[i]] = self.index
                    tempValue[i] = self.value[tuple(tempState.astype(int))]

                choose = np.where(tempValue == np.max(tempValue))[0]
                chooseIndex = np.random.randint(len(choose))
                state[available[choose[chooseIndex]]] = self.index

            return state

    def valueUpdate(self, State, Reward=0) -> None:
        self.currentState = State.copy()
        self.value[tuple(self.previousState.astype(int))] = self.value[tuple(self.previousState.astype(int))] + self.alpha * (
            self.value[tuple(self.currentState.astype(int))] - self.value[tuple(self.previousState.astype(int))] + Reward
        )
        self.previousState = self.currentState.copy()

    def isWin(self, State):
        state = State.copy()
        state = state.reshape([3, 3])
        for i in range(0, 3):
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
    plt.title("TD agent win rate")
    plt.xlabel("step")
    plt.ylabel("win rate")

    plt.show()
    plt.savefig("./pics/TD_agent_win_rate.png")


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
