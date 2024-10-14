# from game.game_engine import BigTwoGame

# from agent.rl_agent import RLAgentPlayer
# from agent.random_agent import RandomPlayer
from env.big_two_env import BigTwoEnv
def main():

    env = BigTwoEnv()
    state = env.reset()
    done = False
    while not done:
        # For testing, we'll have the agent take random actions
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        env.render()
    print("Test game completed.")

if __name__ == "__main__":
    main()