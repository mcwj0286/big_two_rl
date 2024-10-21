# main.py

from env.big_two_env import BigTwoEnv

def main():
    env = BigTwoEnv()
    state = env.reset()
    done = False
    while not done:
        # For testing, we'll have the agent take random actions
        action = env.action_space.sample()  # Replace with agent's action selection
        state, reward, done, info = env.step(action)
        env.render()
    print("Test game completed.")

if __name__ == "__main__":
    main()