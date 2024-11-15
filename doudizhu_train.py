#%%
import torch
import numpy as np
from rlcard.envs.doudizhu import DoudizhuEnv
from rlcard.agents.dqn_agent import DQNAgent

# Training config
training_episodes = 10000
evaluate_every = 100
evaluate_num = 100

# Initialize environment
env = DoudizhuEnv(config={
    'seed': 42,
    'allow_step_back': False
})

# Create agents
landlord_agent = DQNAgent(
    num_actions=env.num_actions,
    state_shape=env.state_shape[0],
    mlp_layers=[512, 512],
    replay_memory_size=20000,
    replay_memory_init_size=1000,  # Wait for 1000 samples before training
    update_target_estimator_every=1000,
    batch_size=32,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
)

peasant_agent = DQNAgent(
    num_actions=env.num_actions,
    state_shape=env.state_shape[1],
    mlp_layers=[512, 512],
    replay_memory_size=20000,
    replay_memory_init_size=1000,
    update_target_estimator_every=1000,
    batch_size=32,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
)

agents = [landlord_agent, peasant_agent, peasant_agent]

# Training loop
print("Starting training...")
for episode in range(training_episodes):
    # Initialize game state
    state, player_id = env.reset()
    episode_rewards = []
    while True:
        # Agent takes action
        current_agent = landlord_agent if player_id == 0 else peasant_agent
        action = current_agent.step(state)
        
        # Environment step
        next_state, next_player_id = env.step(action)
        
        # Get reward and done flag
        reward = 0.0
        if env.is_over():
            reward = env.get_payoffs()[player_id]
            episode_rewards.append(reward)
            done = True
        else:
            done = False
            
        # Process state for memory
        obs = np.array(state['obs'], dtype=np.float32)
        next_obs = np.array(next_state['obs'], dtype=np.float32)
        
        # Store transition in memory
        transition = [obs, action, reward, next_obs, done]
        current_agent.feed_memory(obs, action, reward, next_obs, 
                                list(next_state['legal_actions'].keys()), done)

        # Only train after init_size samples
        if current_agent.total_t > current_agent.replay_memory_init_size:
            current_agent.train()
        
        if done:
            break
            
        state = next_state
        player_id = next_player_id

    # Evaluation phase
    if episode % evaluate_every == 0:
        print(f"\nEpisode {episode}")
        rewards = []
        for _ in range(evaluate_num):
            state, player_id = env.reset()
            done = False
            while not done:
                current_agent = landlord_agent if player_id == 0 else peasant_agent
                action, _ = current_agent.eval_step(state)
                state, player_id = env.step(action)
                done = env.is_over()
            rewards.append(env.get_payoffs())
        
        mean_reward = np.mean(rewards, axis=0)
        print(f'Average rewards: Landlord: {mean_reward[0]:.3f}, '
              f'Peasant 1: {mean_reward[1]:.3f}, Peasant 2: {mean_reward[2]:.3f}')

# Save trained agents
# landlord_agent.save_checkpoint('./checkpoints', 'landlord.pt')
# peasant_agent.save_checkpoint('./checkpoints', 'peasant.pt')