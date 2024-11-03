import numpy as np
import torch
import joblib
import time

# Import your environment and PPO model classes
from game.big2Game import vectorizedBig2Games
from models.PPONetwork_pytorch import PPONetwork, PPOModel

# Hyperparameters
obs_dim = 412          # Observation dimension
act_dim = 1695         # Action dimension
n_games = 64           # Number of parallel games
n_steps = 20           # Steps per training update
total_steps = 1_000_000  # Total training steps
gamma = 0.995          # Discount factor
lam = 0.95             # GAE lambda
learning_rate = 2.5e-4
min_learning_rate = 1e-6
clip_range = 0.2
ent_coef = 0.01        # Entropy coefficient
vf_coef = 0.5          # Value function coefficient
max_grad_norm = 0.5
n_minibatches = 4      # Number of minibatches
n_opt_epochs = 5       # Number of optimization epochs
save_every = 500       # Model save frequency
reward_normalization = 5.0  # Reward scaling factor

# Initialize environment and PPO model
env = vectorizedBig2Games(n_games)
network = PPONetwork(obs_dim, act_dim)
ppo_model = PPOModel(
    network,
    ent_coef=ent_coef,
    vf_coef=vf_coef,
    max_grad_norm=max_grad_norm,
    learning_rate=learning_rate
)

# Helper function to reshape arrays
def flatten_and_swap(arr):
    shape = arr.shape
    return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

# Training loop
ep_infos = []
losses = []
games_completed = 0
updates = total_steps // (n_games * n_steps)
start_time = time.time()

for update in range(1, updates + 1):
    # Adjust learning rate and clip range
    progress = 1 - (update - 1) / updates
    lr_now = max(learning_rate * progress, min_learning_rate)
    clip_range_now = clip_range * progress

    # Storage for batch data
    mb_obs = []
    mb_actions = []
    mb_values = []
    mb_neglogpacs = []
    mb_rewards = []
    mb_dones = []
    mb_avail_acs = []

    # Collect experience
    for _ in range(n_steps):
        # Get current states and available actions
        curr_gos, curr_states, curr_avail_acs = env.getCurrStates()
        curr_states = curr_states.squeeze()
        curr_avail_acs = curr_avail_acs.squeeze()

        # Get actions and values from the network
        actions, values, neglogpacs = network.act(curr_states, curr_avail_acs)
        rewards, dones, infos = env.step(actions)

        # Store data
        mb_obs.append(curr_states)
        mb_actions.append(actions)
        mb_values.append(values)
        mb_neglogpacs.append(neglogpacs)
        mb_rewards.append(rewards / reward_normalization)
        mb_dones.append(dones)
        mb_avail_acs.append(curr_avail_acs)

        # Handle completed games
        for idx, done in enumerate(dones):
            if done:
                games_completed += 1
                ep_infos.append(infos[idx])
                print(f"Game {games_completed} finished. Duration: {infos[idx]['numTurns']} turns")

    # Convert batch data to arrays
    mb_obs = np.array(mb_obs, dtype=np.float32)
    mb_actions = np.array(mb_actions, dtype=np.int64)
    mb_values = np.array(mb_values, dtype=np.float32)
    mb_neglogpacs = np.array(mb_neglogpacs, dtype=np.float32)
    mb_rewards = np.array(mb_rewards, dtype=np.float32)
    mb_dones = np.array(mb_dones, dtype=bool)
    mb_avail_acs = np.array(mb_avail_acs, dtype=np.float32)

    # Compute advantages and returns using GAE
    mb_returns = np.zeros_like(mb_rewards)
    mb_advs = np.zeros_like(mb_rewards)
    last_gae_lam = 0
    for t in reversed(range(n_steps)):
        if t == n_steps - 1:
            next_non_terminal = 1.0 - mb_dones[-1]
            next_values = mb_values[-1]
        else:
            next_non_terminal = 1.0 - mb_dones[t + 1]
            next_values = mb_values[t + 1]
        delta = mb_rewards[t] + gamma * next_values * next_non_terminal - mb_values[t]
        mb_advs[t] = last_gae_lam = delta + gamma * lam * next_non_terminal * last_gae_lam
    mb_returns = mb_advs + mb_values

    # Flatten batches
    mb_obs = flatten_and_swap(mb_obs)
    mb_actions = flatten_and_swap(mb_actions)
    mb_values = flatten_and_swap(mb_values)
    mb_neglogpacs = flatten_and_swap(mb_neglogpacs)
    mb_returns = flatten_and_swap(mb_returns)
    mb_advs = flatten_and_swap(mb_advs)
    mb_avail_acs = flatten_and_swap(mb_avail_acs)

    # Training step
    batch_size = mb_obs.shape[0]
    minibatch_size = batch_size // n_minibatches
    indices = np.arange(batch_size)
    mb_loss_vals = []

    for epoch in range(n_opt_epochs):
        np.random.shuffle(indices)
        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            minibatch_indices = indices[start:end]

            # Extract minibatch
            batch_obs = torch.tensor(mb_obs[minibatch_indices], dtype=torch.float32)
            batch_actions = torch.tensor(mb_actions[minibatch_indices], dtype=torch.int64)
            batch_returns = torch.tensor(mb_returns[minibatch_indices], dtype=torch.float32)
            batch_values = torch.tensor(mb_values[minibatch_indices], dtype=torch.float32)
            batch_neglogpacs = torch.tensor(mb_neglogpacs[minibatch_indices], dtype=torch.float32)
            batch_advs = torch.tensor(mb_advs[minibatch_indices], dtype=torch.float32)
            batch_avail_acs = torch.tensor(mb_avail_acs[minibatch_indices], dtype=torch.float32)

            # Update the PPO model
            pg_loss, vf_loss, entropy_loss = ppo_model.train(
                clip_range_now,
                batch_obs,
                batch_avail_acs,
                batch_returns,
                batch_actions,
                batch_values,
                batch_neglogpacs,
                batch_advs
            )
            mb_loss_vals.append([pg_loss, vf_loss, entropy_loss])

    # Record losses
    loss_values = np.mean(mb_loss_vals, axis=0)
    losses.append(loss_values)

    # Save model and logs periodically
    if update % save_every == 0:
        model_path = f"output/modelParameters{update}.pt"
        torch.save(network.state_dict(), model_path)
        joblib.dump(losses, "output/losses.pkl")
        joblib.dump(ep_infos, "output/epInfos.pkl")
        print(f"Update {update}: Model saved.")

    # Optional: Print progress
    if update % 10 == 0:
        print(f"Update {update}/{updates} completed.")

end_time = time.time()
print(f"Training completed in {end_time - start_time:.2f} seconds.")