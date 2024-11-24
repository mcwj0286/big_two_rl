# mainBig2PPOSimulation.py

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # Add current directory to system path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import copy
from models.PPONetwork_pytorch import PPONetwork, PPOModel  # Import PPO network and model
from game.big2Game import vectorizedBig2Games  # Import vectorized game environment

# Helper function to reshape minibatch for training
def sf01(arr):
    """
    Swap and then flatten axes 0 and 1.
    
    Args:
        arr (np.ndarray): Input array with shape (a, b, ...).
    
    Returns:
        np.ndarray: Reshaped array with shape (a*b, ...).
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])

class big2PPOSimulation(object):
    def __init__(
        self,
        inp_dim=412,            # Dimension of input features
        n_games=8,              # Number of parallel games
        n_steps=20,             # Number of steps per game
        n_minibatches=4,        # Number of minibatches for training
        n_opt_epochs=5,         # Number of optimization epochs
        lam=0.95,               # GAE lambda
        gamma=0.995,            # Discount factor
        ent_coef=0.01,          # Entropy coefficient
        vf_coef=0.5,            # Value function coefficient
        max_grad_norm=0.5,      # Maximum gradient norm for clipping
        min_learning_rate=1e-6, # Minimum learning rate
        learning_rate=2.5e-4,   # Initial learning rate
        clip_range=0.2,         # Clipping range for PPO
        save_every=500,         # Save model every 'n' updates
    ):
        # Initialize PPO network and model for training
        self.training_network = PPONetwork(inp_dim, 1695)  # PPONetwork with input dim and action space size
        self.training_model = PPOModel(
            self.training_network,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            learning_rate=learning_rate,
        )

        # Assign the same training network to all players
        self.player_networks = {
            1: self.training_network,
            2: self.training_network,
            3: self.training_network,
            4: self.training_network,
        }
        self.train_on_player = [True, True, True, True]  # Flags to indicate training on each player

        # Initialize the vectorized game environment
        self.vectorized_game = vectorizedBig2Games(n_games)

        # Set training parameters
        self.n_games = n_games
        self.inp_dim = inp_dim
        self.n_steps = n_steps
        self.n_minibatches = n_minibatches
        self.n_opt_epochs = n_opt_epochs
        self.lam = lam
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.clip_range = clip_range
        self.save_every = save_every

        self.reward_normalization = 5.0  # Normalize rewards by this factor

        # Initialize training statistics and buffers
        self.tot_training_steps = 0
        self.ep_infos = []      # Episode information
        self.games_done = 0     # Counter for completed games
        self.losses = []        # List to store loss values

        # Buffers to store previous observations and other related data
        self.prev_obs = []
        self.prev_gos = []
        self.prev_avail_acs = []
        self.prev_rewards = []
        self.prev_actions = []
        self.prev_values = []
        self.prev_dones = []
        self.prev_neglogpacs = []

    def run(self):
        """
        Run vectorized games for n_steps and generate minibatch for training.
        
        Returns:
            map: Reshaped and flattened minibatch data.
        """
        # Initialize minibatch containers
        mb_obs, mb_pGos, mb_actions, mb_values = [], [], [], []
        mb_neglogpacs, mb_rewards, mb_dones, mb_availAcs = [], [], [], []

        # Append previous observations and related data to minibatch
        for i in range(len(self.prev_obs)):
            mb_obs.append(self.prev_obs[i])
            mb_pGos.append(self.prev_gos[i])
            mb_actions.append(self.prev_actions[i])
            mb_values.append(self.prev_values[i])
            mb_neglogpacs.append(self.prev_neglogpacs[i])
            mb_rewards.append(self.prev_rewards[i])
            mb_dones.append(self.prev_dones[i])
            mb_availAcs.append(self.prev_avail_acs[i])

        # Determine how many steps to use based on previous observations
        if len(self.prev_obs) == 4:
            end_length = self.n_steps
        else:
            end_length = self.n_steps - 4

        # Run the simulation for n_steps
        for _ in range(self.n_steps):
            # Get current game states, available actions from the environment
            curr_gos, curr_states, curr_avail_acs = self.vectorized_game.getCurrStates()
            print(f'curr_gos: {curr_gos.shape},curr_states: {curr_states.shape}, curr_avail_acs: {curr_avail_acs.shape}')
            curr_states = np.squeeze(curr_states)
            curr_avail_acs = np.squeeze(curr_avail_acs)
            curr_gos = np.squeeze(curr_gos)

            # Get actions, values, and neg log probabilities from the network
            actions, values, neglogpacs = self.training_network.act(
                curr_states, curr_avail_acs
            )
            # Execute actions in the environment
            rewards, dones, infos = self.vectorized_game.step(actions)

            # Collect data for minibatch
            mb_obs.append(curr_states.copy())
            mb_pGos.append(curr_gos)
            mb_availAcs.append(curr_avail_acs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(list(dones))

            # Assign rewards appropriately if the state is terminal
            to_append_rewards = np.zeros((self.n_games,))
            mb_rewards.append(to_append_rewards)
            for i in range(self.n_games):
                if dones[i]:
                    reward = rewards[i]
                    mb_rewards[-1][i] = reward[mb_pGos[-1][i] - 1] / self.reward_normalization
                    mb_rewards[-2][i] = reward[mb_pGos[-2][i] - 1] / self.reward_normalization
                    mb_rewards[-3][i] = reward[mb_pGos[-3][i] - 1] / self.reward_normalization
                    mb_rewards[-4][i] = reward[mb_pGos[-4][i] - 1] / self.reward_normalization
                    # Mark previous steps as done
                    mb_dones[-2][i] = True
                    mb_dones[-3][i] = True
                    mb_dones[-4][i] = True
                    self.ep_infos.append(infos[i])  # Store episode info
                    self.games_done += 1
                    # print(f"Game {self.games_done} finished. Lasted {infos[i]['numTurns']} turns")

        # Update previous observations and related data with the latest
        self.prev_obs = mb_obs[end_length:]
        self.prev_gos = mb_pGos[end_length:]
        self.prev_rewards = mb_rewards[end_length:]
        self.prev_actions = mb_actions[end_length:]
        self.prev_values = mb_values[end_length:]
        self.prev_dones = mb_dones[end_length:]
        self.prev_neglogpacs = mb_neglogpacs[end_length:]
        self.prev_avail_acs = mb_availAcs[end_length:]

        # Convert lists to numpy arrays and truncate to end_length
        mb_obs = np.asarray(mb_obs, dtype=np.float32)[:end_length]
        mb_availAcs = np.asarray(mb_availAcs, dtype=np.float32)[:end_length]
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)[:end_length]
        mb_actions = np.asarray(mb_actions, dtype=np.int64)[:end_length]
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)[:end_length]
        mb_dones = np.asarray(mb_dones, dtype=np.bool_)

        # Initialize arrays for returns and advantages
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)

        # Compute Generalized Advantage Estimation (GAE) for multiple steps
        for k in range(4):
            lastgaelam = 0
            for t in reversed(range(k, end_length, 4)):
                nextnonterminal = 1.0 - mb_dones[t]
                nextvalues = mb_values[t + 4]
                delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
                mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam

        mb_values = mb_values[:end_length]
        mb_returns = mb_advs + mb_values  # Calculate returns

        # Return the reshaped minibatch data
        return map(sf01, (mb_obs, mb_availAcs, mb_returns, mb_actions, mb_values, mb_neglogpacs))

    def train(self, n_total_steps):
        """
        Train the PPO model for a total number of steps.
        
        Args:
            n_total_steps (int): Total number of training steps.
        """
        n_updates = n_total_steps // (self.n_games * self.n_steps)  # Calculate number of updates

        for update in range(n_updates):
            # Update learning rate and clipping range with linear decay
            alpha = 1.0 - update / n_updates
            lrnow = self.learning_rate * alpha
            if lrnow < self.min_learning_rate:
                lrnow = self.min_learning_rate
            cliprangenow = self.clip_range * alpha

            # Run simulation to get minibatch data
            (
                states,
                avail_acs,
                returns,
                actions,
                values,
                neglogpacs,
            ) = self.run()

            batch_size = states.shape[0]  # Total batch size
            self.tot_training_steps += batch_size  # Update total training steps

            n_training_batch = batch_size // self.n_minibatches  # Size of each minibatch

            mb_lossvals = []
            inds = np.arange(batch_size)  # Indices for shuffling

            # Perform multiple optimization epochs
            for _ in range(self.n_opt_epochs):
                np.random.shuffle(inds)  # Shuffle indices for each epoch
                for start in range(0, batch_size, n_training_batch):
                    end = start + n_training_batch
                    mb_inds = inds[start:end]  # Get minibatch indices

                    # Prepare minibatch data as PyTorch tensors
                    mb_states = torch.tensor(states[mb_inds], dtype=torch.float32)
                    mb_avail_acs = torch.tensor(avail_acs[mb_inds], dtype=torch.float32)
                    mb_returns = torch.tensor(returns[mb_inds], dtype=torch.float32)
                    mb_actions = torch.tensor(actions[mb_inds], dtype=torch.int64)
                    mb_values = torch.tensor(values[mb_inds], dtype=torch.float32)
                    mb_neglogpacs = torch.tensor(neglogpacs[mb_inds], dtype=torch.float32)

                    # # Log minibatch shapes for debugging
                    # print(f"Minibatch states: {mb_states.shape}")
                    # print(f"Minibatch available actions: {mb_avail_acs.shape}")
                    # print(f"Minibatch returns: {mb_returns.shape}")
                    # print(f"Minibatch actions: {mb_actions.shape}")
                    # print(f"Minibatch values: {mb_values.shape}")
                    # print(f"Minibatch neglogpacs: {mb_neglogpacs.shape}")

                    # Update the PPO model with the minibatch
                    pg_loss, vf_loss, entropy_loss = self.training_model.train(
                        lrnow,
                        cliprangenow,
                        mb_states,
                        mb_avail_acs,
                        mb_returns,
                        mb_actions,
                        mb_values,
                        mb_neglogpacs,
                    )
                    mb_lossvals.append([pg_loss, vf_loss, entropy_loss])
                   

            # Calculate mean loss over minibatches
            lossvals = np.mean(mb_lossvals, axis=0)
            self.losses.append(lossvals)  # Store loss values
            print(f"Update: {update}, Loss: {lossvals}")
            # Save the model and training stats periodically
            if len(self.losses) > 1 and lossvals[0] < self.losses[-2][0]:
                name = f"output/modelParameters_best.pt"
                torch.save(self.training_network.state_dict(), name)
                joblib.dump(self.losses, "output/losses.pkl")
                joblib.dump(self.ep_infos, "output/epInfos.pkl")
            # if update % self.save_every == 0:
            #     name = f"output/modelParameters_best.pt"
            #     torch.save(self.training_network.state_dict(), name)  # Save model parameters
            #     joblib.dump(self.losses, "output/losses.pkl")        # Save loss history
            #     joblib.dump(self.ep_infos, "output/epInfos.pkl")    # Save episode info

if __name__ == "__main__":
    import time

    # Initialize the PPO simulation with specified parameters
    main_sim = big2PPOSimulation(n_games=64, n_steps=20, learning_rate=2.5e-4, clip_range=0.2)
    start = time.time()  # Start timer
    main_sim.train(100000)  # Begin training for a large number of steps
    end = time.time()  # End timer
    print(f"Time Taken: {end - start}")  # Print total training time