# mainBig2PPOSimulation.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
print(sys.path)
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import copy
from models.PPONetwork_pytorch import PPONetwork, PPOModel
from game.big2Game import vectorizedBig2Games

# Helper function to reshape minibatch for training
def sf01(arr):
    """
    Swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])


class big2PPOSimulation(object):
    def __init__(
        self,
        inp_dim=412,
        n_games=8,
        n_steps=20,
        n_minibatches=4,
        n_opt_epochs=5,
        lam=0.95,
        gamma=0.995,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        min_learning_rate=1e-6,
        learning_rate=2.5e-4,
        clip_range=0.2,
        save_every=500,
    ):
        # Network/model for training
        self.training_network = PPONetwork(inp_dim, 1695)
        self.training_model = PPOModel(
            self.training_network,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            learning_rate=learning_rate,
        )

        # For now, each player uses the same (up-to-date) network to make decisions
        self.player_networks = {
            1: self.training_network,
            2: self.training_network,
            3: self.training_network,
            4: self.training_network,
        }
        self.train_on_player = [True, True, True, True]

        # Environment
        self.vectorized_game = vectorizedBig2Games(n_games)

        # Params
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

        self.reward_normalization = 5.0  # Divide rewards by this number

        # Episode/training information
        self.tot_training_steps = 0
        self.ep_infos = []
        self.games_done = 0
        self.losses = []

        # Final 4 observations need to be carried over
        self.prev_obs = []
        self.prev_gos = []
        self.prev_avail_acs = []
        self.prev_rewards = []
        self.prev_actions = []
        self.prev_values = []
        self.prev_dones = []
        self.prev_neglogpacs = []

    def run(self):
        # Run vectorized games for n_steps and generate minibatch for training
        mb_obs, mb_pGos, mb_actions, mb_values = [], [], [], []
        mb_neglogpacs, mb_rewards, mb_dones, mb_availAcs = [], [], [], []

        for i in range(len(self.prev_obs)):
            mb_obs.append(self.prev_obs[i])
            mb_pGos.append(self.prev_gos[i])
            mb_actions.append(self.prev_actions[i])
            mb_values.append(self.prev_values[i])
            mb_neglogpacs.append(self.prev_neglogpacs[i])
            mb_rewards.append(self.prev_rewards[i])
            mb_dones.append(self.prev_dones[i])
            mb_availAcs.append(self.prev_avail_acs[i])

        if len(self.prev_obs) == 4:
            end_length = self.n_steps
        else:
            end_length = self.n_steps - 4

        for _ in range(self.n_steps):
            curr_gos, curr_states, curr_avail_acs = self.vectorized_game.getCurrStates()
            curr_states = np.squeeze(curr_states)
            curr_avail_acs = np.squeeze(curr_avail_acs)
            curr_gos = np.squeeze(curr_gos)

            actions, values, neglogpacs = self.training_network.act(
                curr_states, curr_avail_acs
            )
            rewards, dones, infos = self.vectorized_game.step(actions)

            mb_obs.append(curr_states.copy())
            mb_pGos.append(curr_gos)
            mb_availAcs.append(curr_avail_acs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(list(dones))

            # Back assign rewards if state is terminal
            to_append_rewards = np.zeros((self.n_games,))
            mb_rewards.append(to_append_rewards)
            for i in range(self.n_games):
                if dones[i]:
                    reward = rewards[i]
                    mb_rewards[-1][i] = reward[mb_pGos[-1][i] - 1] / self.reward_normalization
                    mb_rewards[-2][i] = reward[mb_pGos[-2][i] - 1] / self.reward_normalization
                    mb_rewards[-3][i] = reward[mb_pGos[-3][i] - 1] / self.reward_normalization
                    mb_rewards[-4][i] = reward[mb_pGos[-4][i] - 1] / self.reward_normalization
                    mb_dones[-2][i] = True
                    mb_dones[-3][i] = True
                    mb_dones[-4][i] = True
                    self.ep_infos.append(infos[i])
                    self.games_done += 1
                    print(f"Game {self.games_done} finished. Lasted {infos[i]['numTurns']} turns")

        self.prev_obs = mb_obs[end_length:]
        self.prev_gos = mb_pGos[end_length:]
        self.prev_rewards = mb_rewards[end_length:]
        self.prev_actions = mb_actions[end_length:]
        self.prev_values = mb_values[end_length:]
        self.prev_dones = mb_dones[end_length:]
        self.prev_neglogpacs = mb_neglogpacs[end_length:]
        self.prev_avail_acs = mb_availAcs[end_length:]

        mb_obs = np.asarray(mb_obs, dtype=np.float32)[:end_length]
        mb_availAcs = np.asarray(mb_availAcs, dtype=np.float32)[:end_length]
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)[:end_length]
        mb_actions = np.asarray(mb_actions, dtype=np.int64)[:end_length]
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)[:end_length]
        mb_dones = np.asarray(mb_dones, dtype=np.bool_)

        # Discount/bootstrap value function with generalized advantage estimation
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        for k in range(4):
            lastgaelam = 0
            for t in reversed(range(k, end_length, 4)):
                nextnonterminal = 1.0 - mb_dones[t]
                nextvalues = mb_values[t + 4]
                delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
                mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam

        mb_values = mb_values[:end_length]
        mb_returns = mb_advs + mb_values

        return map(sf01, (mb_obs, mb_availAcs, mb_returns, mb_actions, mb_values, mb_neglogpacs))

    def train(self, n_total_steps):
        n_updates = n_total_steps // (self.n_games * self.n_steps)

        for update in range(n_updates):
            alpha = 1.0 - update / n_updates
            lrnow = self.learning_rate * alpha
            if lrnow < self.min_learning_rate:
                lrnow = self.min_learning_rate
            cliprangenow = self.clip_range * alpha

            (
                states,
                avail_acs,
                returns,
                actions,
                values,
                neglogpacs,
            ) = self.run()

            batch_size = states.shape[0]
            self.tot_training_steps += batch_size

            n_training_batch = batch_size // self.n_minibatches

            mb_lossvals = []
            inds = np.arange(batch_size)
            for _ in range(self.n_opt_epochs):
                np.random.shuffle(inds)
                for start in range(0, batch_size, n_training_batch):
                    end = start + n_training_batch
                    mb_inds = inds[start:end]

                    # Prepare minibatch data
                    mb_states = torch.tensor(states[mb_inds], dtype=torch.float32)
                    mb_avail_acs = torch.tensor(avail_acs[mb_inds], dtype=torch.float32)
                    mb_returns = torch.tensor(returns[mb_inds], dtype=torch.float32)
                    mb_actions = torch.tensor(actions[mb_inds], dtype=torch.int64)
                    mb_values = torch.tensor(values[mb_inds], dtype=torch.float32)
                    mb_neglogpacs = torch.tensor(neglogpacs[mb_inds], dtype=torch.float32)

                    # Update the PPO model
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

            lossvals = np.mean(mb_lossvals, axis=0)
            self.losses.append(lossvals)

            if update % self.save_every == 0:
                name = f"output/modelParameters{update}.pt"
                torch.save(self.training_network.state_dict(), name)
                joblib.dump(self.losses, "output/losses.pkl")
                joblib.dump(self.ep_infos, "output/epInfos.pkl")


if __name__ == "__main__":
    import time

    main_sim = big2PPOSimulation(n_games=64, n_steps=20, learning_rate=2.5e-4, clip_range=0.2)
    start = time.time()
    main_sim.train(1000000000)
    end = time.time()
    print(f"Time Taken: {end - start}")