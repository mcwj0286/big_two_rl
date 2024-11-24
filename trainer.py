# trainer.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ppo_gameplay_dataset import PPOGameplayDataset, collate_fn
from models.decision_transformer_original import DecisionTransformer
import torch.optim as optim
# Import ModelEvaluator
from evaluate_model import ModelEvaluator
def train_decision_transformer(
    hdf5_path,
    state_dim,
    act_dim,
    n_blocks=6,
    h_dim=1024,
    n_heads=8,
    drop_p=0.1,
    max_timestep=1000,
    batch_size=16,
    learning_rate=1e-4,
    max_epochs=10,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    # device = 'cpu',
    save_model_path='output/decision_transformer.pt'
):
    # Initialize Dataset and DataLoader
    dataset = PPOGameplayDataset(hdf5_path)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=1  # Adjust based on your system
    )

    # Initialize the Decision Transformer model
    model = DecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        n_blocks=n_blocks,
        h_dim=h_dim,
        context_len=30,  
        n_heads=n_heads,
        drop_p=drop_p,
        max_timestep=max_timestep
    ).to(device)
    # Initialize ModelEvaluator
    evaluator = ModelEvaluator(
        model=model,
        state_dim=state_dim,
        act_dim=act_dim,
        ppo_model_path='output/modelParameters_best.pt',
        device=device
    )
    # Define optimizer and learning rate scheduler
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Define loss function
    # criterion = nn.CrossEntropyLoss(ignore_index=-1)  # Ignore padding index
    # criterion = nn.BCEWithLogitsLoss()
    # Define loss function
    criterion = nn.MSELoss()
    # Training loop
    model.train()
    best_win_rate = 0
    best_avg_reward = float('-inf')
    for epoch in range(max_epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            # Move data to device
            states = batch['states'].to(device)           # Shape: (B, T, state_dim)
            actions = batch['actions'].to(device)         # Shape: (B, T , act_dim)
            rewards = batch['rewards'].to(device)         # Shape: (B, T)
            timesteps = batch['timesteps'].to(device)     # Shape: (B, T)
            attention_mask = batch['attention_mask'].to(device)  # Shape: (B, T)

            # # Compute returns-to-go
            returns_to_go = torch.flip(torch.cumsum(torch.flip(rewards, dims=[1]), dim=1), dims=[1])
            returns_to_go = returns_to_go.unsqueeze(-1)  # Shape: (B, T, 1)

            # use sparse reward
            # rewards = rewards.unsqueeze(-1)  # Shape: (B, T, 1)

            # Forward pass
            action_preds = model(
                timesteps=timesteps,
                states=states,
                actions=actions,
                returns_to_go=returns_to_go,
                # attention_mask=attention_mask
            )

            action_preds = action_preds  # Detach to avoid backpropagating through the model
              # Output shape: (B, T, act_dim)
            # print(f"Action preds shape: {action_preds.shape}")
            # Reshape for loss computation
            B, T, act_dim = action_preds.shape
            action_preds = action_preds.view(B * T, act_dim)
            actions_target = actions.view(B * T, act_dim)

            # Apply attention mask
            attention_mask = attention_mask.view(B * T).bool() 
            
            action_preds_valid = action_preds[attention_mask]
            actions_target_valid = actions_target[attention_mask]
            # print(f'check masking')
            # print(action_preds_valid[-1])
            # print(actions_target_valid[-1])
            # print(f"Action preds valid shape: {action_preds_valid.shape}")
            # print(f"Actions target valid shape: {actions_target_valid.shape}")
            # # Compute loss
            loss = criterion(action_preds_valid, actions_target_valid)
            # print(f"Loss: {loss.item()}")
            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{max_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{max_epochs}] Average Loss: {avg_loss:.4f}")

        # # Perform validation after each epoch
        model.eval()
        win_rate, avg_reward = evaluator.evaluate_game(num_games=100)  # Adjust num_games as needed
        model.train()
        print(f"Validation Metrics - Win Rate: {win_rate:.2f}%, Avg. Reward: {avg_reward:.2f}")
        # Check if both win rate and rewards have improved
        if win_rate > best_win_rate and avg_reward > best_avg_reward:
            best_win_rate = win_rate
            best_avg_reward = avg_reward
            # Save the current model as the best model
            torch.save(model.state_dict(), save_model_path)
            print(f"New best model saved with win rate: {win_rate:.2f}% and avg reward: {avg_reward:.2f}")
        else:
            print(f"No improvement in validation metrics.")


    # Save the trained model
    torch.save(model.state_dict(), save_model_path)
    print("Training complete. Model saved.")

    # Close the dataset
    dataset.close()

if __name__ == "__main__":
    hdf5_path = 'output/pytorch_ppo_trajectories.hdf5'  # Replace with your HDF5 file path
    state_dim = 412  # Update based on your data
    act_dim = 1695   # Update based on your action space

    train_decision_transformer(
        hdf5_path=hdf5_path,
        state_dim=state_dim,
        act_dim=act_dim,
        max_epochs=200,
        batch_size=64,
        learning_rate=1e-5
    )