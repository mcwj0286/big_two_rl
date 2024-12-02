# trainer.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ppo_gameplay_dataset import PPOGameplayDataset, collate_fn
# from models.decision_transformer_original import DecisionTransformer
from models.decision_transformer import DecisionTransformer

import torch.optim as optim
# Remove the import of GradScaler and autocast
# from torch.cuda.amp import GradScaler, autocast  # Removed
# Import ModelEvaluator
from evaluate_model import ModelEvaluator

def train_decision_transformer(
    hdf5_path,
    state_dim,
    act_dim,
    n_blocks=6,
    h_dim=128,
    n_heads=8,
    drop_p=0.1,
    max_timestep=1000,
    batch_size=16,
    learning_rate=1e-4,
    max_epochs=10,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    # device = 'cpu',
    save_model_path='output/official_decision_transformer.pt',
    patience=30,  # Added for early stopping
    gradient_clip=1.0  # Added for gradient clipping
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
        hidden_size=h_dim,
        max_ep_len=90,
        seq_len=30,  
        max_length=max_timestep,
        action_tanh = False
    ).to(device)
    # Print the number of parameters in the model
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters in the decision transformer: {total_params}")
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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)  # Added learning rate scheduler

    # Define loss function
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    # Remove the initialization of GradScaler
    # scaler = GradScaler()  # Removed

    # Training loop
    model.train()
    best_win_rate = 0
    best_avg_reward = float('-inf')
    epochs_no_improve = 0  # Added for early stopping
    for epoch in range(max_epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            # Move data to device with correct dtype
            states = batch['states'].to(device).float()
            actions = batch['actions'].to(device).float()  # Ensure actions are float
            rewards = batch['rewards'].to(device).float()
            timesteps = batch['timesteps'].to(device).long()
            attention_mask = batch['attention_mask'].to(device).float()
            action_one_hot = batch['actions_one_hot'].to(device).float()
            # Compute returns-to-go
            returns_to_go = torch.flip(torch.cumsum(torch.flip(rewards, dims=[1]), dim=1), dims=[1])
            returns_to_go = returns_to_go.unsqueeze(-1)  # Shape: (B, T, 1)
             # Add validation checks
            print(f"States shape: {states.shape}, range: [{states.min():.2f}, {states.max():.2f}]")
            print(f"Actions shape: {action_one_hot.shape}, range: [{action_one_hot.min():.2f}, {action_one_hot.max():.2f}]")
            print(f"Returns shape: {returns_to_go.shape}, range: [{returns_to_go.min():.2f}, {returns_to_go.max():.2f}]")
            print(f"Timesteps shape: {timesteps.shape}, range: [{timesteps.min()}, {timesteps.max()}]")

            # sparse rewards
            # rewards = rewards.unsqueeze(-1)  # Shape: (B, T, 1)

            # Forward pass without autocast
            optimizer.zero_grad()
            _, action_preds, _ = model(
                timesteps=timesteps,
                states=states,
                actions=action_one_hot,
                returns_to_go=returns_to_go,
                attention_mask=attention_mask.bool()
            )
            
            B, T, act_dim = action_preds.shape
            action_preds = action_preds.view(B * T, act_dim)      # Shape: (B*T, act_dim)
            actions_target = actions.view(B * T)                  # Shape: (B*T)

            # Apply attention mask
            attention_mask_flat = attention_mask.view(B * T).bool()
            action_preds_valid = action_preds[attention_mask_flat]
            actions_target_valid = actions_target[attention_mask_flat]
            # Debug prints
            print(f"Shapes after reshape:")
            print(f"action_preds: {action_preds.shape}")
            print(f"actions_target: {actions_target.shape}")
            print(f"attention_mask_flat: {attention_mask_flat.shape}")


            # Filter out padded action targets (equal to act_dim)
            valid_indices = actions_target_valid != act_dim
            action_preds_valid = action_preds_valid[valid_indices]
            actions_target_valid = actions_target_valid[valid_indices]
            if action_preds_valid.size(0) == 0:
                continue  # Skip this batch if no valid data

            # Compute loss
            loss = criterion(action_preds_valid, actions_target_valid)

            # Backpropagation without gradient scaling
            loss.backward()
            # Gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{max_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{max_epochs}] Average Loss: {avg_loss:.4f}")

        # Perform validation after each epoch
        model.eval()
        win_rate, avg_reward = evaluator.evaluate_game(num_games=100)  # Adjust num_games as needed
        model.train()
        print(f"Validation Metrics - Win Rate: {win_rate:.2f}%, Avg. Reward: {avg_reward:.2f}")

        # Scheduler step based on win_rate
        scheduler.step(win_rate)

        # Check if both win rate and rewards have improved
        if avg_reward > best_avg_reward:
            best_win_rate = win_rate
            best_avg_reward = avg_reward
            epochs_no_improve = 0  # Reset counter
            # Save the current model as the best model
            torch.save(model.state_dict(), save_model_path)
            print(f"New best model saved with win rate: {win_rate:.2f}% and avg reward: {avg_reward:.2f}")
        else:
            epochs_no_improve += 1
            print(f"No improvement in validation metrics. Early stopping counter: {epochs_no_improve}/{patience}")
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break  # Exit training loop

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
        batch_size=16,
        learning_rate=5e-6
    )