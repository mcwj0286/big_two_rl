# Project Report

## 1. Introduction

Big 2 is a strategic four-player card game originating from East Asia, known for its complex gameplay and diverse action space. The objective is to be the first player to discard all your cards by forming various combinations such as singles, pairs, straights, and full houses. Successfully mastering Big 2 requires not only tactical planning but also the ability to anticipate opponents' moves and adapt strategies dynamically.

This project focuses on developing an artificial intelligence agent capable of playing Big 2 at a competitive level using deep reinforcement learning techniques. Leveraging the Proximal Policy Optimization (PPO) algorithm, the AI is trained through self-play, allowing it to learn optimal strategies without human intervention. The implementation involves simulating numerous game trajectories, analyzing action distributions, and continuously refining the agent's decision-making processes.

The significance of this project lies in its exploration of applying advanced reinforcement learning algorithms to complex, imperfect-information games. By tackling the challenges posed by Big 2's intricate action space and multiplayer dynamics, the project aims to contribute to the broader field of game AI and provide insights into effective training methodologies for sophisticated strategic environments.

## 2. Background

Reinforcement learning (RL) is a branch of machine learning where agents learn optimal behaviors through interactions with an environment, receiving rewards or penalties based on their actions. Among various RL algorithms, Proximal Policy Optimization (PPO) has gained prominence due to its balance between performance and computational efficiency. PPO optimizes policies by ensuring updates remain within a safe region, preventing drastic policy changes that could destabilize learning.

Proximal Policy Optimization has been successfully applied to a range of games and simulation environments, demonstrating its ability to handle high-dimensional state and action spaces. Its effectiveness in continuous and discrete action settings makes it a suitable choice for complex games like Big 2.

Big 2 presents unique challenges for AI development. Unlike deterministic games, Big 2 involves imperfect information, where players do not have complete visibility of opponents' hands. This uncertainty necessitates strategies that account for various possible opponent actions and outcomes. Additionally, the game's combinatorial nature, with numerous possible card combinations and sequences, increases the complexity of decision-making processes for the AI agent.

Previous work in game AI has explored reinforcement learning in games like Poker and Bridge, which also feature imperfect information and multi-agent interactions. These studies provide foundational techniques and insights that can be adapted to the context of Big 2. However, the specific rules and action space of Big 2 require tailored approaches to effectively capture the strategic depth of the game.

In summary, the integration of PPO within the framework of Big 2 aims to address the dual challenges of imperfect information and a vast action space. By building upon established RL methodologies and adapting them to the specific demands of Big 2, this project seeks to advance the capabilities of game-playing AI agents in complex strategic environments.

### 2.1 Decision Transformer

Decision Transformer is an innovative approach that reimagines reinforcement learning by leveraging sequence modeling techniques, initially developed for natural language processing. Instead of relying solely on traditional value-based or policy-based methods, Decision Transformer formulates the decision-making process as a sequence prediction task. This involves treating the agent's interactions with the environment—comprising states, actions, and rewards—as a trajectory that can be modeled and predicted using Transformer architectures.

The core idea behind Decision Transformer is to utilize the self-attention mechanisms of Transformers to capture long-range dependencies and contextual information within the agent's experience. By processing entire trajectories, the model can learn to predict optimal actions based on the cumulative context of past states and actions, rather than making decisions based solely on the current state. This allows for more coherent and strategically aligned behavior, as the agent can consider the broader sequence of events leading up to each decision.

Key advantages of Decision Transformer include:

1. **Flexibility in Handling Various Tasks**: By framing RL as a sequence modeling problem, Decision Transformer can seamlessly adapt to different tasks and environments without the need for task-specific architectures.

2. **Enhanced Contextual Understanding**: The ability to process entire trajectories enables the model to incorporate historical context, leading to more informed and contextually appropriate action selections.

3. **Scalability**: Transformer-based models scale effectively with increasing amounts of data and complexity, making them well-suited for environments with extensive state and action spaces, such as Big 2.

In the context of this project, integrating Decision Transformer offers a complementary approach to PPO. While PPO focuses on optimizing policy updates within safe regions to ensure stable learning, Decision Transformer leverages sequence modeling to enhance the agent's ability to plan and execute complex strategies over extended periods. Combining these methodologies can potentially lead to a more robust and capable AI agent, capable of navigating the intricate and dynamic landscape of Big 2.

Moreover, Decision Transformer's emphasis on trajectory modeling aligns well with the project's objective of simulating numerous game trajectories and analyzing action distributions. By incorporating Decision Transformer, the AI agent can better understand and adapt to the multifaceted strategies employed by human opponents, thereby improving its overall performance and competitiveness in the game.

## 3. Objectives

- Develop an AI agent capable of playing Big 2 using deep reinforcement learning techniques.
- Implement the Proximal Policy Optimization (PPO) algorithm to train the agent through self-play.
- Integrate Decision Transformer to enhance the agent's sequence modeling and strategic planning capabilities.
- Simulate extensive game trajectories to facilitate comprehensive training and evaluation of the AI.
- Analyze action distributions and refine decision-making processes to improve the agent’s competitiveness.
- Address the challenges of imperfect information and a vast action space inherent in Big 2.
- Contribute to the field of game AI by demonstrating the application of advanced RL methodologies to a complex strategic environment.

## 4. Methodology
### 4.1 Game Environment

The game environment for Big 2 was meticulously designed to replicate the intricacies and rules of the actual card game. Implemented within the `Big2Game` class in `game/big2Game.py`, the environment manages all aspects of gameplay, including card distribution, action validation, state progression, and reward allocation.

Key components of the game environment include:

- **State Representation**: Each game state is encoded using a combination of one-hot encoding for player hands, last played actions, and additional game-specific features. This comprehensive state representation enables the reinforcement learning agent to effectively interpret and respond to the current game scenario.

   - **Shape and Dimension**: The state vector is a fixed-length tensor with a dimension of **412**. This includes:
     - **Player Hands**: Each player's hand is represented using a one-hot encoded vector of length **52**, corresponding to the 52 possible cards in a standard deck. For four players, this accounts for **208** dimensions.
     - **Last Played Actions**: The last actions taken by each player are encoded into **4** dimensions, capturing the most recent moves.
     - **Game-Specific Features**: Additional features such as the current player's turn, remaining cards, and other relevant game states contribute to the remaining **200** dimensions.
     
     This structured representation ensures that the agent has a holistic view of the game state, facilitating informed decision-making.

- **Action Space**: The action space encompasses all possible valid moves a player can make, such as playing singles, pairs, straights, flushes, and full houses, as well as the option to pass. The `enumerateOptions.py` module precomputes action indices to facilitate efficient action selection and validation during gameplay.

   - **Shape and Dimension**: The action space is represented as a one-hot encoded vector with a dimension of **1695**. Each unique action corresponds to a specific index within this vector, allowing the agent to select actions by assigning probabilities across the entire action space. The high dimensionality accounts for the vast number of possible card combinations and strategic moves available in Big 2, ensuring that the agent can explore and learn from a comprehensive set of actions.

- **Game Mechanics**: The environment handles the core game mechanics, including turn management, action processing, hand updates, and game termination conditions. The `step` function orchestrates the progression of the game by applying player actions, updating hands, and determining game outcomes.

- **Parallel Simulation**: To support scalable training, the environment is vectorized using the `vectorizedBig2Games` class. This allows for concurrent simulation of multiple game instances, significantly speeding up the data collection process required for training the reinforcement learning models.

- **Reward Structure**: Rewards are strategically assigned based on game outcomes to encourage the AI agent to adopt winning strategies. The reward mechanism penalizes players for the number of cards remaining in their hands and rewards them for successfully discarding all cards, thereby aligning the agent's objectives with the game's win conditions.

This robust game environment serves as the foundation for training the AI agent using the Proximal Policy Optimization (PPO) algorithm. By accurately modeling the game's dynamics and providing a rich state and action space, the environment enables the development of sophisticated strategies and enhances the agent's ability to compete effectively in Big 2.

### 4.2 Proximal Policy Optimization (PPO)

- **Algorithm Overview**: Proximal Policy Optimization (PPO) is a policy gradient method that strikes a balance between exploration and exploitation by optimizing a surrogate objective function. PPO ensures that the policy updates are not too drastic, maintaining stability during training.

- **Implementation Details**:
    - **Policy Network Architecture**: The PPO agent utilizes a neural network comprising multiple fully connected layers with ReLU activation functions. The input layer receives the state representation (dimension 412), followed by hidden layers that process the information before outputting a probability distribution over the action space (dimension 1695).
    - **Optimization Parameters**: Key hyperparameters include:
        - **Learning Rate**: Set to 2.5e-4 to ensure gradual policy updates.
        - **Clip Range**: 0.2, which restricts the policy update to prevent large deviations.
        - **Entropy Coefficient**: 0.01, encouraging exploration by penalizing certainty.
        - **Value Function Coefficient**: 0.5, balancing the importance of the value loss.
        - **Maximum Gradient Norm**: 0.5, for gradient clipping to prevent exploding gradients.
    - **Training Process**: The PPO training loop involves collecting trajectories through self-play, computing advantages using Generalized Advantage Estimation (GAE), and performing multiple optimization epochs using minibatches sampled from the collected data.

### 4.3 Data Collection and Trajectory Management

- **Trajectory Collection**: The agent interacts with the simulated game environment by selecting actions based on its current policy. Each action leads to a state transition and receives corresponding rewards. These interactions are recorded as trajectories, capturing the sequence of states, actions, rewards, and timesteps for each player.

- **Storage Format**: Trajectories are stored in HDF5 files using the `ppo_gameplay_collect.py` script. Each player's trajectory is saved as a separate sequence within the HDF5 file, organized for efficient retrieval and preprocessing.

- **Data Preprocessing**: Before training, the collected trajectories undergo preprocessing steps:
    - **Padding**: Sequences are padded to a fixed length (e.g., 30 timesteps) to ensure uniformity across batches.
    - **Normalization**: Rewards are normalized to stabilize training.
    - **Attention Masking**: Attention masks are created to differentiate between valid tokens and padding, facilitating effective handling of variable-length sequences during model training.

### 4.4 Integration of Decision Transformer

- **Rationale**: To enhance strategic planning and decision-making, Decision Transformer is integrated alongside PPO. This approach leverages sequence modeling to provide the agent with a holistic understanding of past actions and states, improving its ability to anticipate and plan future moves.

- **Implementation Approach**:
    - **Architectural Integration**: The Decision Transformer module is incorporated into the PPO framework, allowing the agent to process entire trajectories through transformer layers. This integration enables the model to capture long-range dependencies and contextual information from past interactions.
    - **Training Strategy**: Both PPO and Decision Transformer components are trained concurrently, with the transformer enhancing the policy network's ability to make informed decisions based on historical data.

### 4.5 Training Process

- **Parallel Simulation**: Utilizing the `vectorizedBig2Games` class, multiple game instances (e.g., 64 parallel games) are simulated simultaneously. This parallelization accelerates data collection, providing diverse and extensive trajectories for training.

- **Minibatch Sampling**: During each training update, a portion of the collected trajectories is randomly sampled and divided into minibatches. This stochastic sampling promotes robust learning by exposing the agent to varied gameplay scenarios within each training epoch.

- **Optimization Loop**:
    1. **Data Collection**: Gather trajectories from parallel game simulations.
    2. **Preprocessing**: Pad, normalize, and mask the trajectories.
    3. **Advantage Calculation**: Compute advantages using GAE to estimate the benefit of actions taken.
    4. **Policy Update**: Perform multiple optimization epochs with shuffled minibatches, updating the policy network while ensuring updates remain within the clipping range.
    5. **Model Saving**: Periodically save the model parameters and training losses to facilitate checkpointing and analysis.

- **Hyperparameter Tuning**: Hyperparameters such as learning rate, batch size, number of optimization epochs, and clipping range are systematically tuned based on empirical results to achieve optimal performance and training stability.

### 4.6 Evaluation Metrics

- **Win Rate**: The primary metric for evaluating the agent's performance is its win rate against baseline strategies and human players. This provides a direct measure of the agent's competitiveness and effectiveness in gameplay.

- **Reward Analysis**: Analyzing the distribution of rewards over training episodes helps assess the agent's learning progress and the effectiveness of the reward structure in guiding optimal behavior.

- **Action Distribution**: Examining the frequency and variety of actions taken by the agent offers insights into its decision-making patterns and strategic preferences.

- **Training Stability**: Monitoring metrics such as policy loss, value loss, and entropy over training iterations ensures that the PPO algorithm maintains stable and convergent training dynamics.

- **Generalization**: Testing the agent in varied gameplay scenarios, including against different types of opponents, evaluates its ability to generalize learned strategies beyond the training environment.

### 4.7 Tools and Technologies

- **Programming Language**: Python was chosen for its rich ecosystem of libraries and frameworks suitable for machine learning and game development.

- **Libraries and Frameworks**:
    - **TensorFlow and PyTorch**: Utilized for implementing and training neural network models.
    - **Tkinter**: Employed for developing the graphical user interface (GUI) for interacting with the AI agent.
    - **H5py**: Used for handling HDF5 file operations related to trajectory storage and retrieval.
    - **NumPy and Matplotlib**: Leveraged for numerical computations and data visualization.
    - **Joblib**: Employed for saving and loading model parameters and training statistics.
  
- **Hardware**: Training was conducted on systems equipped with GPUs to accelerate neural network computations, enabling efficient handling of large-scale data and complex model architectures.

- **Version Control**: Git was used for version control, managing codebase changes, and facilitating collaboration through GitHub repositories.

## 5. Implementation

The implementation of the AI agent for Big 2 encompasses several key components, including the game environment setup, the PPO algorithm, integration of the Decision Transformer, data collection and preprocessing, and the training pipeline. Below is a comprehensive overview of each component:

### 5.1 System Architecture

The system is structured into distinct modules to ensure modularity and scalability:

- **Game Environment (`game/big2Game.py`)**: Simulates the Big 2 game mechanics, managing card distribution, action validation, state updates, and reward allocation.
- **PPO Simulation (`mainBig2PPOSimulation.py`)**: Handles the training loop, including data collection through self-play, advantage estimation, and policy updates.
- **Trainer (`trainer.py`)**: Implements the Decision Transformer and manages its training alongside PPO.
- **Data Collection (`ppo_gameplay_collect.py`)**: Facilitates the collection and storage of game trajectories in HDF5 format for efficient access and preprocessing.

### 5.2 Proximal Policy Optimization (PPO)

Proximal Policy Optimization is central to training the AI agent. The implementation details are as follows:

- **Algorithm Implementation**: The PPO algorithm is implemented within the `big2PPOSimulation` class in `mainBig2PPOSimulation.py`. It leverages the `training_model` for policy and value network updates.
  
- **Hyperparameters**:
    - **Input Dimension**: 412
    - **Number of Games**: 8 parallel games
    - **Steps per Game**: 20
    - **Minibatches**: 4
    - **Optimization Epochs**: 5
    - **GAE Lambda**: 0.95
    - **Discount Factor (Gamma)**: 0.995
    - **Entropy Coefficient**: 0.01
    - **Value Function Coefficient**: 0.5
    - **Maximum Gradient Norm**: 0.5
    - **Learning Rate**: 2.5e-4 (with a minimum of 1e-6)
    - **Clip Range**: 0.2
    - **Save Interval**: Every 500 updates

- **Training Loop**: The training loop involves the following steps:
    1. **Data Collection**: Simulate multiple games in parallel using the `vectorizedBig2Games` class.
    2. **Trajectory Storage**: Collect observations, actions, rewards, and other relevant data, storing them in buffers.
    3. **Advantage Estimation**: Compute advantages using Generalized Advantage Estimation (GAE) to evaluate the benefit of actions.
    4. **Policy and Value Updates**: Perform multiple optimization epochs on minibatches sampled from the collected data.
    5. **Model Saving**: Periodically save the model parameters and training statistics for checkpointing and analysis.

### 5.3 Decision Transformer Integration

To enhance the agent's strategic planning capabilities, Decision Transformer is integrated alongside PPO:

- **Implementation**: The Decision Transformer is implemented in `trainer.py` within the `train_decision_transformer` function. It processes entire trajectories to capture long-range dependencies using transformer layers.
  
- **Hyperparameters**:
    - **Number of Blocks**: 6
    - **Hidden Dimension**: 1024
    - **Number of Heads**: 8
    - **Dropout Probability**: 0.1
    - **Context Length**: 30
    - **Max Timestep**: 1000
    - **Batch Size**: 16
    - **Learning Rate**: 1e-4
    - **Maximum Epochs**: 10
    - **Gradient Clipping**: 1.0

- **Training Strategy**: Both PPO and Decision Transformer models are trained concurrently. The transformer enhances the policy network by providing a holistic understanding of past actions and states, allowing the agent to anticipate and plan future moves more effectively.

### 5.4 Data Collection and Preprocessing

Efficient data handling is crucial for training robust models:

- **Trajectory Collection**: Implemented in `ppo_gameplay_collect.py`, trajectories are collected by simulating games where the agent interacts with the environment, making decisions based on its current policy.
  
- **Storage**: Trajectories are stored in HDF5 files using the `H5py` library, facilitating fast read and write operations.

- **Preprocessing Steps**:
    - **Padding**: Ensures that all sequences have a uniform length by padding shorter sequences.
    - **Normalization**: Rewards are normalized to maintain stability during training.
    - **Attention Masking**: Differentiates between valid data points and padding, enabling the transformer to focus on relevant information.

### 5.5 Training Pipeline

The training pipeline orchestrates the entire process from data collection to model optimization:

1. **Initialization**: Set up the game environment, PPO simulation parameters, and initialize models.
2. **Parallel Game Simulation**: Utilize the `vectorizedBig2Games` class to run multiple game instances concurrently, accelerating data collection.
3. **Data Aggregation**: Collect and aggregate data from all parallel games, storing observations, actions, rewards, and other pertinent information.
4. **Advantage Calculation**: Compute advantages using GAE to assess the quality of actions taken.
5. **Minibatch Training**: Shuffle and divide the aggregated data into minibatches, performing multiple optimization epochs to update the policy and value networks.
6. **Decision Transformer Training**: Concurrently train the Decision Transformer using the collected trajectories to enhance strategic planning.
7. **Model Evaluation and Saving**: Periodically evaluate the model's performance, saving the best-performing models based on loss metrics.

### 5.6 Tools and Technologies

The implementation leverages a range of tools and technologies to facilitate development and training:

- **Programming Language**: Python, chosen for its extensive machine learning and game development libraries.
  
- **Libraries and Frameworks**:
    - **TensorFlow and PyTorch**: Utilized for building and training neural network models.
    - **Tkinter**: Used for developing a graphical user interface (GUI) for interacting with the AI agent.
    - **H5py**: Handles HDF5 file operations for trajectory storage and retrieval.
    - **NumPy and Matplotlib**: Employed for numerical computations and data visualization.
    - **Joblib**: Facilitates saving and loading model parameters and training statistics.
  
- **Hardware**: Training is conducted on GPU-equipped systems to accelerate neural network computations, enabling efficient handling of large datasets and complex model architectures.
  
- **Version Control**: Git is used for version control, managing codebase changes, and facilitating collaboration through GitHub repositories.

### 5.7 Code Structure

The project is organized into several key files and directories:

- **`mainBig2PPOSimulation.py`**: Contains the `big2PPOSimulation` class, managing the PPO training loop, data collection, and model updates.
  
- **`trainer.py`**: Implements the Decision Transformer and handles its training process.
  
- **`ppo_gameplay_collect.py`**: Facilitates the collection and storage of gameplay trajectories.
  
- **`game/big2Game.py`**: Defines the game environment, encapsulating all game rules and mechanics.
  
- **`enumerateOptions.py`**: Precomputes and manages the action indices for efficient action selection and validation.

This modular structure ensures clarity, maintainability, and scalability, allowing for seamless integration of additional features and algorithms in the future.

## 6. Results and Analysis
// ...existing code...

## 7. Discussion
// ...existing code...

## 8. Conclusion
// ...existing code...

## 9. Future Work
// ...existing code...
---
