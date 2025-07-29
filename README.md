# Deep Reinforcement Learning Applications in Probability-Driven Scenarios
## Big Two Card Game AI Agent Comparison


A comprehensive research project implementing and evaluating four distinct AI methodologies for playing Big Two (鋤大弟), focusing on **loss minimization in highly randomized scenarios** with incomplete information. This work demonstrates the evolution from traditional machine learning to cutting-edge transformer architectures in strategic decision-making environments.

## 🎯 Research Objectives

This project addresses the fundamental challenge of **developing AI agents that can minimize losses in probability-driven scenarios** where decisions must be made with limited information and uncertain outcomes. Specifically, we aim to:

1. **Compare Algorithm Performance**: Systematic evaluation of traditional ML vs. deep RL approaches
2. **Loss Minimization Focus**: Optimize strategies specifically for risk mitigation rather than pure win maximization  
3. **Imperfect Information Handling**: Develop robust decision-making in environments with hidden information
4. **Generalization Potential**: Create approaches applicable to real-world scenarios like financial markets and risk management

### Implemented Algorithms
1. **Decision Tree** - Traditional rule-based approach using sklearn
2. **Deep Q-Network (DQN)** - Value-based deep reinforcement learning
3. **Proximal Policy Optimization (PPO)** - State-of-the-art policy gradient method
4. **Decision Transformer** - Transformer architecture treating RL as sequence modeling

## 🃏 Big Two Game Environment

**Big Two** serves as an ideal testbed for probability-driven decision-making research due to its unique characteristics:

### Game Complexity
- **State Space**: 412-dimensional representation capturing hand compositions, game history, and player positions
- **Action Space**: 1695 possible actions including all valid card combinations
- **Player Configuration**: 4-player competitive environment with rotating turn order
- **Game Duration**: Variable length games with average 60-80 turns per game

### Research Challenges
- **Imperfect Information**: Players must make decisions without knowledge of opponents' cards
- **Sparse Reward Structure**: Feedback only available at game completion, requiring effective credit assignment
- **Strategic Depth**: Balance between aggressive plays and conservative risk management
- **Probability Assessment**: Estimating opponent strategies and remaining card distributions

## 🏗️ Architecture

### Game Engine (`game/`)
- `big2Game.py` - Core game mechanics and vectorized environment
- `gameLogic.py` - Card game rules and validation
- `enumerateOptions.py` - Action space enumeration
- `generateGUI.py` - Graphical user interface

### AI Models (`models/`)

#### 1. Decision Tree (`decision_tree.py`)
```python
class Big2DecisionTree:
    def __init__(self, max_depth=10):
        self.model = DecisionTreeClassifier(max_depth=max_depth)
```
- **Approach**: Rule-based decision making using sklearn
- **Strengths**: Interpretable, fast inference
- **Limitations**: Struggles with large state spaces and complex patterns

#### 2. Deep Q-Network (`ddqn.py`)
```python
class QNetwork(nn.Module):
    def __init__(self, action_dim, state_dim, hidden_dim):
        self.fc_1 = nn.Linear(state_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_3 = nn.Linear(hidden_dim, action_dim)
```
- **Approach**: Deep Q-learning with experience replay and target networks
- **Features**: Double DQN implementation, epsilon-greedy exploration
- **Strengths**: Handles large state spaces, learns from experience

#### 3. Proximal Policy Optimization (`PPONetwork_pytorch.py`)
```python
class PPONetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        self.fc1 = nn.Linear(obs_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.pi = nn.Linear(256, act_dim)  # Policy head
        self.vf = nn.Linear(256, 1)       # Value head
```
- **Approach**: Policy gradient method with clipped objective
- **Features**: Actor-critic architecture, self-play training
- **Strengths**: Stable training, effective for complex environments

#### 4. Decision Transformer (`decision_transformer.py`)
```python
class DecisionTransformer(TrajectoryModel):
    def __init__(self, state_dim=412, act_dim=1695, hidden_size=512):
        config = transformers.GPT2Config(
            vocab_size=1, n_embd=hidden_size, n_layer=6, n_head=8
        )
```
- **Approach**: Transformer architecture treating RL as sequence modeling
- **Features**: GPT-2 based, return-to-go conditioning, causal attention
- **Strengths**: Handles sequential dependencies, effective credit assignment

## 🚀 Training Methodology

### Self-Play Training
- **PPO & DQN**: Agents learn through self-play against copies of themselves
- **Decision Transformer**: Trained on trajectories collected from PPO gameplay
- **Validation**: 100 games against random agents every checkpoint

### Data Collection Pipeline
```python
# PPO trajectory collection for Decision Transformer
python ppo_gameplay_collect.py  # Collects gameplay data
python load_trajectories.py     # Processes trajectory data
python odt_trainer.py          # Trains Decision Transformer
```

### Training Scripts
- `mainBig2PPOSimulation.py` - PPO self-play training
- `DDQN_trainer.py` - Deep Q-Network training
- `decision_tree_trainer.py` - Decision tree training on collected data
- `odt_trainer.py` - Decision transformer training

## 📊 Evaluation Framework

### Performance Metrics
- **Win Rate**: Percentage of games won against other agents
- **Average Reward**: Normalized rewards (-1 to 1 range)
- **Game Completion Time**: Efficiency metric
- **Strategic Analysis**: Action distribution and decision patterns

### Evaluation Pipeline
```python
python evaluate_model.py  # Comprehensive agent vs agent evaluation
```

### Tournament Structure
- **Round-robin format**: Each agent plays against every other agent
- **Statistical significance**: 1000 games per matchup
- **Human evaluation**: 20 games per agent against human players

## 🔧 Installation & Usage

### Requirements
```bash
pip install torch numpy sklearn matplotlib pandas h5py
pip install stable-baselines3 transformers
pip install gym joblib tensorboard
```

### Quick Start
```bash
# Clone repository
git clone [repository-url]
cd big_two_rl

# Train PPO agent
python mainBig2PPOSimulation.py

# Train DQN agent  
python DDQN_trainer.py

# Train Decision Tree
python decision_tree_trainer.py

# Train Decision Transformer
python odt_trainer.py

# Evaluate all models
python evaluate_model.py
```

### Playing Against Trained Agents
```bash
# Launch GUI for human vs AI gameplay
python game/generateGUI.py
```

## 📈 Experimental Results & Key Findings

### Performance Evaluation Metrics
- **Win Rate**: Percentage of games won against other agents and human players
- **Average Reward**: Normalized rewards reflecting strategic effectiveness
- **Training Performance**: Learning curves and convergence analysis over 140,000 episodes
- **Action Distribution**: Strategic pattern analysis from 1000+ game samples

### Algorithm Performance Rankings

#### 1. **Proximal Policy Optimization (PPO)** - *Champion Performance*
- **Agent vs Agent**: 58-88% win rate across all matchups
- **Human vs Agent**: **72% win rate** - highest human-competitive performance
- **Training**: Final reward +10.1 (from initial -2.1), convergence at ~100K episodes
- **Strategy**: Conservative gameplay with 49.83% pass actions, balanced risk management
- **Technical**: Actor-critic architecture with robust value function estimation

#### 2. **Deep Q-Network (DQN)** - *Strong Second Place*
- **Agent vs Agent**: 42-81% win rate range
- **Human vs Agent**: **60% win rate** - solid human-competitive performance  
- **Training**: Final reward +5.8 (from initial -3.4), convergence at ~80K episodes
- **Limitations**: Value overestimation in large action spaces, ε-greedy exploration constraints
- **Technical**: Double DQN with experience replay and target networks

#### 3. **Decision Transformer** - *Mixed Performance*
- **Agent vs Agent**: 25-77% win rate (highly variable)
- **Human vs Agent**: **51% win rate** - moderate human-competitive performance
- **Training Challenge**: Severe action distribution imbalance (49.83% pass actions)
- **Reward Sensitivity**: Performance varied dramatically with reward normalization
- **Innovation**: First transformer application to Big Two, successful with proper reward scaling

#### 4. **Decision Tree** - *Baseline Reference*
- **Agent vs Agent**: 12-23% win rate - consistently lowest
- **Human vs Agent**: **15% win rate** - limited strategic capability
- **Limitations**: Rigid rule-based approach, unable to adapt to opponent strategies
- **Advantage**: Fully interpretable decision logic, fast inference (<1ms per action)

### Critical Research Insights

#### **Sparse Reward Challenge Successfully Addressed**
- **PPO Dominance**: Achieved 72% win rate against humans through superior credit assignment
- **Decision Transformer Breakthrough**: Reward normalization (-1 to 1) increased performance from 2% to 87% win rate
- **Action Distribution Bias**: 49.83% pass actions created severe training imbalance, limiting transformer effectiveness
- **Strategic Implications**: Conservative strategies emerged as most effective in imperfect information environments

#### **Technical Breakthroughs**
- **Action Masking Integration**: Essential for handling 1695 possible actions in Big Two's complex action space
- **Self-Play Effectiveness**: 140,000 episode training demonstrated clear learning progression in both PPO and DQN
- **Reward Engineering**: Proper reward normalization proved critical for transformer-based approaches
- **Architecture Comparison**: Policy-based methods (PPO) outperformed value-based methods (DQN) in strategic depth

#### **Human-Competitive AI Achievement**
- **PPO**: 72% win rate against human players - demonstrating superhuman strategic capability
- **DQN**: 60% win rate - solid human-level performance with room for improvement
- **Decision Transformer**: 51% win rate - near human-level with significant potential
- **Decision Tree**: 15% win rate - baseline traditional approach limitations confirmed

#### **Strategic Pattern Discovery**
- **Conservative Dominance**: Top-performing agents adopted risk-averse strategies with frequent passing
- **Single Card Preference**: 9.92% of actions were single-card plays, indicating strategic simplicity
- **Complex Combination Avoidance**: Limited use of advanced card combinations suggests optimization challenges
- **Adaptation Capability**: Deep RL methods showed superior adaptation to opponent strategies vs. rule-based approaches

## 📁 Project Structure

```
big_two_rl/
├── game/                    # Game environment
│   ├── big2Game.py         # Core game mechanics
│   ├── gameLogic.py        # Game rules
│   └── generateGUI.py      # User interface
├── models/                  # AI implementations
│   ├── decision_tree.py    # Decision tree agent
│   ├── ddqn.py            # Deep Q-Network
│   ├── PPONetwork_pytorch.py # PPO implementation
│   └── decision_transformer.py # Transformer agent
├── training scripts/        # Training pipelines
│   ├── mainBig2PPOSimulation.py
│   ├── DDQN_trainer.py
│   └── odt_trainer.py
├── evaluation/             # Model evaluation
│   └── evaluate_model.py
```

## 🎓 Research Contributions & Impact

### Academic Significance
This work contributes to several key areas in AI and machine learning research:

#### **Imperfect Information Games**
- Novel application of decision transformers to card game domains
- Comparative analysis of RL algorithms in sparse reward environments
- Innovative reward shaping techniques for loss minimization

#### **Transfer Learning Potential**
- **Financial Applications**: Portfolio optimization under market uncertainty
- **Risk Management**: Strategic decision-making with incomplete information  
- **Business Intelligence**: Competitive strategy development in dynamic environments
- **Autonomous Systems**: Decision-making under uncertainty and partial observability

### Technical Innovations
- **Hybrid Architecture**: Combining self-play training with transformer sequence modeling
- **Action Masking Integration**: Efficient handling of large, constrained action spaces
- **Reward Normalization**: Novel approach to sparse reward problems in card games
- **Vectorized Environment**: Parallel game execution for efficient training

### Future Research Directions
- **Multi-Agent Extensions**: Opponent modeling and meta-learning approaches
- **Transfer Learning**: Applying learned strategies to other imperfect information games
- **Real-World Deployment**: Integration with financial decision support systems
- **Interpretability**: Developing explainable AI for strategic decision-making

## 📚 References & Related Work

### Core Algorithms
- Chen, L., et al. (2021). *Decision Transformer: Reinforcement Learning via Sequence Modeling*
- Schulman, J., et al. (2017). *Proximal Policy Optimization Algorithms*
- Mnih, V., et al. (2015). *Human-level control through deep reinforcement learning*
- Vaswani, A., et al. (2017). *Attention is All You Need*

### Imperfect Information Games
- Heinrich, J., & Silver, D. (2016). *Deep reinforcement learning from self-play in imperfect-information games*
- Brown, N., et al. (2020). *Combining deep reinforcement learning and search for imperfect-information games*
- Zinkevich, M., et al. (2007). *Regret minimization in games with incomplete information*
- Silver, D., et al. (2017). *Mastering the game of Go without human knowledge*


## 🏆 Project Impact

### **Academic Recognition**
- Comprehensive comparison of traditional ML vs. modern deep RL approaches
- Novel application of transformers to loss minimization in card games
- Reproducible research with complete codebase and documentation

### **Industry Relevance**
- **FinTech Applications**: Risk assessment and portfolio optimization techniques
- **Gaming Industry**: AI agent development for competitive gaming environments  
- **Decision Support Systems**: Uncertainty handling in business applications
- **Research Community**: Open-source implementation advancing imperfect information game research

### **Technical Skills Demonstrated**
- **Deep Learning Frameworks**: PyTorch, Transformers, Stable-Baselines3
- **Advanced RL Techniques**: Self-play training, experience replay, action masking, reward normalization
- **Research Methodology**: Statistical significance testing (1000+ games per matchup), ANOVA analysis
- **Software Engineering**: Modular architecture, parallel environments (8-32 games), comprehensive evaluation framework
- **Data Analysis**: Action distribution analysis, strategic pattern mining, performance visualization

