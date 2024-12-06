<!-- Final Report Template – Guide Only -->
<!-- Explanatory Notes -->

<!-- 
    1. The structure outlined in this template serves as a flexible guide rather than a rigid blueprint for the final report. While the chapters and sections presented here are commonly found in research theses or technical reports, the specific nature of the undertaken research may necessitate variations in structure. Additionally, the order of items within chapters can be adjusted accordingly. The template reflects the traditional technical report structure, which aims to demonstrate a coherent line of argument across six chapters: introduction, literature review, research design, results, discussion, and conclusions. Furthermore, logical relationships exist between pairs of chapters, such as the connections between the introduction and conclusions, literature review and discussion, and research design and results.
    2. This template has been formatted for double-sided printing, a requirement for the final report, while providing the option for single-sided printing in earlier stages. The document features mirror margins with odd and even pages. After the title page, at the end of preliminary pages, and at the conclusion of each chapter, an odd page section break has been included. This ensures that the subsequent page begins on the right-hand side, as an odd page, eliminating the need for the insertion of blank pages.
    3. Automatic numbering of headings has been implemented in this template, enabling the use of cross-referencing and automatic numbering for figure and table captions, which include the corresponding chapter number. While the headings and styles utilized in this template can be modified to suit individual preferences, the current formatting facilitates consistent numbering and referencing throughout the document.
-->

# Full Title of Your Project. The Style is Called “Title, Project Title”
[Put your group member full names with student number here]

[For your Confirmation document, list your Supervisors here]

Submitted in [partial] fulfilment of the requirements for the degree of
[e.g. Doctor of Philosophy or Master of Education (Research)]

Centre for Learning Innovation  
Faculty of Education  
Queensland University of Technology  
[year in which the thesis is submitted]

---

## Abstract
Put your abstract here (do this last).

## Table of Contents
- Keywords i
- Abstract ii
- Table of Contents iii
- List of Figures iv
- List of Tables v
- List of Abbreviations vi
- Statement of Original Authorship vii
- Acknowledgements viii
- Chapter 1: Introduction 1
  - 1.1 Background 1
  - 1.2 Context 1
  - 1.3 Purposes 1
  - 1.4 Significance, Scope and Definitions 1
  - 1.5 Thesis Outline 2
- Chapter 2: Literature Review 3
  - 2.1 Historical Background [optional] 4
  - 2.2 Topic 1 4
  - 2.3 Topic 2 4
  - 2.4 Topic 3 4
  - 2.5 Summary and Implications 4
- Chapter 3: Research Design 5
  - 3.1 Methodology and Research Design 5
  - 3.2 Participants 6
  - 3.3 Instruments 6
  - 3.4 Procedure and Timeline 6
  - 3.5 Analysis 6
  - 3.6 Ethics and Limitations 7
- Chapter 4: Results 9
- Chapter 5: Analysis 11
- Chapter 6: Conclusions 13
- Bibliography 15
- Appendices 17
  - Appendix A Title 17

---

## Chapter 1: Introduction
<!-- The introduction chapter needs to state the objectives of the program of research, include definitions of the key concepts and variables and give a brief outline of the background and research approach. The aim of the introduction is to contextualise the proposed research.

In the opening paragraph, give an overall view of what is included in the chapter. For example:
This chapter outlines the background (section 1.1) and context (section 1.2) of the research, and its purposes (section 1.3). Section 1.4 describes the significance and scope of this research and provides definitions of terms used. Finally, section 1.5 includes an outline of the remaining chapters of the thesis. -->
This chapter outlines the background (section 1.1) and context (section 1.2) of the research on "Deep Reinforcement Learning Applications in Probability-Driven Scenarios," specifically applied to the Big Two card game. Section 1.3 defines the objectives of the study, including key concepts and variables. Section 1.4 discusses the significance and scope of the research, while providing definitions of critical terms used throughout the thesis. Finally, section 1.5 presents an outline of the remaining chapters, establishing a clear framework for the research.


### 1.1 Background
<!-- Give the background of the problem to be explored in your study and what led you to doing the thesis. For example, you might discuss educational trends related to the problem, unresolved issues, social concerns. You might also include some personal background. -->
In today's complex decision-making environments, randomness and uncertainty play a significant role in human choices. Probability-driven scenarios reflect this reality, requiring decision-makers to assess risks and rewards effectively. This research explores how artificial intelligence (AI), particularly through deep reinforcement learning (DRL), can model and navigate these uncertainties, mirroring human decision-making processes. The Big Two card game serves as an ideal context for this exploration due to its strategic depth and inherent randomness, making it a suitable platform for training AI algorithms

### 1.2 Context
<!-- Outline the context of the study (i.e., the major foci of your study) and give a statement of the problem situation (basic difficulty – area of concern, felt need). -->
The study focuses on applying deep reinforcement learning to improve decision-making in probability-driven scenarios and also compares the result to find whether deep reinforcement learning is the best way to solve the problem under the scenarios or not. The primary challenge addressed is how AI can learn to make optimal decisions in environments characterized by imperfect information. By evaluating the performance of three methodologies—traditional algorithms (decision trees), machine learning approaches (reinforcement learning), and deep reinforcement learning techniques (PPO with Transformer architectures)—this research aims to uncover insights into the effectiveness of these approaches in the context of the Big Two game.

### 1.3 Purposes
<!-- Define the purpose and specific aims and objectives of the study. Emphasise the practical outcomes or products of the study. Delineate the research problem and outline the questions to be answered or the overarching objectives to be achieved. -->
The primary objectives of this study are:
To compare the effectiveness of three methods: traditional algorithms, machine learning approaches, and deep reinforcement learning.
To evaluate the performance of deep reinforcement learning against a baseline, specifically the PPO agent as discussed in the paper "Application of Self-Play Reinforcement Learning to a Four-Player Game of Imperfect Information" (2018).
To discuss the limitations of each method and explore their applicability to real-life decision-making scenarios.
This research seeks to answer key questions, including: How do the different methods compare in terms of decision-making performance? How does deep reinforcement learning measure up against the established PPO baseline? What are the implications of these findings for real-world applications?


### 1.4 Significance, Scope and Definitions
<!-- Discuss the importance of your research in terms of the topic (problem situation), the methodology, and the gap in the literature. Outline the scope and delimitations of the study (narrowing of focus). Define and discuss terms to be used (largely conceptual here; operational definitions may follow in Research Design chapter). -->
This research is significant as it addresses a critical gap in the literature regarding the application of deep reinforcement learning in probability-driven scenarios. By focusing on a card game, we aim to contribute to the understanding of how AI can learn and adapt in uncertain environments. The scope of the study is confined to the Big Two card game, allowing for an in-depth examination of different methodologies.
Key terms and concepts relevant to this research include:
Deep Reinforcement Learning (DRL): A machine learning approach that combines reinforcement learning with deep neural networks to enable agents to learn optimal behaviors from their environments.
Probability-Driven Scenarios: Contexts where decisions are influenced by uncertainty and randomness, necessitating probabilistic reasoning for effective outcomes.
Minimum Cost: An optimization goal focused on minimizing total costs associated with decision-making processes.
Linear Regression: A statistical method used to model relationships between variables and predict outcomes based on input data.

### 1.5 Report Outline
<!-- Outline the chapters for the remainder of your thesis. -->

The subsequent chapters of this thesis will include:
Chapter 2: Literature Review – A comprehensive overview of existing research on deep reinforcement learning, probability-driven scenarios, and decision-making in games.
Chapter 3: Methodology – A detailed description of the research design, including data collection methods and analytical techniques used.
Chapter 4: Results – Presentation and analysis of the findings from the conducted experiments.
Chapter 5: Discussion – Interpretation of results in the context of the existing literature and practical implications.
Chapter 6: Conclusion – A summary of findings, contributions to the field, and suggestions for future research.



## Chapter 2: Literature Review
This chapter reviews the existing literature relevant to artificial intelligence approaches in card games, particularly focusing on three main methodologies: traditional decision trees (Section 2.2), reinforcement learning approaches (Section 2.3), and transformer-based methods (Section 2.4). The review examines how these approaches have evolved and their applications in game-playing scenarios, particularly those involving imperfect information like card games.

### 2.1 Historical Background
The challenge of developing AI systems for card games has been a significant research area since the early days of artificial intelligence. Unlike perfect information games like chess, card games present unique challenges due to their probabilistic nature and hidden information. Early approaches relied heavily on rule-based systems and decision trees, which evolved into more sophisticated machine learning techniques, and eventually to modern deep learning approaches.

### 2.2 Decision Trees in Game Playing
Decision trees have been a fundamental approach in game-playing AI systems. They provide a systematic way to evaluate possible moves and their consequences through a tree-like structure of decision points.

#### 2.2.1 Traditional Approaches
Early implementations of decision trees in card games focused on minimax algorithms and their variants. Studies by Smith and Jones (2015) demonstrated that decision trees could effectively handle simple card games with limited branching factors. However, their effectiveness diminishes in complex games due to the exponential growth of possible states.

#### 2.2.2 Limitations and Adaptations
The main limitation of decision trees in card games is the explosion of possible states, known as the "curse of dimensionality" (Brown et al., 2018). Various pruning techniques have been developed to address this, including alpha-beta pruning and monte carlo tree search (MCTS).

### 2.3 Reinforcement Learning Methods
Reinforcement learning has emerged as a powerful approach for handling complex game scenarios, offering advantages over traditional decision trees in dealing with large state spaces and uncertainty.

#### 2.3.1 Deep Q-Network (DQN)
DQN, introduced by Mnih et al. (2015) in their seminal paper with DeepMind, represents a significant advancement over traditional Q-learning by combining Q-learning with deep neural networks. Unlike traditional Q-learning, DQN can handle high-dimensional state spaces effectively through its neural network architecture. The method introduced several key innovations including experience replay and target networks, which greatly improved the stability of learning in complex environments.

Recent applications of DQN to card games have shown promising results. Wang et al. (2021) demonstrated that DQN could learn effective strategies in card games with large state spaces, outperforming traditional Q-learning approaches. The method's ability to automatically learn relevant features from raw input states makes it particularly suitable for complex card games like Big Two, where the state space is too large for traditional tabular methods.

Key advantages of DQN include:
- Ability to handle continuous state spaces
- Improved generalization through neural network function approximation
- More stable learning through experience replay
- Better handling of partial observability in card games

### 2.4 Decision Transformers
The emergence of transformer architectures has led to new approaches in game-playing AI systems, particularly through decision transformers. The evolution of this technology represents a significant shift in how we approach sequential decision-making problems.

#### 2.4.1 Historical Evolution
The development of decision transformers builds upon several key milestones in AI research:

1. Transformer Architecture (Vaswani et al., 2017): Originally designed for natural language processing, the transformer architecture introduced the self-attention mechanism, revolutionizing sequence modeling.

2. GPT Models (2018-2020): The success of large language models demonstrated transformers' capability to learn complex patterns from sequential data.

3. Decision Transformer (Chen et al., 2021): Combined transformer architecture with reinforcement learning concepts, treating the decision-making process as a sequence prediction task rather than traditional value or policy learning.

This progression represents a paradigm shift from traditional reinforcement learning approaches to a more unified sequence modeling framework.

#### 2.4.2 Transformer Architecture in Games
Decision transformers, introduced by Chen et al. (2021), represent a novel approach that treats reinforcement learning as a sequence modeling problem. This architecture has shown promising results in various domains, including game playing scenarios.

#### 2.4.3 Key Technical Innovations
Decision transformers introduced several important technical innovations:

1. Return-to-go Conditioning: Enables explicit control over the desired level of performance
2. Causal Attention Masking: Ensures the model only attends to past information when making decisions
3. State-Action-Return Sequences: Reformulates the reinforcement learning problem as sequence modeling

These innovations have made decision transformers particularly effective in environments with long-term dependencies and complex state spaces, characteristics that are prevalent in card games like Big Two.

### 2.5 Summary and Implications
The literature review reveals several key insights relevant to our research on the Big Two card game. Each approach offers distinct characteristics that address different aspects of our challenge:

1. Decision trees, while providing explainable decisions, face significant limitations in Big Two due to:
   - The game's large state space
   - Imperfect information nature
   - Complex branching factors in possible actions

2. Reinforcement learning methods, particularly DQN and PPO, show promise through:
   - Effective handling of large state spaces
   - Ability to learn from experience in imperfect information scenarios
   - Proven success in similar card game environments
However, these methods may face challenges in Big Two's sparse reward environment, where feedback is only available at the end of the game.

3. Decision transformers offer unique advantages for our context:
   - Demonstrated capability in sparse reward environments through effective credit assignment
   - Ability to learn from sequential decision-making processes
   - Comparable performance to traditional reinforcement learning methods
   - Potential to better handle the delayed reward structure of Big Two

The gap in the literature primarily exists in comparing these approaches' effectiveness specifically in the context of Big Two, particularly considering the game's unique challenges:
- Sparse reward structure (rewards only at game completion)
- Imperfect information environment
- Complex state and action spaces
- Need for long-term strategic planning

This research aims to address this gap by implementing and comparing these three approaches, with particular attention to their performance in handling sparse rewards and imperfect information scenarios characteristic of Big Two.

### 3.1 Methodology

#### 3.1.1 Methodology
This research employs a quantitative experimental methodology to compare the effectiveness of different artificial intelligence approaches in playing Big Two. The study follows a systematic implementation and evaluation process that directly addresses our research questions about the comparative effectiveness of different AI approaches in probability-driven scenarios.

Implementation Process:
1. Development of Big Two Environment
   - Custom implementation of game rules and mechanics
   - State and action space definition
   - Reward system implementation

2. AI Agent Implementation
   - Decision Tree: Traditional rule-based approach
   - DQN: Deep Q-Network implementation
   - PPO: Proximal Policy Optimization implementation
   - Decision Transformer: Sequence modeling approach

3. Training Methodology
   - Self-play training for DQN and PPO agents
   - Collection of PPO gameplay data for Decision Transformer training
   - Continuous validation against random policy agents (100 games per checkpoint)
   - Performance metrics tracking: win rates and cumulative rewards

4. Evaluation Strategy
   a. Agent vs Agent Evaluation
      - Comprehensive tournament (1000 games per matchup)
      - Round-robin format between all agents
      - Metrics: win rates, average rewards, game completion time
   
   b. Human Evaluation
      - 20 games per agent against human players
      - Collection of qualitative feedback
      - Assessment of strategy effectiveness

This methodology directly addresses our research questions through:
1. Quantitative Comparison: The agent vs agent evaluation provides statistical evidence for comparing the effectiveness of different approaches
2. Real-world Applicability: Human evaluation tests the practical effectiveness of each method
3. Learning Efficiency: Training metrics and validation results demonstrate each method's ability to learn and adapt

The methodology is particularly suited to our research questions because:
- The large number of games in agent evaluation (1000 per matchup) ensures statistical significance
- The combination of both automated and human evaluation provides comprehensive performance assessment
- The continuous validation against random agents during training helps track learning progress
- The methodology allows for both quantitative comparison and qualitative analysis of playing strategies

### 3.2 Instruments
Our research utilizes several key instruments:

1. Big Two Game Environment:
   - Custom-built Python implementation
   - State representation system
   - Action space definition
   - Reward mechanism

2. AI Agents:
   - Decision Tree: Rule-based implementation
   - DQN: PyTorch implementation with experience replay
   - PPO: Stable-baselines3 implementation
   - Decision Transformer: Custom implementation based on transformer architecture

3. Data Collection Tools:
   - Game state logger
   - Performance metrics tracker
   - Human feedback forms

### 3.3 Procedure and Timeline
The research procedure is structured to systematically implement and evaluate different AI approaches in the Big Two card game environment, following our established codebase architecture. Each phase has specific data collection and recording procedures to ensure rigorous comparison.

Phase 1: Environment Development (Weeks 1-2)
Implementation Procedure:
- Implementation of core game mechanics in `game/big2Game.py`:
  * Card distribution and validation
  * Action space definition (1695 possible actions)
  * State representation system
  * Reward mechanism with normalization (-1 to 1)
- Development of vectorized environment (`vectorizedBig2Games`) for parallel execution
Data Collection:
- Unit tests for game rule verification
- State space dimensionality validation
- Action space enumeration verification

Phase 2: Agent Implementation (Weeks 3-6)
Implementation Procedure:
- Sequential development of AI approaches in modular format:
  * Week 3: Decision Tree with rule-based logic
  * Week 4: DQN implementation using PyTorch with experience replay
  * Week 5: PPO implementation using stable-baselines3
  * Week 6: Decision Transformer with attention mechanisms
Data Recording:
- Code structure documentation
- Model architecture specifications
- Initial testing results

Phase 3: Training (Weeks 7-10)
Procedure:
- PPO training using `mainBig2PPOSimulation.py`:
  * Self-play training with parallel game instances
  * Checkpoint saving every 500 updates
  * Clip range: 0.2
- Data collection through `ppo_gameplay_collect.py`:
  * Trajectory storage in HDF5 format
  * State-action pairs recording
  * Reward signal tracking
Data Management:
- TensorBoard logging for training metrics
- H5py for trajectory storage
- Joblib for model checkpointing

Phase 4: Evaluation (Weeks 11-14)
- Execute systematic evaluation across all agents
- Conduct human playtesting sessions
- Collect performance metrics and feedback

Phase 5: Analysis (Weeks 15-18)
- Process collected data using NumPy/Pandas
- Generate visualizations with Matplotlib
- Compile comprehensive results

### 3.4 Analysis
Our analysis approach is structured to systematically evaluate the performance and characteristics of different AI methods in the Big Two card game. The analysis methodology is designed to address our research questions through both quantitative and qualitative methods.

1. Training Performance Analysis
   - Data Processing:
     * Normalization of rewards to [-1, 1] range to address sparse reward issues
     * Aggregation of win rates over 1000-game intervals
     * TensorBoard metrics processing using Python's tensorboard.data_compat
   - Statistical Methods:
     * Moving average calculation with window size of 100 games
     * Learning curve analysis using exponential smoothing (α = 0.1)
     * Convergence assessment through variance analysis

2. Comparative Performance Analysis
   - Data Structure:
     * Results matrix: 4×4 for agent matchups (DT, DQN, PPO, Decision Transformer)
     * Per-game metrics: win rate, cards remaining, turn count
   - Statistical Tests:
     * One-way ANOVA for comparing win rates across methods
     * Post-hoc Tukey HSD test for pairwise comparisons
     * Effect size calculation using Cohen's d
     * Confidence intervals (95%) for win rate differences

3. Strategy Analysis
   - Action Distribution Analysis:
     * Calculation of action frequency distributions
     * Chi-square tests for strategy comparison
     * Sequential pattern mining using PrefixSpan algorithm
   - Decision Point Analysis:
     * Critical turn identification through reward attribution
     * Action value distribution analysis using kernel density estimation

4. Implementation Details
   - Tools:
     * Primary analysis: Python's scipy.stats package
     * Visualization: matplotlib and seaborn
     * Data processing: pandas and numpy
   - Code Structure:
     

## Chapter 4: Experimental Results
<!-- Chapter 4 provides a detailed account of the results obtained from the study, presenting them without interpretation, inference, or evaluation, which will be addressed in Chapter 5. The results are presented in a factual and objective manner, closely linked to the study's design. However, in certain investigations of a historical, case-study, or anthropological nature, it may be appropriate to interweave factual and interpretive material instead of presenting them as distinct "findings."

To guide the reader through this chapter, it is recommended to begin with a paragraph outlining the structure of the chapter. The results should be reported in a manner that provides evidence for the research question(s) as outlined in Chapter 1. One approach to organizing the results is to use headings that correspond to each main question of the hypothesis/objectives from Chapter 1 or the theoretical framework from Chapter 2. Alternatively, if applicable, the results can be presented in a sequence that reflects the stages of the study.

By adhering to an objective and factual presentation of the results, this chapter aims to provide a clear and comprehensive account of the findings, setting the stage for their subsequent interpretation and evaluation in Chapter 5. -->
This chapter presents the experimental results obtained from comparing different AI approaches in the Big Two card game environment. Section 4.1 details the training performance metrics for each AI method, including learning curves and convergence analysis. Section 4.2 provides comparative performance data from agent-vs-agent evaluations, supported by statistical analysis. Section 4.3 examines the strategic patterns and decision-making processes employed by each agent. Section 4.4 presents results from human evaluation sessions, including win rates and qualitative feedback. Finally, Section 4.5 summarizes the key findings. All results are presented objectively, with interpretation and discussion reserved for Chapter 5.

### 4.1 Training Performance
This section details the performance metrics collected during the training phase for each AI approach.

#### 4.1.1 DQN Training Results
- Learning curves
- Win rate trends
- Reward normalization impact

#### 4.1.2 PPO Training Results
- Training stability
- Performance metrics
- Comparison with baseline

#### 4.1.3 Decision Transformer Training Results
- Training efficiency
- Sequential decision-making performance
- Reward attribution analysis

### 4.2 Comparative Performance
This section compares the performance of different AI approaches in head-to-head matchups.

#### 4.2.1 Agent vs Agent Evaluation
- Win rates across 1000 games
- Average rewards
- Game completion times
t
#### 4.2.2 Statistical Analysis
- ANOVA results
- Post-hoc comparisons
- Effect size calculations

### 4.3 Strategy Analysis
This section analyzes the strategic patterns and decision-making processes of the AI agents.

#### 4.3.1 Action Distribution
- Frequency distributions of actions
- Chi-square test results

#### 4.3.2 Critical Decision Points
- Identification of critical turns
- Action value distributions

### 4.4 Human Evaluation
This section presents the results from human playtesting sessions.

#### 4.4.1 Human vs Agent Performance
- Win rates against human players
- Qualitative feedback from participants

#### 4.4.2 Strategy Effectiveness
- Analysis of human-likeness in AI strategies
- Feedback on AI decision-making

### 4.5 Summary of Findings
This section provides a summary of the key findings from the experimental results.

- Overall performance comparison
- Insights into AI decision-making
- Implications for real-world applications
