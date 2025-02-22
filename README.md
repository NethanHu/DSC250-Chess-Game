# Advanced Strategic Decision-Making in Chess Using Monte Carlo Tree Search

Author: Selena Ge, Riqian Hu, Liyuan Jin, Letong Liang

### Abstract
  Chess has long been a cornerstone for evaluating artificial intelligence due to its complexity, requirement for strategic foresight, and immense decision space. While traditional chess engines have relied heavily on heuristic-based evaluations, recent advancements have increasingly incorporated deep learning techniques to improve performance. This research aims to introduce Monte Carlo Tree Search (MCTS) as a promising approach for enhancing strategic decision-making in chess. MCTS has demonstrated significant success in other gaming domains due to its advanced simulation capabilities, proven effectiveness, and high interpretability compared to purely black-box models.
By leveraging MCTS, we aim to improve both the interpretability and performance of decision-making in chess gameplay. Specifically, this study will evaluate MCTS's ability to simulate multiple future states, optimize moves across various game scenarios, and achieve competitive results. The evaluation will be conducted using specific performance metrics (to be determined), offering insights into its strengths and limitations as a decision-making framework in high-complexity environments.

## 1. Literature Review
The success of AlphaGo (Silver et al., 2016) in defeating human Go champions highlighted the effectiveness of Monte Carlo Tree Search (MCTS) in complex decision-making, particularly in strategic board games. This milestone prompted further research into MCTS for other domains, including chess, which also features a vast state space and deep strategic complexity. By balancing exploration and exploitation through randomized simulations, MCTS offers a flexible alternative to classical search algorithms like alpha-beta (Coulom, 2006; Kocsis \& Szepesvári, 2006). Its adaptability has made MCTS central to modern game AI, as evidenced by AlphaZero’s superhuman performance in chess, shogi, and Go (Silver et al., 2018).

MCTS combines tree search with stochastic sampling to efficiently traverse large search spaces. Browne et al. (2012) identify four core steps—selection, expansion, simulation, and backpropagation—that balance exploration (examining unvisited or less-visited states) and exploitation (focusing on high-reward states). Variants such as Upper Confidence Bounds for Trees (UCT) (Kocsis \& Szepesvári, 2006) and heuristic-guided simulations further enhance performance. While scalability remains challenging in high-dimensional domains, ongoing research continues to refine MCTS for a broad range of applications (Browne et al., 2012).

MCTS also offers interpretability through its tree-based decision process. Gao et al. (2024) introduce SC-MCTS*, a contrastive MCTS framework aimed at improving both efficiency and interpretability in Large Language Model (LLM) reasoning. By integrating contrastive decoding into the reward model, SC-MCTS* refines decision-making while preserving visibility into search trajectories. Adjustments to UCT and backpropagation further clarify node selection and value propagation, as demonstrated by experiments on the Blocksworld dataset.

Beyond board games, MCTS has proven useful in Neural Architecture Search (NAS). Wang et al. (2018) propose AlphaX, which uses MCTS guided by a Meta-Deep Neural Network to identify promising regions of the architecture search space. By employing a distributed design and transfer learning, AlphaX reduces computational overhead and achieves state-of-the-art results on CIFAR-10 and ImageNet, showcasing MCTS’s versatility in complex tasks.
In chess, MCTS’s scalability, flexibility, and interpretability make it an attractive option for advanced strategic decision-making. Traditional chess engines often rely on alpha-beta pruning and handcrafted evaluation functions, but MCTS can offer a more adaptive, powerful method for move selection (Silver et al., 2018). As ongoing research explores heuristic enhancements, neural-network rollouts, and contrastive analysis, MCTS-driven chess engines stand to improve not only in performance but also in transparency and strategic sophistication.

## 2. Methodology
### 2.1 Game Representation
Utilize a bitboard representation for efficient computation of legal moves and state transitions while leveraging the python-chess library to handle game rules and state transitions.

### 2.2 MCTS Implementation
The MCTS algorithm will be adapted to chess gameplay, where the tree structure consists of nodes representing individual chess positions, edges representing possible moves from each position, and a root node corresponding to the current game state. Each iteration of MCTS follows four key stages:
- **Selection**: The UCT algorithm is used to balance exploration and exploitation during the selection phase. The formula is given by:
$$
UCT = \frac{Q(v')}{N(v')} + c \sqrt{\frac{\ln N(v)}{N(v')}}
$$
where:
$ Q(v') $ denotes the total reward from node $ v' $, $ N(v')$ denotes the number of visits to node $ v' $, $N(v)$ denotes the number of visits to the parent node $v$, $c$ is a constant that controls the exploration-exploitation trade-off.

- **Expansion**: when a leaf node is reached, it is expanded by generating all possible legal moves from the current position and creating corresponding child nodes, utilizing the python-chess library to generate legal moves and update the game state.
- **Simulation**: perform random playouts from the newly expanded node to a terminal state, such as checkmate, stalemate, or draw, while using a lightweight evaluation function to guide the random moves toward more promising outcomes.
- **Backpropagation**: after reaching a terminal state, the result is propagated back up the tree, updating the visit count and reward values for each node along the path from the terminal node to the root node.

## 3. Evaluation Plan
### 3.1 Performance Metrics
- **Average move quality**: ompare the moves suggested by the MCTS algorithm with expert moves from the Kaggle dataset.
- **Win rate**: evaluate the win rate of MCTS-based engine against traditional chess engines.
- **Computational Cost**: measure the time and resources required per move to assess the algorithm's efficiency.

### 3.2 Testing Scenarios
We test the algorithm across various chess scenarios, including **openings**, and **midgame tactics**, **endgames**. This evaluation helps assess the adaptability and performance of the algorithm across different phases of the game.

### 3.3 Dataset
We choose **Kaggle Chess Dataset** as the datasets, which contains over 20,000 chess games in PGN format, including player ratings and game outcomes, providing a rich source of strategic decisions for training and evaluation while facilitating comparative analysis with human gameplay.

### 3.4 Tools and Libraries
We use Python libraries such as python-chess, numpy, pandas, matplotlib, and pytorch for implementation, utilizing CPU or GPU as needed for computation.
