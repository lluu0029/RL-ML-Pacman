## Q1 Value Iteration
The agent uses value iteration to solve the Markov Decision Process (MDP) defined in the PacmanMDP class, and then use the learnt values to play an MDP 
variant of the Pac-Man game to find the optimal path to a positive terminal (food dot) whilst avoiding negative terminals (ghosts). 
Implementation in Q1Agent class.
- **Using command line for a chosen layout, discount factor and number of iterations:**
```bash
python pacman.py -l layouts/VI_smallMaze1_1.lay -p Q1Agent -a discount=1,iterations=100 -g StationaryGhost -n 20
```

## Q2 Q-Learning
The agent uses Q-learning with epsilon greedy action selection to learn an optimal policy for reaching a positive terminal (food dot) whilst avoiding 
negative terminals (ghosts) through a series of trial and error interactions with its environment.
Implementation in Q2Agent class.
- **Using command line for a chosen layout and sample parameters:**
```bash
python pacman.py -l layouts/QL_tinyMaze1_1.lay -p Q2Agent -a epsilon=0.01,alpha=0.6,gamma=0.9 -x 100 -n 200 -g StationaryGhost
```

## Q3 Machine Learning
A single-layer perceptron was trained using a provided dataset that includes actions within a specific game state, labeled as either 0 (weak action) or 1 (strong action). 
The model was designed to predict the optimal action based on a set of features extracted from a game state. When applied to play Pacman, it achieved a win rate of 71.75% across various game layouts.
Implementation in PerceptronPacman class.
- **Using command line for a chosen layout and sample weights:**
```bash
python pacman.py -l layouts/ML_mediumClassic.lay -p Q3Agent -a weights_path="models\q3_weights.model"
```
- **Using command line for model training:**
```bash
python trainPerceptron.py -i num_iterations -l alpha -w weight_save_path
```
