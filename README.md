## Overview

This project simulates an agent-based model using a grid environment. The simulation involves agents navigating through a grid with defined terminals and runways, utilizing a reinforcement learning algorithm with pheromone-based pathfinding. The project is divided into two main components: grid and agent definitions (in `classes.py`) and the main simulation and training routines (in `main.py`).

## Files

### `classes.py`

This file contains the core classes and functions required for the simulation:

- **`Grid`**: Defines the environment in which agents operate. It includes:
  - **Attributes**:
    - `width` and `height`: Dimensions of the grid.
    - `n_terminals`, `n_runways`, `n_paths`: Number of terminals, runways, and paths.
    - `grid`: The grid itself, generated using the `generate_grid()` method.
    - `pheromone_grid`: A grid that tracks pheromone levels, used by agents for navigation.
  - **Methods**:
    - `generate_grid()`: Generates the grid layout based on the provided parameters.
    - Other utility methods for grid management and interaction with agents.
- **`Agent`**: Represents an individual agent in the grid environment. It includes:
  - **Attributes**:
    - `id`: A unique identifier for each agent.
    - `start_pos`: The starting position of the agent on the grid.
    - `position`: The current position of the agent, which is initially set to `start_pos`.
    - `grid`: A reference to the `Grid` object in which the agent operates.
    - `q_table`: A 3D NumPy array that holds the Q-values for each state-action pair. The dimensions are `(grid.height, grid.width, 4)`, where `4` represents the four possible actions (North, South, West, East).
    - `epsilon`: The exploration rate, which determines the probability of the agent choosing a random action (exploration) versus the action with the highest Q-value (exploitation).
    - `alpha`: The learning rate, controlling how much new information overrides the old information in the Q-value update.
    - `gamma`: The discount factor, which determines the importance of future rewards.
  - **Actions**:
    - The agent can take one of four actions corresponding to moving North (`'N'`), South (`'S'`), West (`'W'`), or East (`'E'`).
  - **Methods**:
    - `choose_action(self)`: The agent selects an action based on the epsilon-greedy policy. It either chooses the best action according to the Q-table or explores a random action based on the value of `epsilon`.
    - `learn(self, current_state, action, reward, next_state)`: This method updates the Q-value for a given state-action pair using the Q-learning formula:
    \[
    Q(s, a) = Q(s, a) + lpha 	imes \left( 	ext{reward} + \gamma 	imes \max_{a'} Q(s', a') - Q(s, a) 
ight)
    \]
    where `s` is the current state, `a` is the action taken, `s'` is the next state, and `a'` represents the possible actions from `s'`.
    - `reset(self)`: Resets the agent's position to the starting position and optionally resets the Q-table and other learning parameters.
    - `update_position(self, action)`: Updates the agent's position based on the action taken.

- **Additional functions** related to agent behavior and interaction with the grid, training routines, and visualization tools.

### `main.py`

This is the main script that runs the simulation:

- **`set_random_seed(seed)`**: Sets the random seed for reproducibility.
- **Simulation and Training**:
  - The script initializes the grid and agents, sets up the simulation parameters, and runs the training and simulation loops.
  - It also includes plotting functions to visualize the training progress and agent paths.

## Installation

To run this project, you'll need Python 3.x installed along with the following libraries:

```bash
pip install numpy matplotlib
```

## Usage

1. **Set Up the Environment**:
   - Customize the grid dimensions, number of terminals, runways, and other parameters in `main.py`.
   
2. **Run the Simulation**:
   - Execute `main.py` to start the training and simulation. For example:
   
   ```bash
   python main.py
   ```

3. **Visualize the Results**:
   - The script will generate plots showing the training progress and the paths taken by agents on the grid.

## Customization

- You can modify the grid size, agent parameters (e.g., epsilon, alpha, gamma), and other settings directly in the `main.py` script to fit your specific use case.

## Logging

The project uses Python's `logging` module to record key events during the simulation. You can adjust the logging level or redirect the output to a file by modifying the logging configuration in `main.py`.
