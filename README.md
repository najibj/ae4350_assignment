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
- **Additional classes (e.g., `Agent`) and functions** related to agent behavior and interaction with the grid, training routines, and visualization tools.

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
