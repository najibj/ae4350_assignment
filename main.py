from classes import Grid, Agent, train_agents, plot_training_progress, simulate_agent, visualize_simulation
from time import time
import logging
import random
import numpy as np
import matplotlib.pyplot as plt

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def reset_agents(agents, grid, epsilon, alpha, gamma):
    for agent in agents:
        agent.epsilon = epsilon
        agent.alpha = alpha
        agent.gamma = gamma
        np.zeros((grid.height, grid.width, 4))
    print('Agents reset to baseline')

def run_qlearning(seed, grid_parameters, learning_parameters, reward_values, episodes, agents_per_terminal=1, delta_pheromone = 0.05, pheromone_decay_rate=0.7, sensitivity=False, convergence_tol=0.0):
    
    set_random_seed(seed)

    # Set up logging
    logging.basicConfig(
        filename='q_training_log.txt',  # Log to a file named 'training_log.txt'
        filemode='w',                 # 'w' mode will overwrite the log file each run
        level=logging.INFO,           # Set the logging level (DEBUG, INFO, WARNING, etc.)
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger()

    # Initialize the grid
    width = grid_parameters[0]
    height = grid_parameters[1]
    n_terminals = grid_parameters[2]
    n_runways = grid_parameters[3]
    grid = Grid(width, height, n_terminals, n_runways, n_paths=3)

    grid.display_grid(True)

    # Agent learning properties
    epsilon = learning_parameters[0]  # Exploration rate
    alpha = learning_parameters[1]  # Learning rate
    gamma = learning_parameters[2]  # Discount factor

    # Reinforcement Learning Rewards
    reward_runway = reward_values[0]
    reward_path = reward_values[1]
    reward_visited = reward_values[2]
    reward_illegal = reward_values[3]

    # Initialize the agents
    agents = []
    for n in range(agents_per_terminal):
        for i,start_pos in enumerate(grid.terminals):
            agents.append(Agent((n_terminals*n+i), start_pos, grid, epsilon, alpha, gamma)) 

    # Train the agents
    if not sensitivity:
        start = time()
        rewards, steps, conv_episode = train_agents(grid, agents, episodes, reward_runway, reward_visited, reward_path, reward_illegal, delta_pheromone, pheromone_decay=pheromone_decay_rate, learning='Q')
        plot_training_progress(rewards, steps, conv_episode)
        stop = time()
        print(f'Training time for Q: {stop - start}')

        for c, agent in enumerate(agents):
            # agent = agents[0]  # Select one of the trained agents
            path = simulate_agent(agent, grid)
            visualize_simulation(path, grid, c, learning='Q', save_animation=True)

    else:
        rewards, steps, conv_episode = train_agents(grid, agents, episodes, reward_runway, reward_visited, reward_path, reward_illegal, delta_pheromone, pheromone_decay=pheromone_decay_rate, learning='Q', convergence_tolerance=convergence_tol)

    return rewards, steps, conv_episode

def run_SARSA(seed, grid_parameters, learning_parameters, reward_values, episodes, agents_per_terminal=1, delta_pheromone = 0.05, pheromone_decay_rate=0.7, sensitivity=False, convergence_tol=0.0):
    set_random_seed(seed)

    # Set up logging
    logging.basicConfig(
        filename='S_training_log.txt',  # Log to a file named 'training_log.txt'
        filemode='w',                 # 'w' mode will overwrite the log file each run
        level=logging.INFO,           # Set the logging level (DEBUG, INFO, WARNING, etc.)
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger()

    # Initialize the grid
    width = grid_parameters[0]
    height = grid_parameters[1]
    n_terminals = grid_parameters[2]
    n_runways = grid_parameters[3]
    grid = Grid(width, height, n_terminals, n_runways, n_paths=3)

    grid.display_grid(True)

    # Agent learning properties
    epsilon = learning_parameters[0]  # Exploration rate
    alpha = learning_parameters[1]  # Learning rate
    gamma = learning_parameters[2]  # Discount factor

    # Reinforcement Learning Rewards
    reward_runway = reward_values[0]
    reward_path = reward_values[1]
    reward_visited = reward_values[2]
    reward_illegal = reward_values[3]

    agents = []
    for n in range(agents_per_terminal):
        for i,start_pos in enumerate(grid.terminals):
            agents.append(Agent((n_terminals*n+i), start_pos, grid, epsilon, alpha, gamma)) 

    if not sensitivity:
        start = time()
        rewards, steps, conv_episode = train_agents(grid, agents, episodes, reward_runway, reward_visited, reward_path, reward_illegal, delta_pheromone, pheromone_decay=pheromone_decay_rate, learning='S')
        plot_training_progress(rewards, steps, conv_episode, learning='S')
        stop = time()
        print(f'Training time for SARSA: {stop - start}')

        for c, agent in enumerate(agents):
            # agent = agents[0]  # Select one of the trained agents
            print('starting simulation')
            path = simulate_agent(agent, grid)
            print('simulated')
            visualize_simulation(path, grid, c, learning='SARSA', save_animation=True)
            print(c)
    
    else:
        rewards, steps, conv_episode = train_agents(grid, agents, episodes, reward_runway, reward_visited, reward_path, reward_illegal, delta_pheromone, pheromone_decay_rate, learning='S', convergence_tolerance=convergence_tol)

    return rewards, steps, conv_episode

def sensitivity_analysis(seed, grid_parameters, learning_parameters, reward_values, parameter, values, pheromone_decay=0.7, episodes=100, learning = 'Q', history = False):
    results = {}
    agents_per_terminal = 1
    d_ph = 0.05
    for value in values:
        # Set the parameter value
        if parameter == 'epsilon':
            learning_parameters[0] = value
        elif parameter == 'alpha':
            learning_parameters[1] = value
        elif parameter == 'gamma':
            learning_parameters[2] = value
        elif parameter == 'decay_rate':
            pheromone_decay = value
        elif parameter == 'n_agents':
            agents_per_terminal = value
        elif parameter == 'delta_pheromone':
            d_ph = value
        print(f'Performing sensitivity analysis for {parameter} at value {value}.')
        # Train agents
        if learning == 'Q':
            rewards_history, steps_history, convergence_episode = run_qlearning(seed, grid_parameters, learning_parameters, reward_values, episodes, agents_per_terminal, d_ph, pheromone_decay, sensitivity=True, convergence_tol=0.001)
        else:
            rewards_history, steps_history, convergence_episode = run_SARSA(seed, grid_parameters, learning_parameters, reward_values, episodes, agents_per_terminal, d_ph, pheromone_decay, sensitivity=True, convergence_tol=0.001)
        if history:
            print('plotting history')
            plot_training_progress(rewards_history, steps_history, convergence_episode, learning=learning, folder='sensitivity_analysis/', suffix=f'_{parameter}_{value}')
        # Calculate average metrics
        avg_reward = rewards_history[-1]  # Last episode average reward
        avg_steps = steps_history[-1]  # Last episode average steps to goal
        
        # Store results
        results[value] = (avg_reward, avg_steps, convergence_episode)
    
    return results


# Random seed
seed = 10

# Grid parameters
width = 20
height = 20
n_terminals = 3
n_runways = 2
grid_parameters = [width, height, n_terminals, n_runways]

# Agent learning properties
epsilon = 0.2  # Exploration rate
alpha = 0.1  # Learning rate
gamma = 0.95  # Discount factor
learning_parameters = [epsilon, alpha, gamma]

# Reinforcement Learning Rewards
reward_runway = 100
reward_path = -1
reward_visited = -5
reward_illegal = -10
reward_values = [reward_runway, reward_path, reward_visited, reward_illegal]

agents_per_terminal = 1 # For swarm increase value

episodes = 50
run_qlearning(seed, grid_parameters, learning_parameters, reward_values, episodes, agents_per_terminal, pheromone_decay_rate=0.7, convergence_tol=0.001)

episodes = 50
run_SARSA(seed, grid_parameters, learning_parameters, reward_values, episodes, pheromone_decay_rate=0.7)

episodes_sensitivity = 100

params = ['epsilon', 'alpha', 'gamma', 'decay_rate', 'n_agents', 'delta_pheromone']
terms_dict = {
    'epsilon': 'Exploration Rate',
    'alpha': 'Learning Rate',
    'gamma': 'Discount Factor',
    'decay_rate': 'Decay Rate',
    'n_agents' : 'Agents per Terminal',
    'delta_pheromone' : 'Agent Pheromone Level '
}
param_values = [[0.1, 0.2, 0.3, 0.4], [0.05, 0.1, 0.15, 0.2], [0.8, 0.9, 0.95, 0.99], [0.5, 0.7, 0.8, 0.9], [1, 5, 10, 50], [0, 0.05, 0.1]]
start_idx = 5
end_idx = 5
method = 'Q'
for i, param in enumerate(params[start_idx:end_idx+1]):
    results = sensitivity_analysis(seed, grid_parameters, learning_parameters, reward_values, param, param_values[i+start_idx], pheromone_decay=0.7, episodes=episodes_sensitivity, learning=method, history=True)
    print(epsilon)
    print(alpha)
    print(gamma)


    parameter_name = terms_dict.get(param, param)

    # Plot the results
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.plot(param_values[i+start_idx], [results[val][0] for val in param_values[i+start_idx]], marker='o')
    plt.title(f'Average Total Reward vs {parameter_name}')
    plt.xlabel(param)
    plt.ylabel('Average Total Reward')
    plt.grid()

    plt.subplot(1, 3, 2)
    plt.plot(param_values[i+start_idx], [results[val][1] for val in param_values[i+start_idx]], marker='o')
    plt.title(f'Average Steps to Goal vs {parameter_name}')
    plt.xlabel(param)
    plt.ylabel('Average Steps to Goal')
    plt.grid()

    plt.subplot(1, 3, 3)
    plt.plot(param_values[i+start_idx], [results[val][2] for val in param_values[i+start_idx]], marker='o')
    plt.title(f'Convergence Episode vs {parameter_name}')
    plt.xlabel(param)
    plt.ylabel('Convergence Episode')
    plt.grid()

    plt.tight_layout()
    plt.savefig(f'sensitivity_analysis/{method}/sensitivity_{param}.png')
    plt.close()

    