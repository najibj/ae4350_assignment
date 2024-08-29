import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.colors as mcolors
import logging
import os

class Grid:
    def __init__(self, width, height, n_terminals, n_runways, n_paths=4):
        self.width = width
        self.height = height
        self.n_terminals = n_terminals
        self.n_runways = n_runways
        self.n_paths = n_paths
        self.grid = self.generate_grid()
        self.pheromone_grid = np.zeros((self.height, self.width), dtype=float)
        self.value_grid = np.zeros((self.height, self.width), dtype = float)

    def generate_grid(self):
        # Initialize an empty grid
        grid = np.zeros((self.height, self.width), dtype=int)
        
        # Place runway on the grid
        self.runways = self.place_runways(grid)

        # Place terminals on the grid
        self.terminals = self.place_terminals(grid)
        
        # Connect terminals to runway with paths
        self.create_paths(grid, self.terminals, self.runways)
        
        return grid

    def place_terminals(self, grid):
        terminals = []
        for _ in range(self.n_terminals):
            ok = False
            while not ok:
                row = random.randint(0, self.height - 1)
                col = random.randint(0, self.width - 1)
                ok = True  # Assume it's okay initially
                for r, c in self.runways:
                    if row == r or col == c:
                        ok = False  # Found a conflict, set ok to False
                        break  # Exit the loop early as we found a conflict
            grid[row][col] = 3
            terminals.append((row, col))
        return terminals

    def place_runways(self, grid):
        runways = []
        for _ in range(self.n_runways):
            edge = random.randint(1,4)
            if edge == 1:
                row = 0
                col = random.randint(0, self.width - 1)
            elif edge == 3:
                row = self.height - 1
                col = random.randint(0, self.width - 1)
            elif edge == 2:
                row = random.randint(0, self.height - 1)
                col = self.width - 1
            elif edge == 4:
                row = random.randint(0, self.height - 1)
                col = 0
            grid[row][col] = 5  # Use a different value for runway
            runways.append((row,col))
        return runways

    def create_paths(self, grid, terminals, runways):
        for runway in runways:
            for terminal in terminals:
                for _ in range(self.n_paths):
                    self.create_path(grid, terminal, runway, random_path=True)

    def create_path(self, grid, start, end, random_path):
        current = start
        while current != end:
            next_step = self.next_step(current, end, random_path=random_path)
            if grid[next_step] == 0:  # Only overwrite off-limits areas
                grid[next_step] = 1
            current = next_step

    def next_step(self, current, end, random_path):
        row, col = current
        end_row, end_col = end
        if not random_path:
            if row < end_row:
                row += 1
            elif row > end_row:
                row -= 1
            elif col < end_col:
                col += 1
            elif col > end_col:
                col -= 1
        else:
            step = random.randint(1,10)
            if row < end_row:
                if step in range(9,10) and row>0:
                    row -= 1
                elif step in range(5,9):
                    if col == 0:
                        s = 1
                    elif col == self.width-1:
                        s = -1
                    else:
                        s = random.choice([-1,1])
                    col += s
                else:
                    row +=1
            elif row > end_row:
                if step in range(9,10) and row<self.height-1:
                    row += 1
                elif step in range(5,9):
                    if col == 0:
                        s = 1
                    elif col == self.width-1:
                        s = -1
                    else:
                        s = random.choice([-1,1])
                    col += s
                else:
                    row -=1
            elif col < end_col:
                if step in range(9,10) and col>0:
                    col -= 1
                elif step in range(5,9):
                    if row == 0:
                        s = 1
                    elif row == self.height-1:
                        s = -1
                    else:
                        s = random.choice([-1,1])
                    row += s
                else:
                    col +=1
            elif col > end_col:
                if step in range(9,10) and col<self.width-1:
                    col += 1
                elif step in range(5,9):
                    if row == 0:
                        s = 1
                    elif row == self.height-1:
                        s = -1
                    else:
                        s = random.choice([-1,1])
                    row += s
                else:
                    col -=1           
        return (row, col)
    
    def reset_pheromone_grid(self):
        self.pheromone_grid = np.zeros((self.height, self.width), dtype=float)

    def display_grid(self, save_grid = False):
        # Define a custom colormap
        cmap = mcolors.ListedColormap(['black', 'white', 'gray', 'lightgray', 'darkgray'])
        bounds = [0, 1, 2, 3, 4, 5]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        plt.imshow(self.grid, cmap=cmap, norm=norm)
        plt.title("Grid Layout")
        if save_grid:
            plt.savefig('Current_grid.png')
        # plt.show()

    def display_pheromone_grid(self, episode, save_grid = False, learning = 'Q'):
        plt.imshow(self.pheromone_grid, cmap='YlOrRd', interpolation='nearest')
        plt.title(f"Pheromone Grid Episode {episode}")
        plt.colorbar()
        if save_grid:
            folder = learning
            if not os.path.exists(folder):
                os.mkdir(folder)
                print(f'Directory {folder} made')
            folder = f'{learning}/Pheromone_maps'
            if not os.path.exists(folder):
                os.mkdir(folder)
                print(f'Directory {folder} made')
            if learning == 'Q':
                plt.savefig(f'Q/Pheromone_maps/pheromone_map_{episode}.png')
            else:
                plt.savefig(f'SARSA/Pheromone_maps/pheromone_map_{episode}.png')
        plt.close()
        # plt.show()

class Agent:
    def __init__(self, id, start_pos, grid, epsilon, alpha, gamma):
        self.id = id
        self.start_pos = start_pos
        self.position = start_pos
        self.grid = grid
        self.q_table = np.zeros((grid.height, grid.width, 4))  # Q-values for each state-action pair
        self.epsilon = epsilon  # Exploration rate
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.actions = ['N', 'S', 'W', 'E']  # Possible actions
        self.visited_states = set()

    def reset(self):
        self.position = self.start_pos
        self.visited_states = set()

    def choose_action(self):
        if random.uniform(0, 1) < self.epsilon:
            # print('Random Action')
            return random.choice(self.actions)  # Explore: random action
        else:
            row, col = self.position
            # print('Best Action')
            return self.actions[np.argmax(self.q_table[row, col])]  # Exploit: best action

    def take_action(self, action):
        row, col = self.position
        if action == 'N' and row > 0:
            row -= 1
        elif action == 'S' and row < self.grid.height - 1:
            row += 1
        elif action == 'W' and col > 0:
            col -= 1
        elif action == 'E' and col < self.grid.width - 1:
            col += 1
        return (row, col)

    def update_q_table(self, old_pos, action, reward, new_pos, next_action=None, learning='Q'):
        row, col = old_pos
        new_row, new_col = new_pos
        action_idx = self.actions.index(action)
        if learning == 'S' and next_action is not None:
            next_action_idx = self.actions.index(next_action)
            target = reward + self.gamma * self.q_table[new_row, new_col, next_action_idx]
        else:
            target = reward + self.gamma * np.max(self.q_table[new_row, new_col])
        self.q_table[row, col, action_idx] += self.alpha * (target - self.q_table[row, col, action_idx])

def plot_value_function(agent, episode, save_grid=False, learning = 'Q'):
    value_function = np.max(agent.q_table, axis=2)  # Calculate the value function as the max Q-value for each state
    plt.imshow(value_function, cmap='hot', interpolation='nearest')
    plt.title(f'Value Function - Episode {episode}')
    plt.colorbar()
    if save_grid:
        folder = learning
        if not os.path.exists(folder):
            os.mkdir(folder)
            print(f'Directory {folder} made')
        folder = f'{learning}/Value_functions'
        if not os.path.exists(folder):
            os.mkdir(folder)
            print(f'Directory {folder} made')
        if learning == 'Q':
            plt.savefig(f'Q/Value_functions/value_function_episode_{episode}_agent_{agent.id}.png')
        else:
            plt.savefig(f'SARSA/Value_functions/value_function_episode_{episode}_agent_{agent.id}.png')
    # plt.show()
    plt.close()

def train_agents(grid, agents, episodes, r_runway, r_visited, r_path, r_ill, delta_pheromone=0.05, pheromone_decay=0.9, learning='Q', convergence_tolerance=0.0):
    logging.info(f'Starting training with {learning} method for {episodes} episodes.')
    
    rewards_history = []
    steps_history = []

    convergence_episode = None

    if learning == 'Q': 
        # Q Learning
        for episode in range(episodes):
            logging.info(f'Starting Episode {episode + 1}')
            episode_rewards = []
            episode_steps = []
            for agent in agents:
                print(f'Episode {episode}, agent {agent.id}')
                agent.reset()
                total_reward = 0
                steps = 0
                while agent.position not in grid.runways:
                    old_pos = agent.position
                    agent.visited_states.add(old_pos)
                    action = agent.choose_action()
                    logging.debug(f'Agent {agent.id} at {old_pos} chose action {action}')
                    new_pos = agent.take_action(action)
                    pheromone_level = grid.pheromone_grid[new_pos]
                    if grid.grid[new_pos] == 5:
                        reward = r_runway
                    elif grid.grid[new_pos] == 1 or grid.grid[new_pos] == 4:
                        if new_pos in agent.visited_states:
                            reward += r_visited
                        else:
                            reward = r_path + pheromone_level
                    else:
                        reward = r_ill
                        new_pos = old_pos
                    agent.update_q_table(old_pos, action, reward, new_pos)
                    agent.position = new_pos
                    total_reward += reward
                    steps += 1
                    grid.pheromone_grid[old_pos] += delta_pheromone
                    logging.debug(f'Agent {agent.id} moved to {new_pos} with reward {reward}')
                    if steps > grid.width * grid.height:
                        logging.warning(f'Agent {agent.id} exceeded max steps. Breaking out of loop.')
                        break
                grid.pheromone_grid *= pheromone_decay
                episode_rewards.append(total_reward)
                episode_steps.append(steps)
                logging.info(f'Episode {episode + 1}, Agent {agents.index(agent) + 1}, Total Reward: {total_reward}, Steps: {steps}')
            grid.display_pheromone_grid(episode + 1, True)
            rewards_history.append(np.mean(episode_rewards))
            steps_history.append(np.mean(episode_steps))
            if episode >= 100 and convergence_tolerance != 0.0:
                recent_avg_steps = np.mean(steps_history[-10:])
                previous_avg_steps = np.mean(steps_history[-20:-10])
                diff = np.abs((recent_avg_steps - previous_avg_steps)/previous_avg_steps)
                # print(f"Discrepency {diff}")
                if diff < convergence_tolerance and recent_avg_steps < 20:
                    convergence_episode = episode+1
                    break
        for agent in agents:
            plot_value_function(agent, episodes, True)
    else:
        # SARSA
        for episode in range(episodes):
            logging.info(f'Starting Episode {episode + 1}')
            episode_rewards = []
            episode_steps = []
            for agent in agents:
                agent.reset()
                total_reward = 0
                steps = 0
                old_pos = agent.position
                action = agent.choose_action()
                logging.debug(f'Agent {agent.id} at {old_pos} chose action {action}')
                while agent.position not in grid.runways:
                    new_pos = agent.take_action(action)
                    agent.visited_states.add(old_pos)
                    pheromone_level = grid.pheromone_grid[new_pos]
                    if grid.grid[new_pos] == 5:
                        reward = r_runway
                    elif grid.grid[new_pos] == 1 or grid.grid[new_pos] == 4:
                        if new_pos in agent.visited_states:
                            reward = r_visited
                        else:
                            reward = r_path + pheromone_level
                    else:
                        reward = r_ill
                        new_pos = old_pos

                    next_action = agent.choose_action()
                    agent.update_q_table(old_pos, action, reward, new_pos, next_action, learning='S')
                    logging.debug(f'Agent {agent.id} moved to {new_pos} with reward {reward}')
                    old_pos = new_pos
                    action = next_action
                    agent.position = new_pos
                    total_reward += reward
                    steps += 1
                    grid.pheromone_grid[old_pos] += 0.1
                    if steps > grid.width * grid.height:
                        logging.warning(f'Agent {agent.id} exceeded max steps. Breaking out of loop.')
                        break
                grid.pheromone_grid *= pheromone_decay
                episode_rewards.append(total_reward)
                episode_steps.append(steps)
                logging.info(f'Episode {episode + 1}, Agent {agents.index(agent) + 1}, Total Reward: {total_reward}, Steps: {steps}')
            grid.display_pheromone_grid(episode + 1, False, learning='S')
            rewards_history.append(np.mean(episode_rewards))
            steps_history.append(np.mean(episode_steps))
            if episode >= 100 and convergence_tolerance != 0.0:
                recent_avg_steps = np.mean(steps_history[-10:])
                previous_avg_steps = np.mean(steps_history[-20:-10])
                diff = np.abs((recent_avg_steps - previous_avg_steps)/previous_avg_steps)
                # print(f"Discrepency {diff}")
                if diff < convergence_tolerance and recent_avg_steps < 20:
                    convergence_episode = episode+1
                    break
        for agent in agents:
            plot_value_function(agent, episodes, True, 'S')

    if convergence_episode is None:
        convergence_episode = episodes
    logging.info(f'Finished training with {learning} method.')
    return rewards_history, steps_history, convergence_episode

def train_agents_swarm(grid, agents, episodes, r_runway, r_visited, r_path, r_ill, pheromone_decay=0.9, learning='Q', convergence_tolerance=0.0):
    logging.info(f'Starting training with {learning} method for {episodes} episodes.')
    
    rewards_history = []
    steps_history = []

    convergence_episode = None

    if learning == 'Q': 
        # Q Learning
        for episode in range(episodes):
            logging.info(f'Starting Episode {episode + 1}')
            episode_rewards = []
            episode_steps = []
            for agent in agents:
                agent.reset()
                total_reward = 0
                steps = 0
                while agent.position not in grid.runways:
                    old_pos = agent.position
                    agent.visited_states.add(old_pos)
                    action = agent.choose_action()
                    logging.debug(f'Agent {agent.id} at {old_pos} chose action {action}')
                    new_pos = agent.take_action(action)
                    pheromone_level = grid.pheromone_grid[new_pos]
                    if grid.grid[new_pos] == 5:
                        reward = r_runway
                    elif grid.grid[new_pos] == 1 or grid.grid[new_pos] == 4:
                        if new_pos in agent.visited_states:
                            reward += r_visited
                        else:
                            reward = r_path + pheromone_level
                    else:
                        reward = r_ill
                        new_pos = old_pos
                    agent.update_q_table(old_pos, action, reward, new_pos)
                    agent.position = new_pos
                    total_reward += reward
                    steps += 1
                    grid.pheromone_grid[old_pos] += 0.05
                    logging.debug(f'Agent {agent.id} moved to {new_pos} with reward {reward}')
                    if steps > grid.width * grid.height:
                        logging.warning(f'Agent {agent.id} exceeded max steps. Breaking out of loop.')
                        break
                grid.pheromone_grid *= pheromone_decay
                episode_rewards.append(total_reward)
                episode_steps.append(steps)
                logging.info(f'Episode {episode + 1}, Agent {agents.index(agent) + 1}, Total Reward: {total_reward}, Steps: {steps}')
            grid.display_pheromone_grid(episode + 1, True)
            rewards_history.append(np.mean(episode_rewards))
            steps_history.append(np.mean(episode_steps))
            if episode >= 50 and convergence_tolerance != 0.0:
                recent_avg_steps = np.mean(steps_history[-10:])
                previous_avg_steps = np.mean(steps_history[-20:-10])
                diff = np.abs((recent_avg_steps - previous_avg_steps)/previous_avg_steps)
                # print(f"Discrepency {diff}")
                if diff < convergence_tolerance and recent_avg_steps < 40:
                    convergence_episode = episode+1
                    break
        for agent in agents:
            plot_value_function(agent, episodes, True)
    else:
        # SARSA
        for episode in range(episodes):
            logging.info(f'Starting Episode {episode + 1}')
            episode_rewards = []
            episode_steps = []
            for agent in agents:
                agent.reset()
                total_reward = 0
                steps = 0
                old_pos = agent.position
                action = agent.choose_action()
                logging.debug(f'Agent {agent.id} at {old_pos} chose action {action}')
                while agent.position not in grid.runways:
                    new_pos = agent.take_action(action)
                    agent.visited_states.add(old_pos)
                    pheromone_level = grid.pheromone_grid[new_pos]
                    if grid.grid[new_pos] == 5:
                        reward = r_runway
                    elif grid.grid[new_pos] == 1 or grid.grid[new_pos] == 4:
                        if new_pos in agent.visited_states:
                            reward = r_visited
                        else:
                            reward = r_path + pheromone_level
                    else:
                        reward = r_ill
                        new_pos = old_pos

                    next_action = agent.choose_action()
                    agent.update_q_table(old_pos, action, reward, new_pos, next_action, learning='S')
                    logging.debug(f'Agent {agent.id} moved to {new_pos} with reward {reward}')
                    old_pos = new_pos
                    action = next_action
                    agent.position = new_pos
                    total_reward += reward
                    steps += 1
                    grid.pheromone_grid[old_pos] += 0.1
                    if steps > grid.width * grid.height:
                        logging.warning(f'Agent {agent.id} exceeded max steps. Breaking out of loop.')
                        break
                grid.pheromone_grid *= pheromone_decay
                episode_rewards.append(total_reward)
                episode_steps.append(steps)
                logging.info(f'Episode {episode + 1}, Agent {agents.index(agent) + 1}, Total Reward: {total_reward}, Steps: {steps}')
            grid.display_pheromone_grid(episode + 1, False, learning='S')
            rewards_history.append(np.mean(episode_rewards))
            steps_history.append(np.mean(episode_steps))
            if episode >= 50 and convergence_tolerance != 0.0:
                recent_avg_steps = np.mean(steps_history[-10:])
                previous_avg_steps = np.mean(steps_history[-20:-10])
                diff = np.abs((recent_avg_steps - previous_avg_steps)/previous_avg_steps)
                # print(f"Discrepency {diff}")
                if diff < convergence_tolerance and recent_avg_steps < 40:
                    convergence_episode = episode+1
                    break
        for agent in agents:
            plot_value_function(agent, episodes, True, 'S')

    if convergence_episode is None:
        convergence_episode = episodes
    logging.info(f'Finished training with {learning} method.')
    return rewards_history, steps_history, convergence_episode

def plot_training_progress(rewards_history, steps_history, episodes, learning = 'Q', folder='', suffix=''):
    # average_rewards = np.mean(rewards_history, axis=1)
    # average_steps = np.mean(steps_history, axis=1)

    average_rewards = rewards_history

    average_steps = steps_history
    
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, episodes + 1), average_rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Average Total Reward')
    plt.title('Average Total Reward over Episodes')

    plt.subplot(1, 2, 2)
    plt.plot(range(1, episodes + 1), average_steps)
    plt.xlabel('Episodes')
    plt.ylabel('Average Steps to Goal')
    plt.title('Average Steps to Goal over Episodes')

    if folder != '':
        if not os.path.exists(folder):
            os.mkdir(folder)
            print(f'Directory {folder} made')
        if learning == 'Q': 
            sub_folder = f'{folder}/Q'
        else:
            sub_folder = f'{folder}/SARSA'
        if not os.path.exists(sub_folder):
            os.mkdir(sub_folder)
            print(f'Directory {sub_folder} made')
    
    plt.tight_layout()
    if learning == 'Q':
        plt.savefig(f'{folder}Q/reward_and_step_history{suffix}.png')
    else:
        plt.savefig(f'{folder}SARSA/reward_and_step_history{suffix}.png')
    print('History plotted')
    plt.close()

def simulate_agent(agent, grid, max_steps=100):
    agent.reset()
    path = [agent.position]  # Track the path the agent takes
    steps = 0
    
    while agent.position not in grid.runways and steps < max_steps:
        row, col = agent.position
        action = agent.actions[np.argmax(agent.q_table[row, col])]  # Choose the best action based on the Q-table
        new_pos = agent.take_action(action)
        
        if grid.grid[new_pos] == 5:  # If the agent reaches the runway
            path.append(new_pos)
            break
        elif grid.grid[new_pos] == 1 or grid.grid[new_pos] == 4:  # Valid path
            agent.position = new_pos
            path.append(new_pos)
        else:  # Invalid move, stay in place
            print("Makes invalid move, break")
            break
        
        steps += 1
    
    return path

def visualize_simulation(path, grid, suffix, learning='Q', save_animation=False):
    fig, ax = plt.subplots()
    cmap = mcolors.ListedColormap(['black', 'white', 'gray', 'lightgray', 'darkgray'])
    bounds = [0, 1, 2, 3, 4, 5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    ax.imshow(grid.grid, cmap=cmap, norm=norm)
    
    x_coords = [pos[1] for pos in path]
    y_coords = [pos[0] for pos in path]
    
    ax.plot(x_coords, y_coords, marker='o', color='red', linestyle='-', linewidth=2, markersize=5)
    
    ax.set_title(f"Agent's Path from Terminal to Runway using {learning} learning")
    
    if save_animation:
        plt.savefig(f'{learning}/agent_simulation_{suffix}.png')
    plt.show()
    plt.close()
    
