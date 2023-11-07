from treelib import Tree
import copy
import random
import numpy as np
from connectfour import ConnectFour

class MCTS:

    def __init__(self, env):
        self.org_env = ConnectFour(
            'rgb_array')  # New env and copy of board_state and current_player, otherwise we would have the wrappers as well (e.g. recording video)
        self.org_env.board_state = copy.deepcopy(env.board_state)
        self.org_env.current_player = copy.deepcopy(env.current_player)
        self.search_tree = Tree()
        self.root_node = self.search_tree.create_node('root', data={
            'current_player': copy.deepcopy(self.org_env.current_player),
            'state': copy.deepcopy(self.org_env.board_state), 'returns': 0, 'visits': 0, 'wins': 0})

    def run_mcts(self, n_iterations, c):
        self.c = c

        for i in range(n_iterations):
            self.sim_env = copy.deepcopy(
                self.org_env)  # For each iteration, we need to initizalize the env with the original env, otherwise the board state is wrong
            self.selected_node, reward, terminated = self.select(self.root_node, -np.inf,
                                                                 False)  # Select node in tree that is not fully expanded

            if not terminated:
                self.start_node, reward, terminated = self.expand(
                    self.selected_node)  # Expand selected node and add a child node
                reward = self.simulate(self.start_node, reward,
                                       terminated)  # Run a random simulation from newly expanded child node until episode end
            else:
                self.start_node = self.selected_node  # No expansion took place, i.e. node from where to start back propagation is selected_node

            self.backup(self.start_node,
                        reward)  # Backpropagate reward from simulation start node to root (i.e. node of current board state)

        return self.select_best_action()

    def select(self, current_node, reward, terminated):
        n_childrens = len(current_node.successors(self.search_tree.identifier))
        n_possible_actions = len(self.sim_env.get_possible_actions())

        # Check if selected node is a terminating node
        if terminated:
            return current_node, reward, terminated

        # Current node not fully expanded and not terminated
        elif n_childrens < n_possible_actions:
            return current_node, reward, terminated

        # Current node fully expanded, but not terminated -> go one level deeper based on tree policy ucb1
        else:
            max_ucb1 = -np.inf

            for child_identifier in current_node.successors(self.search_tree.identifier):
                child_node = self.search_tree.get_node(child_identifier)
                avg_return = child_node.data['returns'] / child_node.data['visits']
                ucb1 = avg_return + self.c * np.sqrt(np.log(current_node.data['visits']) / child_node.data['visits'])

                if ucb1 > max_ucb1:
                    max_child_node = child_node
                    max_ucb1 = ucb1

            child_action = int(max_child_node.tag)
            _, reward, terminated, _, _ = self.sim_env.step(
                child_action)  # Perform action of child node in sim_env and check if env is terminated after selection

            # Select existing child node randomly
            # child_identifier = random.choice(current_node.successors(self.search_tree.identifier))
            # child_node = self.search_tree.get_node(child_identifier)
            # child_action = int(child_node.tag)
            # self.sim_env.step(child_action) # Perform action of child node in sim_env

            return self.select(max_child_node, reward, terminated)

    def expand(self, selected_node):
        child_actions = []

        for identifier in selected_node.successors(self.search_tree.identifier):
            child_actions.append(int(self.search_tree.get_node(identifier).tag))

        remaining_actions = [action for action in self.sim_env.get_possible_actions() if action not in child_actions]

        if len(remaining_actions) > 0:
            action = random.choice(remaining_actions)
            _, reward, terminated, _, _ = self.sim_env.step(action)
            child_node = self.search_tree.create_node(f'{action}', parent=selected_node.identifier, data={
                'current_player': copy.deepcopy(self.sim_env.current_player),
                'state': copy.deepcopy(self.sim_env.board_state), 'returns': 0, 'visits': 0, 'wins': 0})
            return child_node, reward, terminated
        else:
            print('final node, no actions remaining')
            return None

    def simulate(self, start_node, reward, terminated):
        parent_node = start_node

        # The expanded node might already be a terminating node, i.e. no simulation required until end
        while not terminated:
            action = random.choice(self.sim_env.get_possible_actions())
            _, reward, terminated, _, _ = self.sim_env.step(action)

            # Optional add roll out child nodes to tree
            # child_node = self.search_tree.create_node(f'{action}', parent=parent_node.identifier, data={'current_player': copy.deepcopy(self.sim_env.current_player), 'state': copy.deepcopy(self.sim_env.board_state), 'returns': 0, 'visits': 0, 'wins': 0})
            # parent_node = child_node

        return reward

    def backup(self, current_node, reward):
        if reward != 0.5:
            # We need to flip the rewards, since the current_node is needed as a choice for the parent node: https://medium.com/@quasimik/monte-carlo-tree-search-applied-to-letterpress-34f41c86e238
            # E.g. when the current player of a node is player 2, but player 1 won, then the win for the current node is 1, else 0, because the current node represent the action taken by player 1
            if current_node.data['current_player'] == reward:
                adj_reward = 0
                wins = 0
            else:
                adj_reward = 1
                wins = 1
        else:  # Draw
            adj_reward = 0.5
            wins = 0

        current_node.data['returns'] += adj_reward
        current_node.data['visits'] += 1
        current_node.data['wins'] += wins

        if not current_node.is_root():
            identifier = current_node.predecessor(self.search_tree.identifier)
            current_node = self.search_tree.get_node(identifier)
            self.backup(current_node, reward)  # Assign the reward to recursively to parents until root is reached

    def select_best_action(self):
        # Get child node of root node with highest average return
        child_identifiers = self.root_node.successors(self.search_tree.identifier)
        max_return = -np.inf
        action = 0

        for child_identifier in child_identifiers:
            child_node = self.search_tree.get_node(child_identifier)
            avg_return = child_node.data['returns'] / child_node.data['visits']

            if avg_return > max_return:
                max_return = avg_return
                action = int(child_node.tag)

        return action