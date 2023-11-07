import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from moviepy.video.io.bindings import mplfig_to_npimage

class ConnectFour(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 10}

    def __init__(self, render_mode):
        self.rows = 6
        self.cols = 7
        self.board_state = np.zeros(shape=(self.rows, self.cols), dtype='int')
        self.current_player = 1

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def get_possible_actions(self):
        empty_fields = np.argwhere(self.board_state == 0)
        possible_actions = list(set(empty_fields[:, 1]))  # List with unique column indices of empty fields

        return possible_actions

    def evaluate_state(self):
        board = self.board_state

        # Check for horizontal wins
        for row in board:
            for col in range(4):
                if row[col] == row[col + 1] == row[col + 2] == row[col + 3] and row[col] != 0:
                    return True, row[col]

        # Check for vertical wins
        for col in range(7):
            for row in range(3):
                if board[row][col] == board[row + 1][col] == board[row + 2][col] == board[row + 3][col] and board[row][
                    col] != 0:
                    return True, board[row][col]

        # Check for diagonal wins (top-left to bottom-right)
        for row in range(3):
            for col in range(4):
                if board[row][col] == board[row + 1][col + 1] == board[row + 2][col + 2] == board[row + 3][col + 3] and \
                        board[row][col] != 0:
                    return True, board[row][col]

        # Check for diagonal wins (top-right to bottom-left)
        for row in range(3):
            for col in range(3, 7):
                if board[row][col] == board[row + 1][col - 1] == board[row + 2][col - 2] == board[row + 3][col - 3] and \
                        board[row][col] != 0:
                    return True, board[row][col]

        # Check for a draw condition -> Return 0.5
        if np.all(board != 0):
            return True, 0.5

        # If no winning condition is met, the game is not over, and the winning player is None
        return False, None

    def step(self, action):
        self.action = action
        self.action_row = np.max(
            np.argwhere(self.board_state[:, self.action] == 0))  # Index of last empty row of action column
        self.board_state[self.action_row, self.action] = self.current_player
        observation = self.board_state
        terminated, reward = self.evaluate_state()
        info = {}

        if self.current_player == 1:
            self.current_player = 2
        else:
            self.current_player = 1

        return observation, reward, terminated, False, info

    def pop(self):
        self.board_state[self.action_row, self.action] = 0

        if self.current_player == 1:
            self.current_player = 2
        else:
            self.current_player = 1

    def reset(self, seed=None, options=None):
        self.__init__(self.render_mode)

        observation = self.board_state
        info = {}

        return observation, info

    def render(self):
        fig, ax = plt.subplots()
        ax.imshow(self.board_state)
        ax.set_title("Connect Four Board State")
        img = mplfig_to_npimage(fig)
        plt.close()

        return img