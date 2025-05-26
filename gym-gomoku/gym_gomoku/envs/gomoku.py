import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
from six import StringIO
import sys, os
import six
import pygame

from gym_gomoku.envs.util import gomoku_util
from gym_gomoku.envs.util import make_random_policy
from gym_gomoku.envs.util import make_beginner_policy
from gym_gomoku.envs.util import make_medium_policy
from gym_gomoku.envs.util import make_expert_policy

# Rules from Wikipedia: Gomoku is an abstract strategy board game, Gobang or Five in a Row, it is traditionally played with Go pieces (black and white stones) on a go board with 19x19 or (15x15) 
# The winner is the first player to get an unbroken row of five stones horizontally, vertically, or diagonally. (so-calle five-in-a row)
# Black plays first if white did not win in the previous game, and players alternate in placing a stone of their color on an empty intersection.

class GomokuState(object):
    '''
    Similar to Go game, Gomoku state consists of a current player and a board.
    Actions are exposed as integers in [0, num_actions), which is to place stone on empty intersection
    '''
    def __init__(self, board, color):
        '''
        Args:
            board: current board
            color: color of current player
        '''
        assert color in ['black', 'white'], 'Invalid player color'
        self.board, self.color = board, color
    
    def act(self, action):
        '''
        Executes an action for the current player
        
        Returns:
            a new GomokuState with the new board and the player switched
        '''
        return GomokuState(self.board.play(action, self.color), gomoku_util.other_color(self.color))
    
    def __repr__(self):
        '''stream of board shape output'''
        # To Do: Output shape * * * o o
        return 'To play: {}\n{}'.format(six.u(self.color), self.board.__repr__())

# Sampling without replacement Wrapper 
# sample() method will only sample from valid spaces
class DiscreteWrapper(spaces.Discrete):
    def __init__(self, n):
        super().__init__(n)
        self.valid_spaces = list(range(n))
    
    def sample(self):
        '''Only sample from the remaining valid spaces
        '''
        if len(self.valid_spaces) == 0:
            print ("Space is empty")
            return None
        np_random, _ = seeding.np_random()
        randint = np_random.integers(0, len(self.valid_spaces))
        return self.valid_spaces[randint]
    
    def remove(self, s):
        '''Remove space s from the valid spaces
        '''
        if s is None:
            return
        if s in self.valid_spaces:
            self.valid_spaces.remove(s)
        else:
            print ("space %d is not in valid spaces" % s)
    
    def contains(self, x):
        """Return boolean specifying if x is a valid member of this space."""
        return x in self.valid_spaces
    
    def __repr__(self):
        """Gives a string representation of this space."""
        return f"DiscreteWrapper({self.n})"


### Environment
class GomokuEnv(gym.Env):
    '''
    GomokuEnv environment. Play against a fixed opponent.
    '''
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 30}
    
    def __init__(self, board_size):
        self.board_size = board_size
        shape = (self.board_size, self.board_size)
        self.observation_space = spaces.Box(low=-1, high=1, shape=shape, dtype=np.int8)
        self.action_space = DiscreteWrapper(self.board_size**2)
        self.moves = []
        self.state = None
        self.done = False
        self.reset()

        self._pygame_initialized = False
        self._pygame_screen = None
        self._pygame_images = {}
        self._pygame_board_size = 600  # pixels
        self._pygame_margin = 40       # margin for the board
        self._pygame_cell_size = (self._pygame_board_size - 2 * self._pygame_margin) // (self.board_size - 1)
    
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        board = Board(self.board_size)
        if options is not None:
            if 'fen' in options:
                board.set_fen(options['fen'])

        self.state = GomokuState(board, gomoku_util.BLACK)  # Always black to play
        self.moves = []
        self.action_space = DiscreteWrapper(self.board_size**2)
        self.done = self.state.board.is_terminal()
        obs = self.state.board.get_board_vector()
        return obs, {}
    
    def close(self):
        self.state = None
    
    def render(self, mode="human"):
        if mode != "human":
            outfile = StringIO() if mode == 'ansi' else sys.stdout
            outfile.write(repr(self.state) + '\n')
            return outfile

        # --- Pygame initialization ---
        if not self._pygame_initialized:
            pygame.init()
            self._pygame_board_size = 750  # match omok_pygame
            self._pygame_screen = pygame.display.set_mode((self._pygame_board_size, self._pygame_board_size))
            pygame.display.set_caption("Gomoku")
            # Load images from source/img
            base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'source', 'img')
            self._pygame_images['board'] = pygame.image.load(os.path.join(base_path, 'game_board.png'))
            self._pygame_images['black'] = pygame.image.load(os.path.join(base_path, 'wblack_stone.png'))
            self._pygame_images['white'] = pygame.image.load(os.path.join(base_path, 'white_stone.png'))
            self._pygame_initialized = True

        # Draw the board background
        self._pygame_screen.blit(
            pygame.transform.scale(self._pygame_images['board'], (self._pygame_board_size, self._pygame_board_size)),
            (0, 0)
        )

        # --- Omok pixel reference ---
        dis = 47
        board_size = self.board_size
        x0 = 625 - 18 - 250
        y0 = 375 - 19

        # Draw stones
        for a in range(board_size):
            for b in range(board_size):
                val = self.state.board.board_state[a][b]
                if val == 1:  # Black
                    stone_img = self._pygame_images['black']
                elif val == -1:  # White
                    stone_img = self._pygame_images['white']
                else:
                    continue
                x = x0 + (b - (board_size // 2)) * dis
                y = y0 + (a - (board_size // 2)) * dis
                stone_size = int(dis * 5 / 6)
                stone_img_scaled = pygame.transform.smoothscale(stone_img, (stone_size, stone_size))
                self._pygame_screen.blit(stone_img_scaled, (x, y))

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
    
    def step(self, action):
        assert not self.done, "Game is already over"
        prev_state = self.state
        self.state = self.state.act(action)
        self.moves.append(self.state.board.last_coord)
        self.action_space.remove(action)
        self.done = self.state.board.is_terminal()
        obs = self.state.board.get_board_vector()
        reward = 0.
        terminated = False
        truncated = False
        if self.done:
            exist, win_color = gomoku_util.check_five_in_row(self.state.board.board_state)
            if win_color == "empty":
                reward = 0.
            else:
                player_color = prev_state.color
                # player_color: 'black' or 'white'
                player_val = 1 if player_color == 'black' else -1
                # win_color: 'black' or 'white'
                win_val = 1 if win_color == 'black' else -1
                player_wins = (player_val == win_val)
                reward = 1. if player_wins else -1.
            terminated = True
        return obs, reward, terminated, truncated, {'state': self.state}

    @property
    def board(self):
        return self.state.board

class Board(object):
    '''
    Basic Implementation of a Go Board, natural action are int [0,board_size**2)
    '''
    
    def __init__(self, board_size, fen=None):
        self.size = board_size
        self.board_state = [[0] * board_size for _ in range(board_size)] # 0 for empty
        self.move = 0                 # how many move has been made
        self.last_coord = (-1,-1)     # last action coord
        self.last_action = None       # last action made
        self.foul = False             # flag for illegal move
        if fen is not None:
            self.set_fen(fen)
    
    def is_game_over(self, claim_draw=False):
        if self.foul:
            return True
        if claim_draw:
            return all(self.board_state[i][j] != 0 for i in range(self.size) for j in range(self.size))
        return self.is_terminal()
    
    def coord_to_action(self, i, j):
        ''' convert coordinate i, j to action a in [0, board_size**2)
        '''
        a = i * self.size + j # action index
        return a
    
    def action_to_coord(self, a):
        coord = (a // self.size, a % self.size)
        return coord
    
    def copy(self, *args, **kwargs):
        '''Create a new Board instance with the same state as the current board'''
        new_board = Board(self.size)
        new_board.board_state = [row[:] for row in self.board_state]  # Deep copy of board state
        new_board.move = self.move
        new_board.last_coord = self.last_coord
        new_board.last_action = self.last_action
        return new_board
    
    def play(self, action, color):
        self.copy(self.board_state) # create a board copy of current board_state
        self.move = self.move
        coord = self.action_to_coord(action)
        # check if it's legal move
        if (self.board_state[coord[0]][coord[1]] != 0): # the action coordinate is not empty
            self.foul = True
            self.last_coord = coord
            self.last_action = action
            return self
        # Place stone: 1 for black, -1 for white
        if color == 'black':
            self.board_state[coord[0]][coord[1]] = 1
        else:
            self.board_state[coord[0]][coord[1]] = -1
        self.move += 1 # move counter add 1
        self.last_coord = coord # save last coordinate
        self.last_action = action
        return self
    
    def is_terminal(self):
        if self.foul:
            return True
        # First check if the board is full
        is_full = all(self.board_state[i][j] != 0 for i in range(self.size) for j in range(self.size))
        if is_full:
            return True
        # Then check for five in a row
        exist, _ = gomoku_util.check_five_in_row(self.board_state)
        return exist
    
    def __repr__(self):
        ''' representation of the board class
            print out board_state
        '''
        out = ""
        size = len(self.board_state)
        
        letters = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')[:size]
        numbers = list(range(1, 100))[:size]
        
        label_move = "Move: " + str(self.move) + "\n"
        label_letters = "     " + " ".join(letters) + "\n"
        label_boundry = "   " + "+-" + "".join(["-"] * (2 * size)) + "+" + "\n"
        
        # construct the board output
        out += (label_move + label_letters + label_boundry)
        
        for i in range(size-1,-1,-1):
            line = ""
            line += (str("%2d" % (i+1)) + " |" + " ")
            for j in range(size):
                # check if it's the last move
                if self.board_state[i][j] == 1:
                    line += 'X'
                elif self.board_state[i][j] == -1:
                    line += 'O'
                else:
                    line += '.'
                if (i,j) == self.last_coord:
                    line += ")"
                else:
                    line += " "
            line += ("|" + "\n")
            out += line
        out += (label_boundry + label_letters)
        return out
    
    def encode(self):
        '''Return: np array
            np.array(board_size, board_size): state observation of the board
        '''
        img = np.array(self.board_state, dtype=np.int8)
        return img

    def get_board_vector(self):
        '''Return a tensor representation of the board state.
        Returns:
            np.ndarray: A 2x15x15 tensor where:
                - Channel 0: Stone positions (1 for black, -1 for white, 0 for empty)
                - Channel 1: Current player color (1 for black, -1 for white)
        '''
        # Create a 2x15x15 tensor
        board_tensor = np.zeros((2, self.size, self.size), dtype=np.float32)
        
        # Channel 0: Stone positions
        board_tensor[0] = np.array(self.board_state, dtype=np.float32)
        
        # Channel 1: Current player color (1 for black, -1 for white)
        # This is determined by the number of moves (even = black, odd = white)
        current_player = 1 if self.move % 2 == 0 else -1
        board_tensor[1].fill(current_player)
        
        return board_tensor
    
    @property
    def legal_moves(self):
        ''' Get all the next legal move, namely empty space that you can place your 'color' stone
            Return: Coordinate of all the empty space, [(x1, y1), (x2, y2), ...]
        '''
        legal_move = []
        for i in range(self.size):
            for j in range(self.size):
                if (self.board_state[i][j] == 0):
                    legal_move.append((i, j))
        return legal_move

    @property
    def legal_actions(self):
        '''Get all legal action IDs for the current board state.
        Returns:
            List[int]: List of valid action IDs for empty positions
        '''
        legal_action = []
        for i in range(self.size):
            for j in range(self.size):
                if (self.board_state[i][j] == 0):
                    legal_action.append(self.coord_to_action(i, j))
        return sorted(legal_action)

    @property
    def turn(self):
        """
        Return the current player's color as a string: 'black' or 'white'.
        """
        return 'black' if self.move % 2 == 0 else 'white'

    def get_square_to_ids_map(self):
        '''Get a mapping of board positions to their corresponding action IDs.
        Returns:
            Dict[Tuple[int, int], List[int]]: Dictionary mapping board coordinates to lists of action IDs
        '''
        square_to_ids = {}
        for i in range(self.size):
            for j in range(self.size):
                if (self.board_state[i][j] == 0):
                    action_id = self.coord_to_action(i, j)
                    square_to_ids[(i, j)] = [action_id]  # In Gomoku, each square maps to exactly one action ID
        return square_to_ids
    
    def move_to_action_id(self, move):
        '''Convert a move (i, j) to an action id.'''
        return self.coord_to_action(move[0], move[1])
    
    def action_id_to_move(self, action_id):
        '''Convert an action id to a move (i, j).'''
        return self.action_to_coord(action_id)
    
    def fen(self):
        """
        Serialize the board state to a FEN-like string for Gomoku.
        Format: <flat_board>/<move>/<last_i>,<last_j>
        Example: '000...0/0/-1,-1'
        """
        flat_board = ''.join(str(self.board_state[i][j]) for i in range(self.size) for j in range(self.size))
        last_i, last_j = self.last_coord
        return f"{flat_board}/{self.move}/{last_i},{last_j}"

    def set_fen(self, fen_str):
        """
        Restore the board state from a FEN-like string for Gomoku.
        """
        try:
            flat_board, move_str, last_coord_str = fen_str.split('/')
            self.move = int(move_str)
            last_i, last_j = map(int, last_coord_str.split(','))
            self.last_coord = (last_i, last_j)
            # Restore board_state
            k = 0
            for i in range(self.size):
                for j in range(self.size):
                    self.board_state[i][j] = int(flat_board[k])
                    k += 1
        except Exception as e:
            raise ValueError(f"Invalid Gomoku FEN string: {fen_str}") from e

    def push(self, move):
        """
        Apply a move to the board state.
        Args:
            move (Tuple[int, int]): The move to apply
        """
        return self.play(self.move_to_action_id(move), color='black' if self.move % 2 == 0 else 'white')
