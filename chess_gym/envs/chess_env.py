import gymnasium as gym
from gymnasium import spaces

import chess
import chess.svg

import numpy as np
import pygame
from io import BytesIO
import cairosvg
from PIL import Image

class MoveSpace(spaces.Space):
    def __init__(self, board):
        super().__init__(dtype=np.int32)
        self.board = board
        self._shape = (6,)  # [from_square, to_square, promotion, drop, promotion_color, drop_color]

    @property
    def shape(self):
        return self._shape

    def sample(self):
        legal_moves = list(self.board.legal_moves)
        if not legal_moves:
            return np.zeros(self.shape, dtype=self.dtype)
        move = np.random.choice(legal_moves)
        return self._move_to_action(move)
    
    def contains(self, x):
        try:
            move = self._action_to_move(x)
            return move in self.board.legal_moves
        except:
            return False

    def _action_to_move(self, action):
        from_square = chess.Square(action[0])
        to_square = chess.Square(action[1])
        
        # Handle promotion
        if action[2] != 0:  # If there's a promotion
            promotion = chess.PieceType(action[2])
        else:
            promotion = None
            
        # Handle drop
        if action[3] != 0:  # If there's a drop
            drop_piece_type = chess.PieceType(action[3])
            drop_color = chess.WHITE if action[5] == 1 else chess.BLACK
            drop = chess.Piece(drop_piece_type, drop_color)
        else:
            drop = None
            
        move = chess.Move(from_square, to_square, promotion, drop)
        return move

    def _move_to_action(self, move):
        from_square = move.from_square
        to_square = move.to_square
        
        # Handle promotion
        if move.promotion is not None:
            promotion = move.promotion
            # For promotions, the color is determined by the current turn
            promotion_color = 1 if self.board.turn else 0
        else:
            promotion = 0
            promotion_color = 0
            
        # Handle drop
        if move.drop is not None:
            drop = move.drop
            # For drops, the color is determined by the current turn
            drop_color = 1 if self.board.turn else 0
        else:
            drop = 0
            drop_color = 0
            
        return np.array([from_square, to_square, promotion, drop, promotion_color, drop_color], dtype=self.dtype)

class ChessEnv(gym.Env):
    """Chess Environment"""
    metadata = {
        'render_modes': ['rgb_array', 'human'],
        'observation_modes': ['rgb_array', 'piece_map'],
        'render_fps': 4
    }

    def __init__(self, render_size=512, observation_mode='rgb_array', claim_draw=True, render_mode=None, **kwargs):
        super(ChessEnv, self).__init__()

        if observation_mode == 'rgb_array':
            self.observation_space = spaces.Box(
                low=0, 
                high=255,
                shape=(render_size, render_size, 3),
                dtype=np.uint8
            )
        elif observation_mode == 'piece_map':
            self.observation_space = spaces.Box(
                low=-6, 
                high=6,
                shape=(8, 8),
                dtype=np.int8
            )
        else:
            raise Exception("observation_mode must be either rgb_array or piece_map")

        self.observation_mode = observation_mode
        self.chess960 = kwargs.get('chess960', False)
        self.board = chess.Board(chess960=self.chess960)

        if self.chess960:
            self.board.set_chess960_pos(np.random.randint(0, 960))

        self.render_size = render_size
        self.claim_draw = claim_draw
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.action_space = MoveSpace(self.board)

    def _get_image(self):
        out = BytesIO()
        bytestring = chess.svg.board(self.board, size=self.render_size).encode('utf-8')
        cairosvg.svg2png(bytestring=bytestring, write_to=out)
        image = Image.open(out)
        return np.asarray(image)

    def _get_piece_configuration(self):
        piece_map = np.zeros((8, 8), dtype=np.int8)
        for square, piece in self.board.piece_map().items():
            rank = square // 8
            file = square % 8
            piece_map[rank, file] = piece.piece_type * (1 if piece.color else -1)
        return piece_map

    def _observe(self):
        if self.observation_mode == 'rgb_array':
            observation = self._get_image()
        else:  # piece_map
            observation = self._get_piece_configuration()
        return observation

    def step(self, action):
        # Convert the action array to a chess move
        move = self.action_space._action_to_move(action)
        self.board.push(move)

        observation = self._observe()
        result = self.board.result()
        reward = (1 if result == '1-0' else -1 if result == '0-1' else 0)
        terminated = self.board.is_game_over(claim_draw=self.claim_draw)
        truncated = False
        info = {
            'turn': self.board.turn,
            'castling_rights': self.board.castling_rights,
            'fullmove_number': self.board.fullmove_number,
            'halfmove_clock': self.board.halfmove_clock,
            'promoted': self.board.promoted,
            'chess960': self.board.chess960,
            'ep_square': self.board.ep_square
        }
        
        if self.render_mode == 'human':
            self.render()

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board.reset()

        if options is not None:
            if 'fen' in options:
                self.board.set_fen(options['fen'])
            if 'turn' in options:
                self.board.turn = options['turn']

        if self.chess960:
            self.board.set_chess960_pos(np.random.randint(0, 960))

        observation = self._observe()
        info = {}
        return observation, info

    def render(self):
        if self.render_mode == "human":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.render_size, self.render_size))
            pygame.display.set_caption('Chess Environment')
            
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        # Convert the SVG to a Pygame surface
        out = BytesIO()
        bytestring = chess.svg.board(self.board, size=self.render_size).encode('utf-8')
        cairosvg.svg2png(bytestring=bytestring, write_to=out)
        image = Image.open(out)
        image = image.convert('RGBA')
        image_data = image.tobytes()
        image_size = image.size
        canvas = pygame.image.fromstring(image_data, image_size, 'RGBA')

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
