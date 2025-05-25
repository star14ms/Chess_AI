import gymnasium as gym
from gymnasium import spaces

import chess
import chess.svg

import numpy as np
from io import BytesIO
import cairosvg
from PIL import Image
from typing import Union, Optional, List
import os

from gymnasium.utils.save_video import save_video

from chess_gym.chess_custom import LegacyChessBoard, FullyTrackedBoard
from utils.visualize import draw_possible_actions_on_board

class MoveSpace(spaces.Space):
    def __init__(self, board: FullyTrackedBoard | LegacyChessBoard, action_space_mode: str = "1700"):
        super().__init__(dtype=np.int32)
        self.board = board
        self.action_space_mode = action_space_mode
        self._shape = (6,)  # [from_square, to_square, promotion, drop, promotion_color, drop_color]

    @property
    def shape(self):
        return self._shape

    def sample(self, return_id: bool = False) -> Union[np.ndarray, int]:
        legal_moves = list(self.board.legal_moves)
        if not legal_moves:
            return np.zeros(self.shape, dtype=self.dtype)
        move = np.random.choice(legal_moves)
        return self._move_to_action(move, return_id=return_id)
    
    def contains(self, action: Union[int, np.ndarray, list]) -> bool:
        try:
            move = self._action_to_move(action)
            return move in self.board.legal_moves
        except:
            return False

    def _action_to_move(self, action: Union[int, np.ndarray, list]) -> chess.Move:
        # Check if action is an integer ID
        if isinstance(action, (int, np.integer)):
            if self.action_space_mode == "4672":
                # Convert from 4672 action space to chess move
                square = (action - 1) // 73
                relative_action = (action - 1) % 73 + 1  # Convert to 1-based
                
                # Handle special cases
                if relative_action >= 70:  # Castling
                    if relative_action == 70:  # Kingside
                        return chess.Move.from_uci('e1g1' if self.board.turn == chess.WHITE else 'e8g8')
                    else:  # Queenside
                        return chess.Move.from_uci('e1c1' if self.board.turn == chess.WHITE else 'e8c8')
                elif relative_action >= 64:  # Promotion
                    promo_pieces = ['q', 'n', 'b', 'r']
                    promo_idx = relative_action - 64
                    promo_char = promo_pieces[promo_idx]
                    # Get the source square and calculate target square
                    start_sq = chess.square_name(square)
                    # For promotions, we need to determine the target square
                    rank = chess.square_rank(square)
                    file = chess.square_file(square)
                    if self.board.turn == chess.WHITE:
                        end_sq = chess.square_name(chess.square(file, 7))
                    else:
                        end_sq = chess.square_name(chess.square(file, 0))
                    return chess.Move.from_uci(f"{start_sq}{end_sq}{promo_char}")
                else:  # Regular move
                    # Calculate direction (0-7) and distance (1-7)
                    if 1 <= relative_action <= 56:  # Queen-like moves
                        direction = (relative_action - 1) // 7
                        distance = (relative_action - 1) % 7 + 1
                        
                        # Calculate file and rank changes based on direction
                        file_change = 0
                        rank_change = 0
                        if direction == 0:  # N
                            rank_change = distance
                        elif direction == 1:  # NE
                            file_change = distance
                            rank_change = distance
                        elif direction == 2:  # E
                            file_change = distance
                        elif direction == 3:  # SE
                            file_change = distance
                            rank_change = -distance
                        elif direction == 4:  # S
                            rank_change = -distance
                        elif direction == 5:  # SW
                            file_change = -distance
                            rank_change = -distance
                        elif direction == 6:  # W
                            file_change = -distance
                        elif direction == 7:  # NW
                            file_change = -distance
                            rank_change = distance
                    elif 57 <= relative_action <= 64:  # Knight moves
                        knight_patterns = {
                            1: (2, 1), 2: (1, 2), 3: (-1, 2), 4: (-2, 1),
                            5: (-2, -1), 6: (-1, -2), 7: (1, -2), 8: (2, -1)
                        }
                        file_change, rank_change = knight_patterns[relative_action - 56]
                    else:
                        return chess.Move.null()
                    
                    start_file = chess.square_file(square)
                    start_rank = chess.square_rank(square)
                    end_file = start_file + file_change
                    end_rank = start_rank + rank_change
                    
                    if 0 <= end_file < 8 and 0 <= end_rank < 8:
                        start_sq = chess.square_name(square)
                        end_sq = chess.square_name(chess.square(end_file, end_rank))
                        return chess.Move.from_uci(f"{start_sq}{end_sq}")
                    else:
                        # Invalid move - return a no-op move
                        return chess.Move.null()
            else:
                if not self.board.is_theoretically_possible_state:
                    raise ValueError("Cannot convert action ID to move for a theoretically impossible board state.")
                return self.board.action_id_to_move(action)
        
        # Check if action is a list or numpy array (legacy format)
        elif isinstance(action, (list, np.ndarray)):
            # Ensure legacy format has the correct shape if it's a numpy array
            if isinstance(action, np.ndarray) and action.shape != self._shape:
                 raise ValueError(f"Invalid action shape for legacy format: expected {self._shape}, got {action.shape}")
            # Ensure legacy format has the correct length if it's a list
            if isinstance(action, list) and len(action) != self._shape[0]:
                 raise ValueError(f"Invalid action length for legacy format: expected {self._shape[0]}, got {len(action)}")
            
            # Convert from legacy 6-element array
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
                
            return chess.Move(from_square, to_square, promotion, drop)
        else:
            # Raise error for unsupported action type
            raise TypeError(f"Unsupported action type: {type(action)}")

    def _move_to_action(self, move: chess.Move, return_id: bool = False) -> Union[np.ndarray, int]:
        if return_id:
            if self.action_space_mode == "4672":
                # Convert move to 4672 action space
                start_sq = move.from_square
                file_change = chess.square_file(move.to_square) - chess.square_file(start_sq)
                rank_change = chess.square_rank(move.to_square) - chess.square_rank(start_sq)
                
                # Handle special cases
                if move.promotion is not None:
                    # Promotion moves (65-68)
                    promo_pieces = ['q', 'n', 'b', 'r']
                    promo_idx = promo_pieces.index(move.promotion.symbol().lower())
                    return start_sq * 73 + 65 + promo_idx
                elif abs(file_change) == 2 and rank_change == 0 and move.from_square in [chess.E1, chess.E8]:
                    # Castling (69-70)
                    if file_change == 2:  # Kingside
                        return start_sq * 73 + 69
                    else:  # Queenside
                        return start_sq * 73 + 70
                else:
                    # Check if it's a knight move
                    is_knight_move = (abs(file_change) == 2 and abs(rank_change) == 1) or (abs(file_change) == 1 and abs(rank_change) == 2)
                    
                    if is_knight_move:
                        # Knight moves are encoded as 57-64
                        knight_patterns = {
                            (2, 1): 1, (1, 2): 2, (-1, 2): 3, (-2, 1): 4,
                            (-2, -1): 5, (-1, -2): 6, (1, -2): 7, (2, -1): 8
                        }
                        move_index = 56 + knight_patterns.get((file_change, rank_change), 0)
                    else:
                        # Queen-like moves are encoded as 1-56
                        # Determine direction (0-7)
                        if rank_change > 0:  # North
                            if file_change > 0: direction = 1  # NE
                            elif file_change < 0: direction = 7  # NW
                            else: direction = 0  # N
                        elif rank_change < 0:  # South
                            if file_change > 0: direction = 3  # SE
                            elif file_change < 0: direction = 5  # SW
                            else: direction = 4  # S
                        else:  # East or West
                            if file_change > 0: direction = 2  # E
                            else: direction = 6  # W
                            
                        # Calculate distance (1-7)
                        distance = max(abs(file_change), abs(rank_change))
                        if distance > 7:
                            raise ValueError(f"Invalid move distance: {distance}")
                            
                        # direction * 7 + distance
                        move_index = direction * 7 + distance
                    
                    if 1 <= move_index <= 73:  # Total of 73 move types per square
                        return start_sq * 73 + move_index
                    else:
                        raise ValueError(f"Invalid move index: {move_index}")
            else:
                if not self.board.is_theoretically_possible_state:
                    raise ValueError("Cannot return action ID for a theoretically impossible board state.")
                return self.board.move_to_action_id(move)
        
        # Otherwise, return the legacy 6-element array
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
            # Note: python-chess Move objects don't store the *color* of the dropped piece directly,
            # only its type. The color is inferred from the turn. We need the PieceType.
            drop = move.drop # This gives the PieceType for drops
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
        'observation_modes': ['rgb_array', 'piece_map', 'vector'],
        'render_fps': 10
    }

    def __init__(self, 
                 render_size=512, 
                 observation_mode='vector', 
                 claim_draw=True, 
                 render_mode=None, 
                 show_possible_actions=False, 
                 save_video_folder: Optional[str] = None,
                 action_space_mode: str = "1700",
                 **kwargs):
        super(ChessEnv, self).__init__()

        # --- Video Saving Setup ---
        self.save_video_folder = save_video_folder
        self.recorded_frames: List[np.ndarray] = []
        self.video_episode_index = 0
        self.video_step_counter = 0 # Total steps across episodes
        self.current_episode_step_counter = 0 # Steps within the current episode

        if self.save_video_folder is not None:
            os.makedirs(self.save_video_folder, exist_ok=True)
            print(f"Video recording enabled. Saving to: {os.path.abspath(self.save_video_folder)}")
        # --- End Video Saving Setup ---

        self.observation_mode = observation_mode
        self.render_mode = render_mode
        self.action_space_mode = action_space_mode

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
        elif observation_mode == 'vector':
            self.observation_space = spaces.Box(
                low=-1,
                high=1,
                shape=(26, 8, 8),
                dtype=np.int8
            )
        else:
            raise ValueError(f"Invalid observation_mode: {observation_mode}. Must be one of {self.metadata['observation_modes']}")

        self.chess960 = kwargs.get('chess960', False)
        # --- Board selection logic moved here ---
        if self.action_space_mode == "4672":
            self.board = LegacyChessBoard(chess960=self.chess960)
        else:
            self.board = FullyTrackedBoard(chess960=self.chess960)

        if self.chess960:
            self.board.set_chess960_pos(np.random.randint(0, 960))

        self.render_size = render_size
        self.claim_draw = claim_draw
        self.show_possible_actions = show_possible_actions
        self.window = None
        self.clock = None
        
        # Set action space based on the action_space_mode
        self.action_space = MoveSpace(self.board, action_space_mode=self.action_space_mode)

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
        elif self.observation_mode == 'piece_map':
            observation = self._get_piece_configuration()
        elif self.observation_mode == 'vector':
            if hasattr(self.board, 'get_board_vector'):
                observation = self.board.get_board_vector()
            else:
                raise AttributeError("Board object does not have method 'get_board_vector'. Required for observation_mode='vector'.")
        else:
            raise RuntimeError(f"Invalid internal state: observation_mode='{self.observation_mode}'")
        return observation

    def step(self, action):
        # Convert the action to a chess move using MoveSpace
        move = self.action_space._action_to_move(action)

        # Push the move regardless of legality
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
        
        # --- Record Frame --- 
        if self.render_mode is not None or self.save_video_folder is not None:
            frame = self.render() # Get frame via rgb_array mode

            if self.save_video_folder is not None and frame is not None: # Ensure render returned something
                self.recorded_frames.append(frame)
        # --- End Record Frame --- 

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None, save_video=False):
        super().reset(seed=seed)
        
        # --- Save video from previous episode --- 
        if save_video and self.save_video_folder is not None and self.recorded_frames:
            self._save_recorded_video()
            # Don't increment episode index here, do it after saving
        # --- End Save Video --- 
        
        # Reset board and internal state
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
        if self.render_mode is not None:
            return self._render_pygame()
        else:
            return self._get_image()

    def _render_pygame(self):
        import pygame
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.render_size, self.render_size))
            pygame.display.set_caption('Chess Environment')
        if self.clock is None:
            self.clock = pygame.time.Clock()

        # --- Logic to show possible actions (if enabled) ---
        if self.show_possible_actions:
            # --- 1. Display standard board FIRST ---
            std_board_image_pil = Image.fromarray(self._get_image())
            std_board_image_pygame = pygame.image.fromstring(
                std_board_image_pil.tobytes(), std_board_image_pil.size, std_board_image_pil.mode
            ).convert()
            self.window.blit(std_board_image_pygame, (0, 0))
            pygame.display.update()
            pygame.time.wait(300) # <-- Pause 1
            # --- End step 1 ---

            # --- 2. Display action IDs SECOND ---
            vis_image_pil = draw_possible_actions_on_board(self.board, size=self.render_size, return_pil_image=True)
            if vis_image_pil is not None:
                vis_image_pygame = pygame.image.fromstring(
                    vis_image_pil.tobytes(), vis_image_pil.size, vis_image_pil.mode
                ).convert()
                self.window.blit(vis_image_pygame, (0, 0))
                pygame.display.update()
                pygame.time.wait(1000) # <-- Pause 2
                # --- End step 2 ---

                # --- 3. Display standard board THIRD (remove IDs) ---
                # Reuse std_board_image_pygame from step 1
                self.window.blit(std_board_image_pygame, (0, 0))
                pygame.display.update()
                pygame.time.wait(300) # <-- Pause 3
                # --- End step 3 ---
            else:
                # Handle case where action IDs couldn't be generated
                print("Board state incompatible with action space, skipping action visualization.")
                # If action IDs failed, the standard board was shown (step 1) and paused.
                # We just proceed to the final render after the initial pause.
                pass 
            # --- End Action ID Display Logic ---

        # --- Original rendering logic (display the final current board state) ---
        # This ensures the *final* state is shown before the game potentially steps again,
        # respecting the render_fps.
        canvas = pygame.Surface((self.render_size, self.render_size))
        canvas.fill((255, 255, 255))
        final_board_image_pil = Image.fromarray(self._get_image())
        final_board_image_pygame = pygame.image.fromstring(
            final_board_image_pil.tobytes(), final_board_image_pil.size, final_board_image_pil.mode
        ).convert()
        self.window.blit(final_board_image_pygame, (0, 0))
        pygame.display.update()

        # Frame rate control
        if self.render_mode == "human":
            self.clock.tick(self.metadata["render_fps"])
            
        return self._get_image()

    def close(self):
        # --- Save video from the final episode --- 
        if self.save_video_folder is not None and self.recorded_frames:
            print("Closing environment and saving final video...")
            self._save_recorded_video()
        # --- End Save Video --- 

        if self.window is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
    
    def _get_board_vector(self):
        return self.board.get_board_vector()

    # --- Video Saving Helper ---
    def _save_recorded_video(self):
        """Internal helper to save the recorded frames to a video file."""
        if not self.save_video_folder or not self.recorded_frames:
            return # Nothing to save or saving not enabled

        try:
            # Calculate the starting step index for the episode just finished
            # step_starting_index = self.video_step_counter - self.current_episode_step_counter
            # The Gymnasium example uses 0 for step_starting_index when saving full episodes
            # Let's stick to that for simplicity, as step_trigger isn't used here.
            step_starting_index = 0 

            print(f"Saving video for episode {self.video_episode_index} ({len(self.recorded_frames)} frames)...")
            save_video(
                frames=self.recorded_frames,
                video_folder=self.save_video_folder,
                # --- Important: Pass fps --- 
                fps=self.metadata.get('render_fps', 10), # Use fps from metadata
                name_prefix="chess-game", # Customize prefix if desired
                episode_index=self.video_episode_index,
                episode_trigger=lambda x: True, # Trigger always true since called conditionally
                step_starting_index=step_starting_index, 
                video_length=None, # Record full episode
                save_logger="bar" # Show progress bar
            )
            # Increment episode index *after* successful save
            self.video_episode_index += 1
            # Clear frames immediately after saving (also done in reset, but safe here too)
            self.recorded_frames = [] 

        except ImportError:
             print("Error: moviepy is required for save_video. Install with: pip install moviepy")
        except Exception as e:
            print(f"Error saving video for episode {self.video_episode_index}: {e}")
            # Optionally clear frames even if save failed to prevent repeated errors
            self.recorded_frames = [] 
    # --- End Video Saving Helper ---