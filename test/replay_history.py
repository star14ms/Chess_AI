# Pygame replay viewer using python-chess visualization (SVG)
# If you don't have these installed:
# !pip install pygame cairosvg

import re
import io
import pygame
import chess
import chess.svg
from typing import List, Optional, Tuple, Dict

try:
    import cairosvg  # SVG -> PNG
except Exception:
    cairosvg = None
    print("cairosvg not available. Install with: pip install cairosvg")
    

def parse_game_file(file_path: str) -> List[Dict]:
    """
    Parse a text file containing chess games in the format:
    Game X: result (reason, move_count moves)
    Initial FEN: ... (optional)
    Initial Quality: ... or White's perspective: ... (optional, can be combined with FEN)
    Reward: ... (optional)
    move1 move2 move3 ...
    
    Returns a list of dictionaries, each containing:
    - game_number: int
    - result: str (e.g., "1/2-1/2")
    - reason: str (e.g., "MAX_MOVES")
    - move_count: int
    - initial_fen: Optional[str]
    - initial_quality: Optional[str]
    - reward: Optional[float]
    - moves: List[str] (SAN notation moves)
    """
    games = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Pattern to match game headers: "Game X: result (reason, move_count moves)"
        game_match = re.match(r'Game (\d+):\s*([^\s]+)\s*\(([^,]+),\s*(\d+)\s*moves\)', line)
        if game_match:
            game_number = int(game_match.group(1))
            result = game_match.group(2)
            reason = game_match.group(3)
            move_count = int(game_match.group(4))
            
            # Initialize optional fields
            initial_fen = None
            initial_quality = None
            reward = None
            
            # Parse optional metadata lines
            i += 1
            while i < len(lines):
                meta_line = lines[i].strip()
                
                # Check for empty line (separator before moves)
                if not meta_line:
                    i += 1
                    break
                
                # Check for Initial FEN (can be combined with Quality)
                if meta_line.startswith('Initial FEN:'):
                    # Handle combined format: "Initial FEN: ... | White's perspective: ..."
                    if ' | ' in meta_line:
                        parts = meta_line.split(' | ')
                        fen_part = parts[0]
                        quality_part = parts[1] if len(parts) > 1 else None
                        
                        # Extract FEN
                        fen_match = re.match(r'Initial FEN:\s*(.+)', fen_part)
                        if fen_match:
                            initial_fen = fen_match.group(1).strip()
                        
                        # Extract Quality
                        if quality_part:
                            quality_match = re.match(r'(?:Initial Quality|White\'s perspective):\s*(.+)', quality_part)
                            if quality_match:
                                initial_quality = quality_match.group(1).strip()
                    else:
                        # Just FEN
                        fen_match = re.match(r'Initial FEN:\s*(.+)', meta_line)
                        if fen_match:
                            initial_fen = fen_match.group(1).strip()
                
                # Check for Initial Quality (standalone)
                elif meta_line.startswith('Initial Quality:'):
                    quality_match = re.match(r'Initial Quality:\s*(.+)', meta_line)
                    if quality_match:
                        initial_quality = quality_match.group(1).strip()
                
                # Check for White's perspective (standalone)
                elif meta_line.startswith("White's perspective:"):
                    quality_match = re.match(r'White\'s perspective:\s*(.+)', meta_line)
                    if quality_match:
                        initial_quality = quality_match.group(1).strip()
                
                # Check for Reward
                elif meta_line.startswith('Reward:'):
                    reward_match = re.match(r'Reward:\s*([\d\.\-]+)', meta_line)
                    if reward_match:
                        try:
                            reward = float(reward_match.group(1))
                        except ValueError:
                            pass
                else:
                    # Line doesn't match any metadata pattern - it's likely the moves line
                    # Don't increment i here, break to start parsing moves from this line
                    break
                
                i += 1
            
            # Parse moves (everything until next game or end of file)
            moves = []
            while i < len(lines):
                move_line = lines[i].strip()
                
                # Check if this is the start of a new game
                if re.match(r'Game \d+:', move_line):
                    break
                
                # Skip empty lines
                if not move_line:
                    i += 1
                    continue
                
                # Parse moves (split by whitespace)
                moves.extend([m for m in move_line.split() if m])
                i += 1
            
            games.append({
                'game_number': game_number,
                'result': result,
                'reason': reason,
                'move_count': move_count,
                'initial_fen': initial_fen,
                'initial_quality': initial_quality,
                'reward': reward,
                'moves': moves
            })
        else:
            i += 1
    
    return games


def _boards_from_san(moves_san: List[str], start_fen: Optional[str] = None) -> Tuple[List[chess.Board], List[Optional[str]]]:
    """Return boards for ply index 0..N, and SAN labels (None for initial)."""
    try:
        board = chess.Board(start_fen) if start_fen else chess.Board()
    except Exception as e:
        print(f"Warning: Could not parse FEN '{start_fen}', using default starting position: {e}")
        board = chess.Board()
    
    boards = [board.copy(stack=False)]
    sans: List[Optional[str]] = [None]

    for san in moves_san:
        try:
            move = board.parse_san(san)
            board.push(move)
            boards.append(board.copy(stack=False))
            sans.append(san)
        except Exception as e:
            # Try alternative parsing: remove check/mate symbols and try again
            # Sometimes python-chess's parse_san() is strict about check symbols
            san_clean = san.rstrip('+#')
            if san_clean != san:
                try:
                    move = board.parse_san(san_clean)
                    board.push(move)
                    boards.append(board.copy(stack=False))
                    sans.append(san)  # Keep original notation for display
                except Exception as e2:
                    print(f"Warning: Could not parse move '{san}': {e} (also failed with cleaned version '{san_clean}': {e2})")
                    break
            else:
                print(f"Warning: Could not parse move '{san}': {e}")
                break

    return boards, sans


def _board_surface_from_chess_svg(board: chess.Board, size_px: int, flipped: bool) -> pygame.Surface:
    """Render board with python-chess.svg, convert to pygame Surface."""
    if cairosvg is None:
        raise RuntimeError("cairosvg is required to render chess.svg into pygame")

    svg_str = chess.svg.board(board=board, size=size_px, flipped=flipped)
    png_bytes = cairosvg.svg2png(bytestring=svg_str.encode("utf-8"))

    # pygame.image.load accepts a file-like object
    return pygame.image.load(io.BytesIO(png_bytes)).convert_alpha()


def replay_game_pygame(
    game_data: dict,
    board_px: int = 768,
    fps: int = 60,
    white_perspective: bool = True,
):
    """Open a Pygame window. Use ←/→ to move back/forward, Esc to quit."""
    moves = list(game_data.get("moves", []))
    initial_fen = game_data.get("initial_fen")
    boards, sans = _boards_from_san(moves, start_fen=initial_fen)

    pygame_initialized = False
    screen = None

    try:
        pygame.init()
        pygame_initialized = True
        pygame.display.set_caption("Chess replay")

        # Adjust margin_top based on whether we have metadata
        has_metadata = game_data.get('initial_quality') or game_data.get('reward') is not None
        margin_top = 96 if has_metadata else 72
        margin = 16
        width = board_px + margin * 2
        height = board_px + margin_top + margin

        screen = pygame.display.set_mode((width, height))
        clock = pygame.time.Clock()

        ui_font = pygame.font.SysFont(None, 24)
        ui_bg = (20, 20, 20)
        ui_fg = (230, 230, 230)

        idx = 0
        running = True

        # Simple cache so we don’t rerender SVG every frame
        cached_idx = None
        cached_surf = None

        def draw(board: chess.Board, san: Optional[str], ply: int):
            nonlocal cached_idx, cached_surf

            # UI bar
            screen.fill(ui_bg)
            title = f"Game {game_data.get('game_number', '?')}: {game_data.get('result', '?')} ({game_data.get('reason', '?')})"
            line1 = ui_font.render(title, True, ui_fg)
            screen.blit(line1, (margin, 10))

            # Show additional metadata if available
            metadata_parts = []
            if game_data.get('initial_quality'):
                metadata_parts.append(f"Quality: {game_data['initial_quality']}")
            if game_data.get('reward') is not None:
                metadata_parts.append(f"Reward: {game_data['reward']:.4f}")
            if metadata_parts:
                metadata_text = " | ".join(metadata_parts)
                line_meta = ui_font.render(metadata_text, True, ui_fg)
                screen.blit(line_meta, (margin, 38))
                y_offset = 66
            else:
                y_offset = 38

            if ply == 0:
                status = "Ply 0 / Start position"
            else:
                status = f"Ply {ply} / {len(boards)-1}   SAN: {san}"
            line2 = ui_font.render(status, True, ui_fg)
            screen.blit(line2, (margin, y_offset))

            if cached_idx != ply:
                cached_surf = _board_surface_from_chess_svg(
                    board,
                    size_px=board_px,
                    flipped=(not white_perspective),
                )
                cached_idx = ply

            screen.blit(cached_surf, (margin, margin_top))
            pygame.display.flip()

        # Enable key repeat for better responsiveness
        pygame.key.set_repeat(200, 50)  # 200ms delay, 50ms interval
        
        while running:
            clock.tick(fps)
            
            # Track if idx changed to force redraw
            idx_changed = False
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    old_idx = idx
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_RIGHT:
                        idx = min(idx + 1, len(boards) - 1)
                        idx_changed = True
                    elif event.key == pygame.K_LEFT:
                        idx = max(idx - 1, 0)
                        idx_changed = True
                    elif event.key == pygame.K_HOME:
                        idx = 0
                        idx_changed = True
                    elif event.key == pygame.K_END:
                        idx = len(boards) - 1
                        idx_changed = True
                    # Alternative keys for macOS compatibility
                    elif event.key == pygame.K_d:
                        idx = min(idx + 1, len(boards) - 1)
                        idx_changed = True
                    elif event.key == pygame.K_a:
                        idx = max(idx - 1, 0)
                        idx_changed = True
                    elif event.key == pygame.K_SPACE:
                        idx = min(idx + 1, len(boards) - 1)
                        idx_changed = True
                    elif event.key == pygame.K_BACKSPACE:
                        idx = max(idx - 1, 0)
                        idx_changed = True
            
            # Always draw, but force cache invalidation if idx changed
            if idx_changed:
                cached_idx = None  # Force cache invalidation
            
            draw(boards[idx], sans[idx], idx)

    except KeyboardInterrupt:
        # Handle cell interruption in Jupyter
        pass
    except Exception as e:
        print(f"Error in pygame replay: {e}")
    finally:
        # Always cleanup, even if interrupted
        if pygame_initialized:
            try:
                if screen is not None:
                    pygame.display.quit()
                pygame.quit()
            except:
                pass  # Ignore errors during cleanup


# Example usage:
games = parse_game_file("./game_history/games_iter_27.txt")
replay_game_pygame(games[0])
