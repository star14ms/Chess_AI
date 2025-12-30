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
    move1 move2 move3 ...
    
    Returns a list of dictionaries, each containing:
    - game_number: int
    - result: str (e.g., "1/2-1/2")
    - reason: str (e.g., "MAX_MOVES")
    - move_count: int
    - moves: List[str] (SAN notation moves)
    """
    games = []
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Pattern to match game headers: "Game X: result (reason, move_count moves)"
    game_pattern = r'Game (\d+):\s*([^\s]+)\s*\(([^,]+),\s*(\d+)\s*moves\)'
    
    # Split content by game headers
    parts = re.split(game_pattern, content)
    
    # Process each game (parts[0] is content before first game, then groups alternate)
    for i in range(1, len(parts), 5):
        if i + 4 < len(parts):
            game_number = int(parts[i])
            result = parts[i + 1]
            reason = parts[i + 2]
            move_count = int(parts[i + 3])
            moves_text = parts[i + 4].strip()
            
            # Parse moves (split by whitespace and filter empty strings)
            moves = [m for m in moves_text.split() if m]
            
            games.append({
                'game_number': game_number,
                'result': result,
                'reason': reason,
                'move_count': move_count,
                'moves': moves
            })
    
    return games


def _boards_from_san(moves_san: List[str], start_fen: Optional[str] = None) -> Tuple[List[chess.Board], List[Optional[str]]]:
    """Return boards for ply index 0..N, and SAN labels (None for initial)."""
    board = chess.Board(start_fen) if start_fen else chess.Board()
    boards = [board.copy(stack=False)]
    sans: List[Optional[str]] = [None]

    for san in moves_san:
        move = board.parse_san(san)
        board.push(move)
        boards.append(board.copy(stack=False))
        sans.append(san)

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
    boards, sans = _boards_from_san(moves)

    pygame_initialized = False
    screen = None

    try:
        pygame.init()
        pygame_initialized = True
        pygame.display.set_caption("Chess replay")

        margin_top = 72
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

            if ply == 0:
                status = "Ply 0 / Start position"
            else:
                status = f"Ply {ply} / {len(boards)-1}   SAN: {san}"
            line2 = ui_font.render(status, True, ui_fg)
            screen.blit(line2, (margin, 38))

            if cached_idx != ply:
                cached_surf = _board_surface_from_chess_svg(
                    board,
                    size_px=board_px,
                    flipped=(not white_perspective),
                )
                cached_idx = ply

            screen.blit(cached_surf, (margin, margin_top))
            pygame.display.flip()

        while running:
            clock.tick(fps)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_RIGHT:
                        idx = min(idx + 1, len(boards) - 1)
                    elif event.key == pygame.K_LEFT:
                        idx = max(idx - 1, 0)
                    elif event.key == pygame.K_HOME:
                        idx = 0
                    elif event.key == pygame.K_END:
                        idx = len(boards) - 1

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
games = parse_game_file("./game_history/games_iter_97.txt")
replay_game_pygame(games[3])
