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
    """
    Open a Pygame window. Use ←/→ to move back/forward, Esc to quit, B to go back to selection.
    
    Returns:
        str: 'back' if user wants to go back to selection, 'quit' if user wants to exit completely, None otherwise
    """
    moves = list(game_data.get("moves", []))
    initial_fen = game_data.get("initial_fen")
    boards, sans = _boards_from_san(moves, start_fen=initial_fen)

    pygame_initialized = False
    screen = None

    try:
        pygame.init()
        pygame_initialized = True
        pygame.display.set_caption("Chess replay")

        # Adjust margin_top based on whether we have metadata and controls hint
        has_metadata = game_data.get('initial_quality') or game_data.get('reward') is not None
        margin_top = 120 if has_metadata else 96  # Extra space for controls hint
        margin = 16
        width = board_px + margin * 2
        height = board_px + margin_top + margin

        screen = pygame.display.set_mode((width, height))
        clock = pygame.time.Clock()

        ui_font = pygame.font.SysFont(None, 24)
        small_font = pygame.font.SysFont(None, 18)
        ui_bg = (20, 20, 20)
        ui_fg = (230, 230, 230)

        idx = 0
        running = True
        return_value = 'quit'  # Default return value

        # Simple cache so we don't rerender SVG every frame
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
            
            # Show controls hint
            controls_text = "←/→: Navigate | B/Tab: Back to selection | Esc: Quit"
            controls_surface = small_font.render(controls_text, True, (150, 150, 150))
            screen.blit(controls_surface, (margin, y_offset + 25))

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
                    return_value = 'quit'
                    break
                elif event.type == pygame.KEYDOWN:
                    old_idx = idx
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        return_value = 'quit'
                        break
                    elif event.key == pygame.K_b or event.key == pygame.K_TAB:
                        # Go back to game selection
                        running = False
                        return_value = 'back'
                        break
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
            
            # Break from while loop if we're returning
            if not running:
                break
            
            # Always draw, but force cache invalidation if idx changed
            if idx_changed:
                cached_idx = None  # Force cache invalidation
            
            draw(boards[idx], sans[idx], idx)

    except KeyboardInterrupt:
        # Handle cell interruption in Jupyter
        return_value = 'quit'
    except Exception as e:
        print(f"Error in pygame replay: {e}")
        return_value = 'quit'
    finally:
        # Cleanup display, but only quit pygame if not going back to selection
        if pygame_initialized:
            try:
                if screen is not None:
                    pygame.display.quit()
                # Only quit pygame completely if we're not going back to selection
                # This allows the selection menu to continue running
                if return_value != 'back':
                    pygame.quit()
            except:
                pass  # Ignore errors during cleanup
    
    return return_value


def select_and_replay_game(
    games: List[Dict],
    board_px: int = 768,
    fps: int = 60,
    white_perspective: bool = True,
):
    """
    Display a game selection menu in pygame, then replay the selected game.
    
    Controls:
    - ↑/↓ or W/S: Navigate through games
    - Enter/Space: Select and replay game
    - Esc: Exit without replaying
    
    In replay view:
    - B or Tab: Go back to game selection
    - Esc: Exit completely
    """
    if not games:
        print("No games to display")
        return
    
    pygame_initialized = False
    screen = None
    
    try:
        pygame.init()
        pygame_initialized = True
        pygame.display.set_caption("Chess Game Selector")
        
        # Calculate window size
        margin = 20
        line_height = 30
        header_height = 60
        games_per_page = 15
        item_height = line_height + 5
        window_width = 1000
        window_height = header_height + min(len(games), games_per_page) * item_height + margin * 2
        
        screen = pygame.display.set_mode((window_width, window_height))
        clock = pygame.time.Clock()
        
        # Fonts
        title_font = pygame.font.SysFont(None, 32)
        font = pygame.font.SysFont(None, 20)
        small_font = pygame.font.SysFont(None, 16)
        
        # Colors
        bg_color = (30, 30, 30)
        header_bg = (50, 50, 50)
        item_bg = (40, 40, 40)
        selected_bg = (70, 100, 150)
        text_color = (230, 230, 230)
        selected_text = (255, 255, 255)
        result_win = (100, 200, 100)
        result_draw = (200, 200, 100)
        result_loss = (200, 100, 100)
        
        selected_idx = 0
        scroll_offset = 0
        running = True
        
        def truncate_fen(fen: str, max_len: int = 50) -> str:
            """Truncate FEN string for display."""
            if not fen:
                return "Standard starting position"
            if len(fen) <= max_len:
                return fen
            return fen[:max_len-3] + "..."
        
        def get_first_player_from_fen(fen: Optional[str]) -> Optional[str]:
            """Determine which player moves first from FEN string.
            
            Returns:
                'w' if white moves first, 'b' if black moves first, None if cannot determine
            """
            if not fen:
                return 'w'  # Standard starting position: white moves first
            
            # FEN format: "board_position active_color castling en_passant halfmove fullmove"
            # The second field (index 1) is the active color
            parts = fen.split()
            if len(parts) >= 2:
                active_color = parts[1].lower()
                if active_color in ('w', 'b'):
                    return active_color
            
            # Default to white if cannot parse
            return 'w'
        
        def get_result_color(result: str, first_player: Optional[str] = None) -> tuple:
            """Get color based on game result from the first player's perspective.
            
            Args:
                result: Game result ("1-0", "0-1", or "1/2-1/2")
                first_player: 'w' if white moves first, 'b' if black moves first, None for default (white)
            
            Returns:
                Green if first player won, red if first player lost, yellow if draw
            """
            if first_player is None:
                first_player = 'w'  # Default: white moves first
            
            if result == "1-0":
                # White won
                if first_player == 'w':
                    return result_win  # First player (white) won -> green
                else:
                    return result_loss  # First player (black) lost -> red
            elif result == "0-1":
                # Black won
                if first_player == 'b':
                    return result_win  # First player (black) won -> green
                else:
                    return result_loss  # First player (white) lost -> red
            else:  # Draw (1/2-1/2)
                return result_draw  # Draw -> yellow
        
        while running:
            clock.tick(fps)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        return
                    elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                        # Select and replay
                        result = replay_game_pygame(games[selected_idx], board_px, fps, white_perspective)
                        # If user wants to go back, continue the loop (don't return)
                        if result == 'back':
                            # Reinitialize pygame for the selection menu
                            pygame.init()
                            screen = pygame.display.set_mode((window_width, window_height))
                            continue
                        elif result == 'quit':
                            running = False
                            return
                    elif event.key in (pygame.K_UP, pygame.K_w):
                        selected_idx = max(0, selected_idx - 1)
                        # Auto-scroll
                        if selected_idx < scroll_offset:
                            scroll_offset = selected_idx
                    elif event.key in (pygame.K_DOWN, pygame.K_s):
                        selected_idx = min(len(games) - 1, selected_idx + 1)
                        # Auto-scroll
                        if selected_idx >= scroll_offset + games_per_page:
                            scroll_offset = selected_idx - games_per_page + 1
                    elif event.key == pygame.K_HOME:
                        selected_idx = 0
                        scroll_offset = 0
                    elif event.key == pygame.K_END:
                        selected_idx = len(games) - 1
                        scroll_offset = max(0, len(games) - games_per_page)
            
            # Draw
            screen.fill(bg_color)
            
            # Header
            header_rect = pygame.Rect(0, 0, window_width, header_height)
            pygame.draw.rect(screen, header_bg, header_rect)
            title_text = title_font.render(f"Select Game to Replay ({len(games)} games)", True, text_color)
            screen.blit(title_text, (margin, 15))
            
            # Instructions
            inst_text = small_font.render("↑/↓: Navigate | Enter/Space: Select | Esc: Exit | In replay: B/Tab to go back", True, (150, 150, 150))
            screen.blit(inst_text, (window_width - margin - inst_text.get_width(), header_height - 25))
            
            # Game list
            visible_start = scroll_offset
            visible_end = min(scroll_offset + games_per_page, len(games))
            
            for i in range(visible_start, visible_end):
                game = games[i]
                list_idx = i - scroll_offset
                y_pos = header_height + list_idx * item_height + margin
                
                # Highlight selected item
                is_selected = (i == selected_idx)
                item_bg_color = selected_bg if is_selected else item_bg
                text_color_use = selected_text if is_selected else text_color
                
                item_rect = pygame.Rect(margin, y_pos, window_width - margin * 2, item_height - 2)
                pygame.draw.rect(screen, item_bg_color, item_rect)
                
                # Game number and result
                game_num = game.get('game_number', i + 1)
                result = game.get('result', '?')
                reason = game.get('reason', '?')
                move_count = game.get('move_count', 0)
                
                # Determine first player from initial FEN
                first_player = get_first_player_from_fen(game.get('initial_fen'))
                result_color = get_result_color(result, first_player)
                header_text = f"Game {game_num}: {result} ({reason}, {move_count} moves)"
                header_surface = font.render(header_text, True, result_color)
                screen.blit(header_surface, (margin + 5, y_pos + 3))
                
                # Initial FEN (truncated)
                initial_fen = game.get('initial_fen')
                fen_display = truncate_fen(initial_fen)
                fen_surface = small_font.render(f"FEN: {fen_display}", True, (180, 180, 180))
                screen.blit(fen_surface, (margin + 5, y_pos + 20))
                
                # Additional info (quality, reward) if available
                info_parts = []
                if game.get('initial_quality'):
                    info_parts.append(f"Quality: {game['initial_quality']}")
                if game.get('reward') is not None:
                    info_parts.append(f"Reward: {game['reward']:.4f}")
                
                if info_parts:
                    info_text = " | ".join(info_parts)
                    info_surface = small_font.render(info_text, True, (150, 150, 200))
                    screen.blit(info_surface, (window_width - margin - info_surface.get_width() - 5, y_pos + 20))
            
            # Scrollbar (if needed)
            if len(games) > games_per_page:
                scrollbar_width = 10
                scrollbar_x = window_width - margin - scrollbar_width
                scrollbar_height = (window_height - header_height - margin * 2)
                scrollbar_rect = pygame.Rect(scrollbar_x, header_height + margin, scrollbar_width, scrollbar_height)
                pygame.draw.rect(screen, (60, 60, 60), scrollbar_rect)
                
                # Thumb
                thumb_height = scrollbar_height * games_per_page / len(games)
                thumb_y = header_height + margin + (scrollbar_height - thumb_height) * scroll_offset / (len(games) - games_per_page)
                thumb_rect = pygame.Rect(scrollbar_x, thumb_y, scrollbar_width, thumb_height)
                pygame.draw.rect(screen, (100, 100, 100), thumb_rect)
            
            pygame.display.flip()
    
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error in game selector: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if pygame_initialized:
            try:
                if screen is not None:
                    pygame.display.quit()
                pygame.quit()
            except:
                pass


# Example usage:
if __name__ == "__main__":
    games = parse_game_file("./game_history/games_iter_60.txt")
    select_and_replay_game(games)
