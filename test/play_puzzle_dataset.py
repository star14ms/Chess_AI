#!/usr/bin/env python3
"""
Play puzzle datasets visually (Pygame), like replay_history for games.

Loads JSON puzzle files (mate_in_5 format: PuzzleId, FEN, Moves UCI)
and lets you step through each puzzle with arrow keys.

Usage:
  python test/play_puzzle_dataset.py <puzzle_json_path> [paths...]
  python test/play_puzzle_dataset.py data/mating_sequences_K_vs_KR.json

Controls:
  - Selection: ↑/↓ Navigate, Click to select, Double-click or Enter to play, Esc Quit
  - Themes: Click "Themes" to expand, click checkboxes to filter (puzzle must have all selected)
  - Replay: ←/→ Step moves, B/Tab Back to selection, Esc/Q Quit
"""

import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import chess
import chess.svg  # Required for chess.svg.board()
import pygame

try:
    import cairosvg
except Exception:
    cairosvg = None
    print("cairosvg not available. Install with: pip install cairosvg")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_puzzle_json(path: Path) -> list[dict]:
    """Load JSON array of puzzles. Tolerates trailing extra data (e.g. from interrupted append)."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        content = f.read()
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        if "Extra data" in str(e):
            # Load only the first valid JSON value (tolerate trailing append corruption)
            data, _ = json.JSONDecoder().raw_decode(content)
        else:
            raise
    return data if isinstance(data, list) else [data]


def _boards_from_uci(fen: str, moves_uci: list[str]):
    """Return (boards, moves) for ply 0..N. boards[0]=initial, moves[0]=None."""
    try:
        board = chess.Board(fen)
    except Exception as e:
        print(f"Warning: Could not parse FEN '{fen}': {e}")
        board = chess.Board()

    boards = [board.copy(stack=False)]
    moves: list[Optional[str]] = [None]

    for uci in moves_uci:
        uci = (uci or "").strip()
        if not uci:
            continue
        try:
            move = chess.Move.from_uci(uci)
            if move not in board.legal_moves:
                print(f"Warning: Illegal move '{uci}' in position, stopping")
                break
            board.push(move)
            boards.append(board.copy(stack=False))
            moves.append(uci)
        except Exception as e:
            print(f"Warning: Could not parse UCI move '{uci}': {e}")
            break

    return boards, moves


def _center_pygame_window() -> None:
    """Center window. Skip on macOS to avoid SDL2 segfaults with pygame._sdl2.video."""
    if sys.platform == "darwin":
        return  # SDL2 Window positioning can segfault on macOS
    try:
        surf = pygame.display.get_surface()
        if surf is None:
            return
        w, h = surf.get_size()
        if hasattr(pygame.display, "get_desktop_sizes") and pygame.display.get_desktop_sizes():
            screen_w, screen_h = pygame.display.get_desktop_sizes()[0]
        else:
            info = pygame.display.Info()
            screen_w = getattr(info, "current_w", 0) or 1920
            screen_h = getattr(info, "current_h", 0) or 1080
        x = max(0, (screen_w - w) // 2)
        y = max(0, (screen_h - h) // 2)
        if sys.platform == "win32":
            from ctypes import windll
            wm_info = pygame.display.get_wm_info()
            hwnd = wm_info.get("window")
            if hwnd is not None and isinstance(hwnd, int):
                windll.user32.MoveWindow(hwnd, x, y, w, h, False)
        else:
            try:
                from pygame._sdl2.video import Window
                win = Window.from_display_module()
                if win is not None:
                    win.position = (x, y)
            except (ImportError, AttributeError, TypeError):
                pass
    except Exception:
        pass


def _board_surface_from_chess_svg(board: chess.Board, size_px: int, flipped: bool) -> pygame.Surface:
    if cairosvg is None:
        raise RuntimeError("cairosvg is required. Install with: pip install cairosvg")
    svg_str = chess.svg.board(board=board, size=size_px, flipped=flipped)
    png_bytes = cairosvg.svg2png(bytestring=svg_str.encode("utf-8"))
    return pygame.image.load(io.BytesIO(png_bytes)).convert_alpha()


def _extract_themes(entry: dict) -> list[str]:
    themes = entry.get("Themes") or entry.get("themes")
    if not themes:
        return []
    if isinstance(themes, list):
        return [str(t).strip() for t in themes if str(t).strip()]
    if isinstance(themes, str):
        return [t for t in themes.split() if t.strip()]
    return []


def _extract_moves(entry: dict) -> list[str]:
    moves_field = entry.get("Moves") or entry.get("moves")
    if not moves_field:
        return []
    if isinstance(moves_field, str):
        return [m for m in moves_field.split() if m.strip()]
    if isinstance(moves_field, list):
        return [str(m).strip() for m in moves_field if str(m).strip()]
    return []


def _extract_fen(entry: dict) -> str:
    return (entry.get("FEN") or entry.get("fen") or chess.STARTING_FEN).strip()


def replay_puzzle_pygame(
    puzzle: dict,
    board_px: int = 768,
    fps: int = 60,
    white_perspective: bool = True,
) -> str:
    """
    Step through one puzzle. Returns 'back' or 'quit'.
    """
    fen = _extract_fen(puzzle)
    moves_uci = _extract_moves(puzzle)
    boards, moves = _boards_from_uci(fen, moves_uci)

    pygame_initialized = False
    screen = None
    return_value = "quit"

    try:
        pygame.init()
        pygame_initialized = True
        pygame.display.set_caption("Puzzle replay")

        puzzle_id = puzzle.get("PuzzleId") or puzzle.get("puzzle_id") or "?"
        themes = puzzle.get("Themes") or puzzle.get("themes") or []
        themes_str = ", ".join(str(t) for t in themes[:8]) if themes else ""

        margin_top = 100
        margin = 16
        width = board_px + margin * 2
        height = board_px + margin_top + margin

        screen = pygame.display.set_mode((width, height))
        _center_pygame_window()
        clock = pygame.time.Clock()

        ui_font = pygame.font.SysFont(None, 24)
        small_font = pygame.font.SysFont(None, 18)
        ui_bg = (20, 20, 20)
        ui_fg = (230, 230, 230)

        idx = 0
        running = True
        cached_idx = None
        cached_surf = None

        def draw(board: chess.Board, move_uci: Optional[str], ply: int) -> None:
            nonlocal cached_idx, cached_surf
            screen.fill(ui_bg)
            line1 = ui_font.render(f"Puzzle: {puzzle_id}", True, ui_fg)
            screen.blit(line1, (margin, 10))
            if themes_str:
                line2 = small_font.render(f"Themes: {themes_str}", True, (150, 180, 220))
                screen.blit(line2, (margin, 38))
            if ply == 0:
                status = "Ply 0 / Start position"
            else:
                status = f"Ply {ply} / {len(boards)-1}   UCI: {move_uci}"
            line3 = ui_font.render(status, True, ui_fg)
            y = 62 if themes_str else 38
            screen.blit(line3, (margin, y))
            controls = small_font.render("←/→: Navigate | B/Tab: Back | Esc/Q: Quit", True, (150, 150, 150))
            screen.blit(controls, (margin, y + 25))

            if cached_idx != ply:
                cached_surf = _board_surface_from_chess_svg(
                    board, size_px=board_px, flipped=(not white_perspective)
                )
                cached_idx = ply
            screen.blit(cached_surf, (margin, margin_top))
            pygame.display.flip()

        pygame.key.set_repeat(200, 50)

        while running:
            clock.tick(fps)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    return_value = "quit"
                    break
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                        return_value = "quit"
                        break
                    elif event.key in (pygame.K_b, pygame.K_TAB):
                        running = False
                        return_value = "back"
                        break
                    elif event.key in (pygame.K_RIGHT, pygame.K_d, pygame.K_SPACE):
                        idx = min(idx + 1, len(boards) - 1)
                    elif event.key in (pygame.K_LEFT, pygame.K_a, pygame.K_BACKSPACE):
                        idx = max(idx - 1, 0)
                    elif event.key == pygame.K_HOME:
                        idx = 0
                    elif event.key == pygame.K_END:
                        idx = len(boards) - 1

            if not running:
                break
            draw(boards[idx], moves[idx], idx)

    except Exception as e:
        print(f"Error in puzzle replay: {e}")
        return_value = "quit"
    finally:
        if pygame_initialized:
            try:
                if return_value != "back" and screen is not None:
                    pygame.display.quit()
                if return_value != "back":
                    pygame.quit()
            except Exception:
                pass

    return return_value


def load_puzzles(paths: list[Path], max_puzzles: Optional[int] = None) -> list[dict]:
    puzzles = []
    for path in paths:
        path = Path(path)
        if not path.exists():
            print(f"Skip (not found): {path}")
            continue
        try:
            entries = _load_puzzle_json(path)
        except Exception as e:
            print(f"Skip (parse error): {path}: {e}")
            continue
        count = 0
        for entry in entries:
            fen = _extract_fen(entry)
            moves = _extract_moves(entry)
            if not fen or not moves:
                continue
            puzzles.append(entry)
            count += 1
            if max_puzzles is not None and len(puzzles) >= max_puzzles:
                break
        print(f"Loaded {count} puzzles from {path}")
        if max_puzzles is not None and len(puzzles) >= max_puzzles:
            break
    return puzzles


def select_and_replay_puzzle(
    puzzles: list[dict],
    board_px: int = 768,
    fps: int = 60,
    white_perspective: bool = True,
    max_display: int = 500,
) -> None:
    """Selection menu + replay. Theme filter on right (like replay_history)."""
    if not puzzles:
        print("No puzzles to display")
        return

    # Collect all themes
    all_themes: list[str] = []
    for p in puzzles:
        for t in _extract_themes(p):
            if t and t not in all_themes:
                all_themes.append(t)
    all_themes = sorted(all_themes)

    selected_themes: set[str] = set()
    theme_state = [True]  # dropdown open
    theme_panel_width = 140
    theme_checkbox_height = 22
    theme_option_margin = 6
    theme_list_max_visible = 14

    pygame_initialized = False
    screen = None

    def _filter_puzzles() -> list[dict]:
        if not selected_themes:
            return puzzles
        return [
            p for p in puzzles
            if _extract_themes(p) and selected_themes <= {str(t).strip() for t in _extract_themes(p) if t}
        ]

    def _get_display_puzzles() -> list[dict]:
        filtered = _filter_puzzles()
        return filtered[:max_display] if max_display else filtered

    display_puzzles = _get_display_puzzles()

    try:
        pygame.init()
        pygame_initialized = True
        pygame.display.set_caption("Puzzle dataset selector")

        margin = 20
        header_height = 80
        rows_per_page = 16
        item_height = 48
        list_panel_width = 900
        theme_option_width = theme_panel_width + margin - theme_option_margin * 2
        window_width = list_panel_width + theme_panel_width + margin * 2
        window_height = header_height + min(len(display_puzzles), rows_per_page) * item_height + margin * 2

        screen = pygame.display.set_mode((window_width, window_height))
        _center_pygame_window()
        clock = pygame.time.Clock()

        font = pygame.font.SysFont(None, 20)
        small_font = pygame.font.SysFont(None, 16)
        bg_color = (30, 30, 30)
        item_bg = (40, 40, 40)
        selected_bg = (70, 100, 150)
        header_row_bg = (55, 55, 55)
        text_color = (230, 230, 230)
        theme_panel_bg = (35, 35, 40)

        selected_row = 0
        scroll_offset = 0
        running = True
        hovered_theme_index: list[Optional[int]] = [None]
        hovered_puzzle_index: list[Optional[int]] = [None]
        hover_bg = (55, 65, 85)  # Slightly lighter than item_bg for hover
        last_click_idx: list[Optional[int]] = [None]
        last_click_time: list[float] = [0.0]

        def truncate(s: str, n: int = 60) -> str:
            return (s[: n - 3] + "...") if len(s) > n else s

        def _rebuild_display() -> None:
            nonlocal display_puzzles, selected_row, scroll_offset
            display_puzzles = _get_display_puzzles()
            selected_row = max(0, min(selected_row, len(display_puzzles) - 1))
            scroll_offset = min(scroll_offset, max(0, len(display_puzzles) - rows_per_page))

        while running:
            clock.tick(fps)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                        break
                    elif event.key in (pygame.K_UP, pygame.K_w):
                        selected_row = max(0, selected_row - 1)
                        if selected_row < scroll_offset:
                            scroll_offset = selected_row
                    elif event.key in (pygame.K_DOWN, pygame.K_s):
                        selected_row = min(len(display_puzzles) - 1, selected_row + 1)
                        if selected_row >= scroll_offset + rows_per_page:
                            scroll_offset = selected_row - rows_per_page + 1
                    elif event.key == pygame.K_HOME:
                        selected_row = 0
                        scroll_offset = 0
                    elif event.key == pygame.K_END:
                        selected_row = len(display_puzzles) - 1
                        scroll_offset = max(0, len(display_puzzles) - rows_per_page)
                    elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                        puzzle = display_puzzles[selected_row]
                        ret = replay_puzzle_pygame(
                            puzzle,
                            board_px=board_px,
                            fps=fps,
                            white_perspective=white_perspective,
                        )
                        if ret == "quit":
                            running = False
                            break
                        pygame.display.set_caption("Puzzle dataset selector")
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mouse_x, mouse_y = event.pos
                    list_start_y = header_height + margin
                    list_end_y = list_start_y + min(rows_per_page, len(display_puzzles)) * item_height
                    # Puzzle list click (check first - has priority when click is in list area)
                    if margin <= mouse_x < list_panel_width and list_start_y <= mouse_y <= list_end_y:
                        row = (mouse_y - list_start_y) // item_height
                        idx = scroll_offset + row
                        if 0 <= idx < len(display_puzzles):
                            now = time.monotonic()
                            is_double = (
                                last_click_idx[0] == idx
                                and (now - last_click_time[0]) < 0.4
                            )
                            last_click_idx[0] = idx
                            last_click_time[0] = now
                            selected_row = idx
                            if is_double:
                                puzzle = display_puzzles[selected_row]
                                ret = replay_puzzle_pygame(
                                    puzzle,
                                    board_px=board_px,
                                    fps=fps,
                                    white_perspective=white_perspective,
                                )
                                if ret == "quit":
                                    running = False
                                    break
                                pygame.display.set_caption("Puzzle dataset selector")
                    # Theme panel (only when click is in theme area)
                    elif mouse_x >= list_panel_width:
                        theme_btn_rect = pygame.Rect(
                            list_panel_width + margin, header_height + margin,
                            theme_panel_width - margin, 28
                        )
                        if theme_btn_rect.collidepoint(mouse_x, mouse_y) and all_themes:
                            theme_state[0] = not theme_state[0]
                        elif theme_state[0] and all_themes:
                            theme_option_x = list_panel_width + theme_option_margin
                            theme_list_y = header_height + margin + 35
                            for ti, theme_name in enumerate(all_themes):
                                cb_y = theme_list_y + ti * theme_checkbox_height
                                cb_rect = pygame.Rect(
                                    theme_option_x, cb_y, theme_option_width, theme_checkbox_height
                                )
                                if cb_rect.collidepoint(mouse_x, mouse_y):
                                    if theme_name in selected_themes:
                                        selected_themes.discard(theme_name)
                                    else:
                                        selected_themes.add(theme_name)
                                    _rebuild_display()
                                    break
                elif event.type == pygame.MOUSEMOTION:
                    hovered_theme_index[0] = None
                    hovered_puzzle_index[0] = None
                    mouse_x, mouse_y = event.pos
                    list_start_y = header_height + margin
                    list_end_y = list_start_y + min(rows_per_page, len(display_puzzles)) * item_height
                    if margin <= mouse_x <= list_panel_width and list_start_y <= mouse_y <= list_end_y:
                        row = (mouse_y - list_start_y) // item_height
                        idx = scroll_offset + row
                        if 0 <= idx < len(display_puzzles):
                            hovered_puzzle_index[0] = idx
                    elif theme_state[0] and all_themes:
                        theme_option_x = list_panel_width + theme_option_margin
                        theme_list_y = header_height + margin + 35
                        for ti in range(len(all_themes)):
                            cb_y = theme_list_y + ti * theme_checkbox_height
                            cb_rect = pygame.Rect(
                                theme_option_x, cb_y, theme_option_width, theme_checkbox_height
                            )
                            if cb_rect.collidepoint(event.pos):
                                hovered_theme_index[0] = ti
                                break

            if not running:
                break

            if not display_puzzles:
                screen.fill(bg_color)
                msg = font.render("No puzzles match selected themes. Clear filters.", True, text_color)
                screen.blit(msg, (margin, header_height))
                pygame.display.flip()
                continue

            screen.fill(bg_color)
            header = font.render(
                f"Puzzles ({len(display_puzzles)} filtered, {len(puzzles)} total) | ↑/↓: Navigate | Click: Select | Double-click/Enter: Play | Esc: Quit",
                True,
                text_color,
            )
            screen.blit(header, (margin, 15))

            # Theme filter panel (right side)
            theme_panel_x = list_panel_width
            theme_panel_rect = pygame.Rect(theme_panel_x, 0, theme_panel_width + margin, window_height)
            pygame.draw.rect(screen, theme_panel_bg, theme_panel_rect)
            pygame.draw.line(screen, (60, 60, 65), (theme_panel_x, 0), (theme_panel_x, window_height))
            if all_themes:
                theme_btn_rect = pygame.Rect(
                    theme_panel_x + margin, header_height + margin,
                    theme_panel_width - margin, 28
                )
                btn_label = "Themes ▼" if theme_state[0] else "Themes ▶"
                if selected_themes:
                    btn_label += f" ({len(selected_themes)})"
                theme_btn_color = selected_bg if theme_state[0] else header_row_bg
                pygame.draw.rect(screen, theme_btn_color, theme_btn_rect, border_radius=4)
                theme_btn_text = small_font.render(btn_label, True, text_color)
                screen.blit(theme_btn_text, (theme_btn_rect.x + 6, theme_btn_rect.y + 6))
                if theme_state[0]:
                    theme_option_x = theme_panel_x + theme_option_margin
                    theme_list_y = header_height + margin + 35
                    for ti, theme_name in enumerate(all_themes):
                        row_y = theme_list_y + ti * theme_checkbox_height
                        row_center_y = row_y + theme_checkbox_height // 2
                        is_checked = theme_name in selected_themes
                        is_hovered = ti == hovered_theme_index[0]
                        row_rect = pygame.Rect(
                            theme_option_x, row_y, theme_option_width, theme_checkbox_height
                        )
                        if is_hovered:
                            pygame.draw.rect(screen, (70, 90, 120), row_rect, border_radius=4)
                        cb_size = 16
                        cb_y = row_center_y - cb_size // 2
                        cb_rect = pygame.Rect(theme_option_x, cb_y, cb_size, cb_size)
                        pygame.draw.rect(screen, (60, 60, 65), cb_rect, border_radius=2)
                        if is_checked:
                            pygame.draw.rect(
                                screen, (100, 180, 100), cb_rect.inflate(-4, -4), border_radius=2
                            )
                        theme_text_color = (255, 255, 255) if is_hovered else text_color
                        theme_text = small_font.render(theme_name, True, theme_text_color)
                        screen.blit(theme_text, (cb_rect.right + 6, row_center_y - theme_text.get_height() // 2))

            list_start_y = header_height + margin
            for i in range(rows_per_page):
                idx = scroll_offset + i
                if idx >= len(display_puzzles):
                    break
                puzzle = display_puzzles[idx]
                y_pos = list_start_y + i * item_height
                is_selected = idx == selected_row
                is_hovered = idx == hovered_puzzle_index[0]
                if is_selected:
                    item_bg_color = selected_bg
                elif is_hovered:
                    item_bg_color = hover_bg
                else:
                    item_bg_color = item_bg
                item_rect = pygame.Rect(margin, y_pos, list_panel_width - margin * 2, item_height - 2)
                pygame.draw.rect(screen, item_bg_color, item_rect)

                pid = puzzle.get("PuzzleId") or puzzle.get("puzzle_id") or "?"
                fen = _extract_fen(puzzle)
                moves = _extract_moves(puzzle)
                line_color = (255, 255, 255) if (is_selected or is_hovered) else text_color
                line1 = font.render(f"{pid} | {len(moves)} moves", True, line_color)
                screen.blit(line1, (margin + 5, y_pos + 4))
                line2 = small_font.render(truncate(fen, 70), True, (180, 180, 200))
                screen.blit(line2, (margin + 5, y_pos + 24))

            pygame.display.flip()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if pygame_initialized:
            try:
                if screen is not None:
                    pygame.display.quit()
                pygame.quit()
            except Exception:
                pass


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Play puzzle datasets visually (like replay_history)")
    parser.add_argument("paths", nargs="+", help="Puzzle JSON file(s)")
    parser.add_argument("--list", "-l", action="store_true", help="Only list puzzles, don't open viewer")
    parser.add_argument(
        "--max", "-n", type=int, default=None,
        help="Max puzzles to load (try -n 100 if segfault on macOS)",
    )
    args = parser.parse_args()

    paths = [Path(p) for p in args.paths]
    puzzles = load_puzzles(paths, max_puzzles=args.max)
    if not puzzles:
        print("No puzzles loaded.")
        sys.exit(1)

    if args.list:
        print(f"Loaded {len(puzzles)} puzzles. Run without --list to open the viewer.")
        return

    select_and_replay_puzzle(puzzles)


if __name__ == "__main__":
    main()
