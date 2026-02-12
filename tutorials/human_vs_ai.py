import argparse
import sys
import re
import random
from typing import Optional, Tuple

import gymnasium as gym
import chess

# Ensure chess_gym is registered
import sys
import os
sys.path.append(os.path.abspath('.'))
import chess_gym  # noqa: F401

from utils.policy_human import sample_action_v2 as heuristic_policy
from _early_version.config import IMG_DIR

# Optional DQN/MCTS model imports
try:
    import torch  # type: ignore[import-not-found]
    import numpy as np  # type: ignore[import-not-found]
    from omegaconf import OmegaConf  # type: ignore[import-not-found]
    from MCTS.training_modules.chess import create_chess_network  # type: ignore[import-not-found]
    from MCTS.mcts_algorithm import MCTS  # type: ignore[import-not-found]
    from MCTS.mcts_node import MCTSNode  # type: ignore[import-not-found]
except Exception:
    torch = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]
    OmegaConf = None  # type: ignore[assignment]
    create_chess_network = None  # type: ignore[assignment]
    MCTS = None  # type: ignore[assignment]
    MCTSNode = None  # type: ignore[assignment]

try:
    import pygame  # Optional; only used when --pygame
except Exception:
    pygame = None  # type: ignore[assignment]

import time

def create_env(use_4672_action_space: bool, show_possible_actions: bool) -> gym.Env:
    action_space_size = 4672 if use_4672_action_space else 1700
    env = gym.make(
        "Chess-v0",
        render_mode=None,
        show_possible_actions=show_possible_actions,
        action_space_size=action_space_size,
    )
    return env


def reset_env(env: gym.Env, fen: Optional[str]) -> Tuple:
    if fen:
        return env.reset(options={"fen": fen})
    return env.reset()


def convert_move_to_action_id(env: gym.Env, move: chess.Move) -> Optional[int]:
    board = env.action_space.board
    action_id: Optional[int] = None

    if hasattr(board, "move_to_action_id"):
        try:
            action_id = board.move_to_action_id(move)  # type: ignore[attr-defined]
        except Exception:
            action_id = None

    if action_id is None and hasattr(env.action_space, "_move_to_action"):
        try:
            action_id = env.action_space._move_to_action(move, return_id=True)  # type: ignore[attr-defined]
        except Exception:
            action_id = None

    if action_id is None:
        return None

    try:
        legal_actions = set(getattr(board, "legal_actions", []))
        if legal_actions and action_id not in legal_actions:
            return None
    except Exception:
        pass
    return action_id


def try_parse_move(board: chess.Board, text: str) -> Optional[chess.Move]:
    text = text.strip()
    try:
        move = board.parse_san(text)
        return move
    except Exception:
        pass
    try:
        move = chess.Move.from_uci(text)
        if move in board.legal_moves:
            return move
    except Exception:
        pass
    return None


def parse_human_input_to_action(env: gym.Env, raw: str) -> Optional[int]:
    board = env.action_space.board
    s = raw.strip()

    if re.fullmatch(r"\d+", s):
        try:
            candidate = int(s)
            legal = getattr(board, "legal_actions", None)
            if not legal or candidate in legal:
                return candidate
        except Exception:
            return None

    move = try_parse_move(board, s)
    if move is None:
        return None
    return convert_move_to_action_id(env, move)


_DQN_MODEL = None
_DQN_DEVICE = None
_DQN_CFG = None


def _load_dqn_model(action_space_size: int):
    global _DQN_MODEL, _DQN_DEVICE, _DQN_CFG
    if torch is None or OmegaConf is None or create_chess_network is None:
        raise RuntimeError("Required packages for DQN are not available. Install torch and omegaconf.")

    # Prefer CUDA; avoid MPS due to missing ops in some kernels
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    _DQN_DEVICE = torch.device(device_str)

    # Load config and initialize network
    cfg_path = os.path.join(os.path.abspath('.'), 'config', 'train_mcts.yaml')
    checkpoint_path = os.path.join(os.path.abspath('.'), 'checkpoints', 'model.pth')
    cfg = OmegaConf.load(cfg_path)
    _DQN_CFG = cfg  # Store config for MCTS use

    # Warn if action space sizes mismatch
    try:
        cfg_size = int(cfg.network.action_space_size)
        if cfg_size != action_space_size:
            print(f"[Warning] DQN checkpoint configured for action_space_size={cfg_size}, but env is {action_space_size}.")
    except Exception:
        pass

    model = create_chess_network(cfg, _DQN_DEVICE)
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = state.get('model_state_dict', state)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(_DQN_DEVICE)
    _DQN_MODEL = model


def choose_ai_action(env: gym.Env, mode: str, mcts_iterations: int = 400) -> int:
    board = env.action_space.board
    if mode == "heuristic":
        # Enable multiprocessing with auto workers for faster root evaluation
        result = heuristic_policy(board, return_id=True, use_multiprocessing=True)
        action_id = result[0] if isinstance(result, tuple) else result
        legal = getattr(board, "legal_actions", None)
        if action_id is None or (isinstance(action_id, int) and action_id == 0) or (legal and action_id not in legal):
            if legal:
                return random.choice(list(legal))
            return env.action_space.sample()
        return int(action_id)
    if mode == "mcts":
        # Ensure model is loaded
        if _DQN_MODEL is None:
            action_space_size = getattr(env, 'action_space_size', 4672)
            _load_dqn_model(action_space_size)

        # Safety: if still None, fallback
        if _DQN_MODEL is None or torch is None or MCTS is None or MCTSNode is None or np is None:
            legal = getattr(board, "legal_actions", None)
            if legal:
                return random.choice(list(legal))
            return env.action_space.sample()

        # Use MCTS for action selection (guarantees legal moves)
        try:
            # Create MCTS root node
            root_node = MCTSNode(board.copy())
            
            # Initialize MCTS
            mcts_player = MCTS(
                network=_DQN_MODEL,
                device=_DQN_DEVICE,
                env=None,  # No rendering during MCTS
                C_puct=_DQN_CFG.mcts.c_puct,
                dirichlet_alpha=_DQN_CFG.mcts.dirichlet_alpha,
                dirichlet_epsilon=_DQN_CFG.mcts.dirichlet_epsilon,
                action_space_size=_DQN_CFG.network.action_space_size,
                history_steps=_DQN_CFG.env.history_steps
            )
            
            # Run MCTS search
            mcts_player.search(
                root_node, 
                iterations=mcts_iterations,
                batch_size=_DQN_CFG.mcts.batch_size,
                progress=None  # No progress bar
            )
            
            # Get policy distribution from MCTS
            temperature = 0.1  # Low temperature for near-deterministic play
            mcts_policy = mcts_player.get_policy_distribution(root_node, temperature=temperature)
            
            # Select highest probability move (deterministic)
            action_id = int(np.argmax(mcts_policy) + 1)  # +1 for 1-indexed
            
            # Verify action is legal
            legal = getattr(board, "legal_actions", None)
            if legal and action_id not in legal:
                print(f"[Warning] MCTS selected illegal action {action_id}, falling back to random legal move")
                return random.choice(list(legal))
            
            return action_id
        except Exception as e:
            print(f"[Error] MCTS failed: {e}. Falling back to random legal move.")
            legal = getattr(board, "legal_actions", None)
            if legal:
                return random.choice(list(legal))
            return env.action_space.sample()
    if mode == "dqn":
        # Ensure model is loaded
        if _DQN_MODEL is None:
            action_space_size = getattr(env, 'action_space_size', 4672)
            _load_dqn_model(action_space_size)

        # Safety: if still None, fallback
        if _DQN_MODEL is None or torch is None:
            legal = getattr(board, "legal_actions", None)
            if legal:
                return random.choice(list(legal))
            return env.action_space.sample()

        # Build observation
        obs_np = board.get_board_vector()
        obs = torch.from_numpy(obs_np).unsqueeze(0).to(device=_DQN_DEVICE, dtype=torch.float32)
        with torch.no_grad():
            policy_logits, _ = _DQN_MODEL(obs)
        policy = policy_logits.squeeze(0)

        # Mask illegal actions (IDs are 1-based)
        legal = getattr(board, "legal_actions", None)
        if not legal:
            return env.action_space.sample()  # Fallback if no legal info
        legal_ids = list(legal)
        indices = torch.tensor([aid - 1 for aid in legal_ids if 1 <= aid <= policy.shape[-1]], device=policy.device, dtype=torch.long)
        if indices.numel() == 0:
            return env.action_space.sample()
        legal_policy = policy.index_select(0, indices)
        best_local = int(torch.argmax(legal_policy).item())
        best_global = int(indices[best_local].item())
        return best_global + 1
    legal = getattr(board, "legal_actions", None)
    if legal:
        return random.choice(list(legal))
    return env.action_space.sample()  # Fallback


def print_board(board: chess.Board) -> None:
    print(str(board))
    print(f"Turn: {'White' if board.turn else 'Black'}")


def print_legal_actions(board: chess.Board) -> None:
    try:
        legal_ids = getattr(board, "legal_actions", [])
        if not legal_ids:
            print("No legal actions available.")
            return
        id_to_san = []
        for aid in legal_ids:
            move = None
            try:
                move = board.action_id_to_move(aid)  # type: ignore[attr-defined]
            except Exception:
                move = None
            if move is not None:
                try:
                    san = board.san(move)
                except Exception:
                    san = move.uci()
                id_to_san.append((aid, san))
        preview = ", ".join(f"{aid}:{san}" for aid, san in id_to_san[:30])
        if preview:
            print(f"Legal actions (first 30): {preview}")
        else:
            print(f"Legal action ids: {list(legal_ids)[:30]}")
    except Exception:
        pass


def game_result_message(board: chess.Board, reward: float, terminated: bool, truncated: bool) -> str:
    if terminated or truncated:
        if reward == 1:
            return "White wins!"
        if reward == -1:
            return "Black wins!"
        if hasattr(board, "is_stalemate") and board.is_stalemate():
            return "It's a stalemate!"
        if hasattr(board, "is_insufficient_material") and board.is_insufficient_material():
            return "It's a draw due to insufficient material!"
        if hasattr(board, "can_claim_draw") and board.can_claim_draw():
            return "It's a draw by repetition or 50-move rule!"
        if truncated:
            return "Game truncated."
        return "Game ended in a draw (other reason)."
    return ""


def load_piece_images(cell_size: int):
    if pygame is None:
        raise RuntimeError("pygame is not available. Install pygame or run without --pygame.")
    size = (cell_size, cell_size)
    def load(name: str):
        path = os.path.join(IMG_DIR, name)
        img = pygame.image.load(path)
        return pygame.transform.scale(img, size)
    images = {
        (chess.PAWN, chess.WHITE): load("pawn_w.png"),
        (chess.PAWN, chess.BLACK): load("pawn_b.png"),
        (chess.KNIGHT, chess.WHITE): load("knight_w.png"),
        (chess.KNIGHT, chess.BLACK): load("knight_b.png"),
        (chess.BISHOP, chess.WHITE): load("bishop_w.png"),
        (chess.BISHOP, chess.BLACK): load("bishop_b.png"),
        (chess.ROOK, chess.WHITE): load("rook_w.png"),
        (chess.ROOK, chess.BLACK): load("rook_b.png"),
        (chess.QUEEN, chess.WHITE): load("queen_w.png"),
        (chess.QUEEN, chess.BLACK): load("queen_b.png"),
        (chess.KING, chess.WHITE): load("king_w.png"),
        (chess.KING, chess.BLACK): load("king_b.png"),
    }
    return images


def square_from_xy(x: int, y: int, cell: int, human_white_bottom: bool) -> Optional[int]:
    file_idx = x // cell
    rank_idx = y // cell
    if file_idx < 0 or file_idx > 7 or rank_idx < 0 or rank_idx > 7:
        return None
    if human_white_bottom:
        file_board = file_idx
        rank_board = 7 - rank_idx
    else:
        file_board = 7 - file_idx
        rank_board = rank_idx
    return chess.square(file_board, rank_board)


def xy_from_square(square: int, cell: int, human_white_bottom: bool) -> tuple[int, int]:
    file_board = chess.square_file(square)
    rank_board = chess.square_rank(square)
    if human_white_bottom:
        file_idx = file_board
        rank_idx = 7 - rank_board
    else:
        file_idx = 7 - file_board
        rank_idx = rank_board
    return file_idx * cell, rank_idx * cell


def find_move_from_squares(board: chess.Board, from_sq: int, to_sq: int) -> Optional[chess.Move]:
    candidate = None
    for move in board.legal_moves:
        if move.from_square == from_sq and move.to_square == to_sq:
            # Prefer queen promotion if multiple options
            if move.promotion == chess.QUEEN:
                return move
            candidate = move
    if candidate is not None:
        return candidate
    # If a promotion is required but not specified, try queen promotion
    try_move = chess.Move(from_sq, to_sq, promotion=chess.QUEEN)
    if try_move in board.legal_moves:
        return try_move
    return None


def draw_board(surface, board: chess.Board, images, cell: int, human_white_bottom: bool, selected: Optional[int], legal_targets: set[int], checkmated_square: Optional[int] = None, exclude: Optional[int] = None) -> None:
    if pygame is None:
        return
    light = (240, 217, 181)
    dark = (181, 136, 99)
    sel_color = (246, 246, 105)
    tgt_color = (106, 162, 75)

    for r in range(8):
        for f in range(8):
            if (r + f) % 2 == 0:
                color = light
            else:
                color = dark
            if human_white_bottom:
                x = f * cell
                y = (7 - r) * cell
            else:
                x = (7 - f) * cell
                y = r * cell
            pygame.draw.rect(surface, color, pygame.Rect(x, y, cell, cell))

    # Draw red gradient first (under pieces) if a checkmated king square is provided
    if checkmated_square is not None:
        rx, ry = xy_from_square(checkmated_square, cell, human_white_bottom)
        s = pygame.Surface((cell, cell), pygame.SRCALPHA)
        center = (cell // 2, cell // 2)
        max_r = max(10, cell // 2)
        for r in range(max_r, 0, -1):
            alpha = int(255 * ((max_r - r) / max_r))
            pygame.draw.circle(s, (255, 0, 0, alpha), center, r)
        surface.blit(s, (rx, ry))

    # Draw pieces
    for sq, piece in board.piece_map().items():
        if exclude is not None and sq == exclude:
            continue
        x, y = xy_from_square(sq, cell, human_white_bottom)
        img = images.get((piece.piece_type, piece.color))
        if img is not None:
            surface.blit(img, (x, y))

    # Draw legal target dots on top (including capturable tiles)
    for tgt in legal_targets:
        tx, ty = xy_from_square(tgt, cell, human_white_bottom)
        center = (tx + cell // 2, ty + cell // 2)
        pygame.draw.circle(surface, tgt_color, center, max(6, cell // 12))

    # Selection highlight on top
    if selected is not None:
        sx, sy = xy_from_square(selected, cell, human_white_bottom)
        pygame.draw.rect(surface, sel_color, pygame.Rect(sx, sy, cell, cell), width=4)


def run_pygame(env: gym.Env, human_is_white: bool, ai_mode: str, window_size: int = 800, choose_side: bool = False, mcts_iterations: int = 400) -> None:
    if pygame is None:
        raise RuntimeError("pygame is not installed. Install it or run without --pygame.")
    pygame.init()
    screen = pygame.display.set_mode((window_size, window_size))
    pygame.display.set_caption("Chess: Human vs AI")
    clock = pygame.time.Clock()

    cell = window_size // 8
    images = load_piece_images(cell)
    human_white_bottom = human_is_white

    # Optional side selection screen
    if choose_side:
        selecting = True
        font = pygame.font.SysFont(None, 48)
        small = pygame.font.SysFont(None, 26)
        while selecting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        pygame.quit()
                        return
                    if event.key in (pygame.K_w, pygame.K_LEFT):
                        human_white_bottom = True
                        selecting = False
                    if event.key in (pygame.K_b, pygame.K_RIGHT):
                        human_white_bottom = False
                        selecting = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mx, _ = pygame.mouse.get_pos()
                    human_white_bottom = mx < window_size // 2
                    selecting = False

            screen.fill((20, 20, 20))
            title = font.render("Choose Side", True, (240, 240, 240))
            opt_left = small.render("White at bottom (W/Left/Click Left)", True, (220, 220, 220))
            opt_right = small.render("Black at bottom (B/Right/Click Right)", True, (220, 220, 220))
            tip = small.render("Esc/Q to quit", True, (180, 180, 180))
            screen.blit(title, (window_size//2 - title.get_width()//2, window_size//2 - 80))
            screen.blit(opt_left, (window_size//2 - opt_left.get_width()//2, window_size//2 - 24))
            screen.blit(opt_right, (window_size//2 - opt_right.get_width()//2, window_size//2 + 8))
            screen.blit(tip, (window_size//2 - tip.get_width()//2, window_size//2 + 48))
            pygame.display.flip()

    while True:
        # Reset
        reset_env(env, None)
        board = env.action_space.board

        selected: Optional[int] = None
        legal_targets: set[int] = set()
        terminated = False
        truncated = False
        reward = 0.0
        checkmated_square: Optional[int] = None
        winner_text = ""
        # Move history and redo stack (list of chess.Move)
        move_history: list[chess.Move] = []
        redo_stack: list[chess.Move] = []

        def update_legal_targets(sel: Optional[int]) -> set[int]:
            if sel is None:
                return set()
            targets = set()
            for move in board.legal_moves:
                if move.from_square == sel:
                    targets.add(move.to_square)
            return targets

        # AI opens if human plays Black (human at bottom False)
        if (not human_white_bottom) and board.turn:
            t0 = time.perf_counter()
            ai_action = choose_ai_action(env, ai_mode, mcts_iterations)
            dt = time.perf_counter() - t0
            try:
                print(f"AI decision time: {dt:.3f}s")
            except Exception:
                pass
            try:
                ai_move_obj = env.action_space._action_to_move(ai_action)
            except Exception:
                ai_move_obj = None
            _, reward, terminated, truncated, _ = env.step(ai_action)
            if ai_move_obj is not None:
                move_history.append(ai_move_obj)
                redo_stack.clear()

        # Initial draw
        draw_board(screen, board, images, cell, human_white_bottom, selected, legal_targets, checkmated_square)
        pygame.display.flip()

        in_game = True
        while in_game:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                elif event.type == pygame.KEYDOWN and (terminated or truncated):
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        pygame.quit()
                        return
                    if event.key in (pygame.K_r, pygame.K_RETURN, pygame.K_SPACE):
                        # Restart same side
                        in_game = False
                        break
                elif event.type == pygame.KEYDOWN:
                    # Undo one move
                    if event.key == pygame.K_LEFT and move_history:
                        last_move = move_history.pop()
                        try:
                            board.pop()
                            redo_stack.append(last_move)
                            terminated = False
                            truncated = False
                            winner_text = ""
                            checkmated_square = None
                            selected = None
                            legal_targets = set()
                        except Exception:
                            pass
                    # Redo one move
                    elif event.key == pygame.K_RIGHT and redo_stack:
                        next_move = redo_stack.pop()
                        try:
                            board.push(next_move)
                            move_history.append(next_move)
                            selected = None
                            legal_targets = set()
                        except Exception:
                            pass
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and not (terminated or truncated):
                    # Human turn only
                    if (board.turn and human_white_bottom) or ((not board.turn) and (not human_white_bottom)):
                        mx, my = pygame.mouse.get_pos()
                        sq = square_from_xy(mx, my, cell, human_white_bottom)
                        if sq is None:
                            selected = None
                            legal_targets = set()
                        else:
                            piece = board.piece_at(sq)
                            if selected is None:
                                if piece is not None and piece.color == board.turn:
                                    selected = sq
                                    legal_targets = update_legal_targets(selected)
                            else:
                                move = find_move_from_squares(board, selected, sq)
                                if move is not None:
                                    # Visualize player's move immediately
                                    action_id = convert_move_to_action_id(env, move)
                                    if action_id is not None:
                                        # Animate player's move
                                        from_sq = move.from_square
                                        to_sq = move.to_square
                                        fx, fy = xy_from_square(from_sq, cell, human_white_bottom)
                                        tx, ty = xy_from_square(to_sq, cell, human_white_bottom)
                                        moving_piece = board.piece_at(from_sq)
                                        img = images.get((moving_piece.piece_type, moving_piece.color)) if moving_piece is not None else None
                                        # Apply board move first so captured piece disappears
                                        _, reward, terminated, truncated, _ = env.step(action_id)
                                        move_history.append(move)
                                        redo_stack.clear()
                                        selected = None
                                        legal_targets = set()
                                        # Animate over 150 ms
                                        duration_ms = 150
                                        start_time = pygame.time.get_ticks()
                                        while True:
                                            now = pygame.time.get_ticks()
                                            t = min(1.0, (now - start_time) / duration_ms)
                                            cx = fx + (tx - fx) * t
                                            cy = fy + (ty - fy) * t
                                            draw_board(screen, board, images, cell, human_white_bottom, selected, legal_targets, exclude=to_sq)
                                            if img is not None:
                                                screen.blit(img, (cx, cy))
                                            pygame.display.flip()
                                            if t >= 1.0:
                                                break
                                            clock.tick(60)
                                        # Then AI responds if game continues
                                        if not terminated and not truncated:
                                            t0 = time.perf_counter()
                                            ai_action = choose_ai_action(env, ai_mode, mcts_iterations)
                                            dt = time.perf_counter() - t0
                                            try:
                                                print(f"AI decision time: {dt:.3f}s")
                                            except Exception:
                                                pass
                                            try:
                                                ai_move_obj = env.action_space._action_to_move(ai_action)
                                            except Exception:
                                                ai_move_obj = None
                                            # Animate AI move
                                            if ai_move_obj is not None:
                                                af, at = ai_move_obj.from_square, ai_move_obj.to_square
                                                afx, afy = xy_from_square(af, cell, human_white_bottom)
                                                atx, aty = xy_from_square(at, cell, human_white_bottom)
                                                moving_piece_ai = board.piece_at(af)
                                                aimg = images.get((moving_piece_ai.piece_type, moving_piece_ai.color)) if moving_piece_ai is not None else None
                                            else:
                                                aimg = None
                                            _, reward, terminated, truncated, _ = env.step(ai_action)
                                            if ai_move_obj is not None:
                                                move_history.append(ai_move_obj)
                                                redo_stack.clear()
                                                # animate over 150ms
                                                duration_ms = 150
                                                start_time = pygame.time.get_ticks()
                                                while True:
                                                    now = pygame.time.get_ticks()
                                                    t = min(1.0, (now - start_time) / duration_ms)
                                                    cx = afx + (atx - afx) * t
                                                    cy = afy + (aty - afy) * t
                                                    draw_board(screen, board, images, cell, human_white_bottom, selected, legal_targets, exclude=at)
                                                    if aimg is not None:
                                                        screen.blit(aimg, (cx, cy))
                                                    pygame.display.flip()
                                                    if t >= 1.0:
                                                        break
                                                    clock.tick(60)
                                    else:
                                        selected = None
                                        legal_targets = set()
                                else:
                                    if piece is not None and piece.color == board.turn:
                                        selected = sq
                                        legal_targets = update_legal_targets(selected)
                                    else:
                                        selected = None
                                        legal_targets = set()

            # Compute end-of-game overlays
            if (terminated or truncated) and not winner_text:
                winner_text = game_result_message(board, reward, terminated, truncated)
                if hasattr(board, "is_checkmate") and board.is_checkmate():
                    king_squares = list(board.pieces(chess.KING, board.turn))
                    if king_squares:
                        checkmated_square = king_squares[0]

            # Draw frame
            draw_board(screen, board, images, cell, human_white_bottom, selected, legal_targets, checkmated_square)

            # End screen overlay
            if terminated or truncated:
                overlay = pygame.Surface((window_size, window_size), pygame.SRCALPHA)
                overlay.fill((0, 0, 0, 120))
                screen.blit(overlay, (0, 0))
                font = pygame.font.SysFont(None, 54)
                small = pygame.font.SysFont(None, 26)
                msg = winner_text if winner_text else "Game Over"
                text = font.render(msg, True, (255, 255, 255))
                info = small.render("Press R/Enter/Space to restart, Esc/Q to quit", True, (230, 230, 230))
                screen.blit(text, (window_size//2 - text.get_width()//2, window_size//2 - 40))
                screen.blit(info, (window_size//2 - info.get_width()//2, window_size//2 + 8))

            pygame.display.flip()
            clock.tick(60)

def main() -> int:
    parser = argparse.ArgumentParser(description="Play Chess: Human vs AI using chess-gym env")
    parser.add_argument("--side", choices=["white", "black", "w", "b"], default="white", help="Choose your side")
    parser.add_argument("--ai", choices=["heuristic", "random", "dqn", "mcts"], default="mcts", help="AI policy (mcts=neural net with MCTS search)")
    parser.add_argument("--fen", type=str, default=None, help="Start position FEN")
    parser.add_argument("--max-steps", type=int, default=512, help="Max plies (half-moves)")
    parser.add_argument("--show-legal", action="store_true", help="Print legal actions each turn")
    parser.add_argument("--use-4672", action="store_true", help="Use 4672 action space instead of default")
    parser.add_argument("--no-4672", dest="use_4672", action="store_false", help="Use non-4672 action space")
    parser.add_argument("--pygame", action="store_true", default=True, help="Use pygame UI instead of CLI input")
    parser.add_argument("--choose-side", action="store_true", help="Show side selection at startup in pygame mode")
    parser.add_argument("--window", type=int, default=800, help="Window size (pixels)")
    parser.add_argument("--mcts-iterations", type=int, default=400, help="Number of MCTS iterations per move (only for --ai mcts)")
    parser.set_defaults(use_4672=True)
    args = parser.parse_args()

    human_is_white = args.side in ("white", "w")

    # If DQN or MCTS is selected, prefer 4672 action space to match checkpoint
    if args.ai in ("dqn", "mcts") and not args.use_4672:
        print(f"[Info] Forcing 4672 action space for {args.ai.upper()} policy to match checkpoint.")
        args.use_4672 = True

    env = create_env(use_4672_action_space=args.use_4672, show_possible_actions=args.show_legal)
    reset_env(env, args.fen)
    board = env.action_space.board

    print("Starting Human vs AI Chess")
    print_board(board)

    terminated = False
    truncated = False
    reward = 0.0
    steps = 0

    if args.pygame:
        run_pygame(env, human_is_white, args.ai, window_size=args.window, choose_side=args.choose_side, mcts_iterations=args.mcts_iterations)
        env.close()
        return 0
    else:
        if not human_is_white and board.turn:  # AI starts if human chose black
            # Measure AI think time
            t0 = time.perf_counter()
            ai_action = choose_ai_action(env, args.ai, args.mcts_iterations)
            dt = time.perf_counter() - t0
            try:
                print(f"AI decision time: {dt:.3f}s")
            except Exception:
                pass
            _, reward, terminated, truncated, _ = env.step(ai_action)
            steps += 1
            print("AI move played.")
            print_board(board)

    while not terminated and not truncated and steps < args.max_steps:
        if args.show_legal:
            print_legal_actions(board)

        try:
            user_input = input("Your move (UCI, SAN, or action id). Type 'help' or 'quit': ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if user_input.lower() in {"quit", "exit"}:
            print("Quitting game.")
            break
        if user_input.lower() == "help":
            print("Enter moves as SAN (e.g., Nf3, O-O), UCI (e2e4, e7e8q), or action id (integer).")
            continue

        human_action = parse_human_input_to_action(env, user_input)
        if human_action is None:
            print("Invalid move or action id. Try again.")
            continue

        _, reward, terminated, truncated, _ = env.step(human_action)
        steps += 1
        print_board(board)
        if terminated or truncated:
            break

        # Measure AI think time
        t0 = time.perf_counter()
        ai_action = choose_ai_action(env, args.ai, args.mcts_iterations)
        dt = time.perf_counter() - t0
        try:
            print(f"AI decision time: {dt:.3f}s")
        except Exception:
            pass
        _, reward, terminated, truncated, _ = env.step(ai_action)
        steps += 1
        print("AI move played.")
        print_board(board)

    msg = game_result_message(board, reward, terminated, truncated)
    if msg:
        print(msg)
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())


