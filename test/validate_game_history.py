#!/usr/bin/env python3
"""
Validate game history file for illegal moves using chess.py
"""
import chess
import re
import sys
from pathlib import Path


def parse_game_history(file_path):
    """Parse the game history file and extract games."""
    games = []
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Split by game markers
    game_blocks = re.split(r'^Game \d+:', content, flags=re.MULTILINE)
    
    for i, block in enumerate(game_blocks[1:], 1):  # Skip first empty block
        lines = block.strip().split('\n')
        if not lines:
            continue
            
        # Parse game header
        header = lines[0]
        fen_line = lines[1] if len(lines) > 1 else ""
        reward_line = lines[2] if len(lines) > 2 else ""
        moves_line = lines[3] if len(lines) > 3 else ""
        
        # Extract FEN
        fen_match = re.search(r'Initial FEN: ([^\|]+)', fen_line)
        if not fen_match:
            continue
        fen = fen_match.group(1).strip()
        
        # Extract moves (everything after "Reward:")
        moves = moves_line.strip().split()
        
        games.append({
            'game_num': i,
            'header': header,
            'fen': fen,
            'moves': moves
        })
    
    return games


def validate_game(game):
    """Validate a single game and return list of illegal moves."""
    illegal_moves = []
    
    try:
        # Create board from FEN
        board = chess.Board(fen=game['fen'])
        
        # Play each move
        for move_idx, move_str in enumerate(game['moves']):
            # Remove check/checkmate markers
            move_str_clean = move_str.rstrip('#+')
            
            try:
                # Try to parse the move
                move = board.parse_san(move_str_clean)
                
                # Check if move is legal
                if move not in board.legal_moves:
                    # Get some legal moves for context
                    legal_moves_sample = list(board.legal_moves)[:5]
                    legal_san_sample = [board.san(m) for m in legal_moves_sample]
                    illegal_moves.append({
                        'move_num': move_idx + 1,
                        'move': move_str,
                        'reason': f'Move is not legal. Sample legal moves: {legal_san_sample}',
                        'board_fen': board.fen()
                    })
                else:
                    # Make the move
                    board.push(move)
                    
            except ValueError as e:
                # parse_san raises ValueError for invalid/ambiguous moves
                # Try to find if there are similar legal moves
                legal_moves = list(board.legal_moves)
                legal_san = [board.san(m) for m in legal_moves]
                # Find moves that contain the same destination square
                similar_moves = [m for m in legal_san if move_str_clean[-2:] in m]
                
                reason = f'Cannot parse move: {str(e)}'
                if similar_moves:
                    reason += f'. Similar legal moves: {similar_moves[:5]}'
                
                illegal_moves.append({
                    'move_num': move_idx + 1,
                    'move': move_str,
                    'reason': reason,
                    'board_fen': board.fen()
                })
                
    except Exception as e:
        illegal_moves.append({
            'move_num': 0,
            'move': 'INITIAL',
            'reason': f'Error setting up board: {str(e)}',
            'board_fen': game['fen']
        })
    
    return illegal_moves


def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_game_history.py <game_history_file>")
        sys.exit(1)
    
    file_path = Path(sys.argv[1])
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    
    print(f"Validating games in: {file_path}")
    print("=" * 80)
    
    games = parse_game_history(file_path)
    print(f"Found {len(games)} games\n")
    
    total_illegal = 0
    games_with_illegal = []
    
    for game in games:
        illegal_moves = validate_game(game)
        
        if illegal_moves:
            total_illegal += len(illegal_moves)
            games_with_illegal.append({
                'game': game,
                'illegal_moves': illegal_moves
            })
    
    # Report results
    if total_illegal == 0:
        print("✓ All moves are legal! No illegal moves found.")
    else:
        print(f"✗ Found {total_illegal} illegal move(s) in {len(games_with_illegal)} game(s):\n")
        
        for item in games_with_illegal:
            game = item['game']
            illegal_moves = item['illegal_moves']
            
            print(f"Game {game['game_num']}: {game['header']}")
            print(f"  Initial FEN: {game['fen']}")
            print(f"  Illegal moves:")
            
            for illegal in illegal_moves:
                print(f"    Move {illegal['move_num']}: {illegal['move']}")
                print(f"      Reason: {illegal['reason']}")
                if illegal['move_num'] > 0:
                    print(f"      Board FEN: {illegal['board_fen']}")
            print()
    
    print("=" * 80)
    print(f"Summary: {len(games_with_illegal)}/{len(games)} games have illegal moves")
    print(f"Total illegal moves: {total_illegal}")


if __name__ == '__main__':
    main()

