import sys
import os
sys.path.append(os.path.abspath('.'))

# Assuming gymnasium and chess_gym are correctly installed/imported
import gymnasium as gym
import chess_gym
import chess

# Import MCTS components from modules
from network import ChessNetwork
from mcts_node import MCTSNode
from mcts_algorithm import MCTS


if __name__ == "__main__":
    # Request the 'vector' observation mode
    env = gym.make("Chess-v0", observation_mode='vector') #, render_mode='human') # Keep env for interaction if needed later
    # Important: Reset returns observation, info tuple now in Gymnasium
    observation, info = env.reset()
    board = env.action_space.board # Get the board from the action space

    # Ensure board has the get_board_vector method
    if not hasattr(board, 'get_board_vector'):
         raise AttributeError("The board object from the environment does not have the 'get_board_vector' method. "
                           "Ensure you are using the FullyTrackedBoard from chess_custom.py.")

    # Initialize network and MCTS
    network = ChessNetwork()
    mcts = MCTS(network, player_color=chess.WHITE, C_puct=1.41) # No env argument needed now

    # Create root node
    root_node = MCTSNode(board.copy()) # Use a copy for the root's state

    # Run the search
    print("Starting MCTS search...")
    # Increase iterations for better results, e.g., 100, 500, or more
    num_iterations = 50 # Set iterations back
    mcts.search(root_node, iterations=num_iterations)

    # Get the best move
    # Use temperature=0 for deterministic play (choose most visited)
    best_move = mcts.get_best_move(root_node, temperature=0.0)

    print(f"\nInitial board state:\n{board}")
    if best_move:
        print(f"\nMCTS recommended move: {board.san(best_move)} ({best_move.uci()})")
    else:
        print("\nMCTS could not recommend a move (possibly no legal moves or search issue).")

    # Optional: Print stats of top moves
    print("\nStats for moves from root:")
    # Sort based on visited children
    children_to_print = {m: c for m, c in root_node.children.items() if c.N > 0}
    sorted_children = sorted(children_to_print.items(), key=lambda item: item[1].N, reverse=True)

    if not sorted_children and root_node.children: # If no children were visited, show unvisited ones
        print("(Showing unvisited children as no nodes were explored)")
        sorted_children = list(root_node.children.items())[:10]

    for move, child_node in sorted_children[:10]: # Print top 10 visited (or first 10 if none visited)
        player_value = -child_node.get_value() # Value for the player whose turn it is at the root
        print(f"- {board.san(move)}: Visits={child_node.N}, Value={player_value:.3f} (value for current player), Prior={child_node.P:.3f}")

    env.close()