import sys
import os
sys.path.append(os.path.abspath('.'))

import math
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Assuming gymnasium and chess_gym are correctly installed/imported
import gymnasium as gym
import chess_gym
import chess

# --- Placeholder Neural Network ---
# This network takes the board vector and outputs policy probabilities for legal moves and a value estimate.
class PlaceholderNetwork(nn.Module):
    def __init__(self, board_vector_size=8*8*13, hidden_size=64):
        super().__init__()
        # Simple network architecture
        self.fc1 = nn.Linear(board_vector_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # Policy head: Outputs logits for actions. Size determined dynamically later.
        self.policy_head = nn.Linear(hidden_size, 1) # Placeholder size, will be ignored
        # Value head: Outputs a single value estimate
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, board_vector, legal_moves):
        """
        Args:
            board_vector (np.ndarray): The flattened board state vector.
            legal_moves (list[chess.Move]): List of legal moves in the current state.

        Returns:
            tuple[dict[chess.Move, float], float]:
                - Policy probabilities mapping legal moves to their probabilities.
                - Value estimate for the current state (-1 to 1).
        """
        if isinstance(board_vector, np.ndarray):
            # Ensure dtype is float32 for PyTorch
            board_vector = torch.tensor(board_vector.flatten(), dtype=torch.float32)
        elif not isinstance(board_vector, torch.Tensor):
             raise TypeError(f"board_vector must be np.ndarray or torch.Tensor, got {type(board_vector)}")
        
        if board_vector.dtype != torch.float32:
             board_vector = board_vector.float() # Ensure float32

        # Ensure it's a batch of 1 if needed (though here we process one state)
        if board_vector.ndim == 1:
            board_vector = board_vector.unsqueeze(0)

        # Simple forward pass
        x = F.relu(self.fc1(board_vector))
        x = F.relu(self.fc2(x))

        # --- Value Head ---
        value = torch.tanh(self.value_head(x)) # Squash value to [-1, 1]

        # --- Policy Head (Simplified Placeholder) ---
        # For this basic example, just output uniform probabilities over legal moves.
        # A real network would use its learned policy.
        num_legal_moves = len(legal_moves)
        if num_legal_moves > 0:
            # Even if network produced logits, we'd softmax them.
            # Here, we just assign uniform probability.
            uniform_prob = 1.0 / num_legal_moves
            policy_probs = {move: uniform_prob for move in legal_moves}
        else:
            policy_probs = {} # No legal moves

        return policy_probs, value.item() # Return dict and scalar value

# --- MCTS Node ---
class MCTSNode:
    def __init__(self, state: chess.Board, parent=None, prior_p=0.0, move_leading_here=None):
        self.state = state  # chess.Board object
        self.parent = parent
        self.move_leading_here = move_leading_here # Move taken by parent to reach this node
        self.children: dict[chess.Move, MCTSNode] = {}  # map from move to MCTSNode
        self.untried_actions = list(state.legal_moves)
        self.N = 0  # visit count
        self.W = 0  # total value of this node (sum of simulation results)
        self.P = prior_p # Prior probability of selecting the move leading to this node

    def is_fully_expanded(self) -> bool:
        return not self.untried_actions

    def is_terminal(self) -> bool:
        return self.state.is_game_over()

    def get_value(self) -> float:
        """Average value W/N."""
        if self.N == 0:
            return 0
        # The value W is stored from the perspective of the player *who arrived* at this node.
        return self.W / self.N

    def select_child_uct(self, C_puct: float) -> 'MCTSNode':
        """Selects child node using the UCT formula."""
        best_score = -float('inf')
        best_child = None

        for move, child in self.children.items():
            # UCT = Q(s,a) + C * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
            # Q(s,a) is the value from the child's perspective (-W/N of child, as W is stored for parent's arrival)
            q_value = -child.get_value() # Negate because child.W is relative to the player *at* the child node
            uct_score = q_value + C_puct * child.P * (math.sqrt(self.N) / (1 + child.N))

            if uct_score > best_score:
                best_score = uct_score
                best_child = child

        if best_child is None:
             # This can happen if N=0 for parent but children exist (shouldn't ideally)
             # Or if there are no children yet (select should not be called)
             # Fallback or raise error
             if self.children:
                 # Fallback to random child if scores are bad/equal? Or most visited?
                 # For simplicity, let's just pick one if available
                 # print("Warning: UCT selection failed to find best child, picking first available.")
                 return list(self.children.values())[0]
             else:
                  raise RuntimeError("Cannot select child from a node with no children.")


        return best_child


# --- MCTS Algorithm ---
class MCTS:
    def __init__(self, network: PlaceholderNetwork, C_puct=1.41):
        self.network = network
        self.C_puct = C_puct

    def _simulate_random_playout(self, board: chess.Board) -> float:
        """Performs a random simulation from the given board state."""
        # Important: Use a copy of the board for simulation!
        sim_board = board.copy()
        while not sim_board.is_game_over():
            legal_moves = list(sim_board.legal_moves)
            if not legal_moves: # Should be caught by is_game_over, but safety check
                break
            random_move = random.choice(legal_moves)
            sim_board.push(random_move)

        result = sim_board.result()
        if result == "1-0":
            return 1.0
        elif result == "0-1":
            return -1.0
        else: # Draw
            return 0.0

    def search(self, root_node: MCTSNode, iterations: int):
        """Runs the MCTS search for a fixed number of iterations."""
        start_time = time.time()
        for i in range(iterations):
            node = root_node
            search_path = [node] # Keep track of path for backpropagation

            # 1. Selection: Traverse the tree until a leaf node is found
            while node.is_fully_expanded() and not node.is_terminal():
                node = node.select_child_uct(self.C_puct)
                search_path.append(node)

            # 2. Expansion: If the node is not terminal, expand one child
            if not node.is_terminal():
                # Get policy and value from network (even if expanding)
                # Need the board vector representation
                board_vector = node.state.get_board_vector() # Assuming FullyTrackedBoard method exists
                legal_moves = list(node.state.legal_moves) # Get fresh list
                policy_probs, leaf_value = self.network(board_vector, legal_moves)

                # Expand one untried action
                if node.untried_actions: # Should be true if not terminal and not fully expanded
                    move = node.untried_actions.pop(random.randrange(len(node.untried_actions)))
                    next_state = node.state.copy()
                    next_state.push(move)
                    prior_p = policy_probs.get(move, 0.0) # Get prior from network output
                    # Handle case where policy_probs might be empty if no legal moves
                    if not legal_moves and not policy_probs:
                        prior_p = 0.0
                    elif not policy_probs and legal_moves: # Should not happen with current placeholder
                        print("Warning: Policy probs empty but legal moves exist.")
                        prior_p = 1.0 / len(legal_moves) if legal_moves else 0.0


                    child_node = MCTSNode(next_state, parent=node, prior_p=prior_p, move_leading_here=move)
                    node.children[move] = child_node
                    node = child_node # Move down to the new node
                    search_path.append(node)

                    # The value used for backpropagation is the network's evaluation of the expanded node
                    value = leaf_value
                else:
                     # This case means node is not terminal but has no untried actions,
                     # which contradicts the while loop condition unless legal_moves is empty.
                     # If legal_moves is empty, it should be terminal.
                     # If it was fully expanded, the loop condition failed.
                     # If somehow reached here, use the network's value directly.
                     # print(f"Warning: Reached expansion phase but node has no untried actions. State: {node.state.fen()}")
                     value = leaf_value # Use the network's prediction for this leaf

            else: # Node is terminal
                # For terminal node, the value is the actual game result
                result_str = node.state.result()
                if result_str == "1-0": value = 1.0
                elif result_str == "0-1": value = -1.0
                else: value = 0.0
                 # Value needs to be from the perspective of the player *whose turn it was* at the parent
                 # Since this is the terminal node, the game ended *before* the current player (at this node) could move.
                 # So the result is directly applicable to the player who just moved (the parent).

            # 3. Simulation (Optional with Network):
            # In AlphaZero-style MCTS, the network's value estimate often replaces the random rollout.
            # We already got `value` from the network (or terminal state).
            # If you wanted a classic MCTS, you'd uncomment the simulation:
            # value = self._simulate_random_playout(node.state)

            # 4. Backpropagation: Update visit counts and values along the path
            # The value should be relative to the player whose turn it is *at the parent* node.
            for node_in_path in reversed(search_path):
                node_in_path.N += 1
                # If the player at node_in_path.parent is the one whose turn it is at node_in_path,
                # then the value should be used directly. Otherwise, it should be negated.
                # node.state.turn is the player *to move* at that node.
                # So, if node_in_path.state.turn != node_in_path.parent.state.turn (if parent exists),
                # the value is from the perspective of the player TO MOVE at node_in_path.
                # We want to update W from the perspective of the player who MOVED TO node_in_path.
                # Therefore, we negate the value if the turn flips between parent and child.
                current_node_perspective_value = value
                if node_in_path.parent:
                    # If the turn at the node we're updating is DIFFERENT from the turn
                    # that the value 'value' represents (which is the player whose turn it is at the *leaf*), negate.
                    # Let's simplify: W always stores value relative to the player whose turn it is at the PARENT.
                    # The 'value' calculated is from the perspective of the player whose turn it is AT THE LEAF NODE.
                    # So, we negate 'value' if the turns don't match.
                    turn_at_leaf = node.state.turn # Player whose turn it is at the node where value was determined
                    turn_at_current_update_node = node_in_path.state.turn

                    # If the player to move at the node we're currently updating
                    # is NOT the same as the player to move at the leaf where value was computed,
                    # then the value needs to be flipped for the W update.
                    # Example: Leaf has White to move, value is +0.8 (good for White).
                    # Backpropagate to parent (Black to move). W should increase by -0.8.
                    # Backpropagate to grandparent (White to move). W should increase by +0.8.
                    if turn_at_current_update_node != turn_at_leaf:
                         current_node_perspective_value = -value # Flip value for parent perspective


                node_in_path.W += current_node_perspective_value
                # Make sure value is flipped for the next level up if turns alternate
                # The 'value' variable itself retains its perspective (player at leaf)
                # The logic above determines how it contributes to W at each step.


        end_time = time.time()
        print(f"Search finished {iterations} iterations in {end_time - start_time:.2f} seconds.")

    def get_best_move(self, root_node: MCTSNode, temperature=0.0) -> chess.Move:
        """
        Selects the best move from the root node after search.
        - temperature = 0: Choose the most visited node.
        - temperature > 0: Sample from visit counts distribution.
        """
        if not root_node.children:
            raise RuntimeError("MCTS search hasn't been run or root has no children.")

        if temperature == 0:
            # Deterministic: Choose the most visited child
            max_visits = -1
            best_move = None
            for move, child in root_node.children.items():
                if child.N > max_visits:
                    max_visits = child.N
                    best_move = move
            if best_move is None: # Should not happen if children exist
                 best_move = list(root_node.children.keys())[0]
            return best_move
        else:
            # Probabilistic: Sample based on visit counts
            moves = list(root_node.children.keys())
            visit_counts = np.array([root_node.children[m].N for m in moves], dtype=float)

            # Apply temperature
            visit_counts_temp = visit_counts**(1.0 / temperature)
            probabilities = visit_counts_temp / np.sum(visit_counts_temp)

            chosen_move = np.random.choice(moves, p=probabilities)
            return chosen_move


# --- Main Execution ---
if __name__ == "__main__":
    # Request the 'vector' observation mode
    env = gym.make("Chess-v0", observation_mode='vector')
    # Important: Reset returns observation, info tuple now in Gymnasium
    observation, info = env.reset()
    board = env.action_space.board # Get the board from the action space

    # Ensure board has the get_board_vector method
    if not hasattr(board, 'get_board_vector'):
         raise AttributeError("The board object from the environment does not have the 'get_board_vector' method. "
                           "Ensure you are using the FullyTrackedBoard from chess_custom.py.")

    # Initialize network and MCTS
    network = PlaceholderNetwork()
    mcts = MCTS(network, C_puct=1.41)

    # Create root node
    root_node = MCTSNode(board.copy()) # Use a copy for the root's state

    # Run the search
    print("Starting MCTS search...")
    # Increase iterations for better results, e.g., 100, 500, or more
    num_iterations = 50
    mcts.search(root_node, iterations=num_iterations)

    # Get the best move
    # Use temperature=0 for deterministic play (choose most visited)
    best_move = mcts.get_best_move(root_node, temperature=0.0)

    print(f"\nInitial board state:\n{board}")
    print(f"\nMCTS recommended move: {board.san(best_move)} ({best_move.uci()})")

    # Optional: Print stats of top moves
    print("\nStats for moves from root:")
    sorted_children = sorted(root_node.children.items(), key=lambda item: item[1].N, reverse=True)
    for move, child_node in sorted_children[:10]: # Print top 10
        print(f"- {board.san(move)}: Visits={child_node.N}, Value={-child_node.get_value():.3f} (value for current player), Prior={child_node.P:.3f}")

    env.close()