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

from collections import defaultdict # Re-add if needed elsewhere, maybe not

# Assuming gymnasium and chess_gym are correctly installed/imported
import gymnasium as gym
import chess_gym
import chess
from utils.policy_human import sample_action # Import the heuristic policy

# --- Placeholder Neural Network ---
# This network takes the board vector and outputs policy probabilities for legal moves and a value estimate.
class PlaceholderNetwork(nn.Module):
    def __init__(self, board_vector_size=8*8*13, hidden_size=64, action_space_size=1700):
        super().__init__()
        # Simple network architecture
        self.fc1 = nn.Linear(board_vector_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # Policy head: Outputs logits for the full action space
        self.policy_head = nn.Linear(hidden_size, action_space_size)
        # Value head: Outputs a single value estimate
        self.value_head = nn.Linear(hidden_size, 1)
        self.action_space_size = action_space_size

    def forward(self, board_vector):
        """
        Args:
            board_vector (np.ndarray): The flattened board state vector.

        Returns:
            tuple[torch.Tensor, float]:
                - Policy logits tensor (shape [action_space_size]).
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

        # --- Policy Head --- 
        # Output raw logits for the entire action space
        policy_logits = self.policy_head(x).squeeze(0) # Remove batch dimension

        return policy_logits, value.item() # Return logits tensor and scalar value

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
    def __init__(self, network: PlaceholderNetwork, player_color: chess.Color = chess.WHITE, C_puct=1.41):
        self.network = network
        self.player_color = player_color
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
            while not node.is_terminal():
                # --- Opponent Move Selection --- 
                if node.state.turn != self.player_color:
                    # Use heuristic policy for the opponent
                    # sample_action returns (action_array, policy_id, policy_title)
                    # We need the move object. Assume env.action_space exists and has the board.
                    # If MCTS is used outside env context, need direct access to action_space/board.
                    # For now, assume root_node.state gives us access indirectly or directly.
                    # We need the MoveSpace to convert array back to move if necessary, 
                    # but sample_action ideally works with the board directly. Re-check sample_action.
                    # Let's assume sample_action can take the board directly and return a move.
                    # Need to instantiate action space temporarily if needed by sample_action? 
                    # A cleaner way: modify sample_action to optionally return the move object.
                    # For now, use the current sample_action which returns array, ID, title.
                    # We need to get the move object. Let's call _move_to_action on the root action space? Risky.
                    # *** TEMPORARY WORKAROUND: Find the move from legal_moves that matches the heuristic choice ***
                    # *** A cleaner solution would involve modifying sample_action or having it use IDs ***
                    opponent_move = None
                    try: 
                         # We need an action space associated with the current node state
                         temp_action_space = chess_gym.MoveSpace(node.state)
                         # Get the heuristic action array, using return_id=False (the default)
                         # avoid_attacks=True is the default and likely desired opponent behavior
                         action_np, policy_id, policy_title = sample_action(temp_action_space)
                         # Convert numpy array back to move (using the node's MoveSpace helper)
                         # This relies on MoveSpace._action_to_move handling the array format
                         opponent_move = temp_action_space._action_to_move(action_np)
                         if opponent_move not in node.state.legal_moves:
                             # print(f"Warning: Heuristic opponent move {opponent_move} not legal in state {node.state.fen()}. Falling back.")
                             opponent_move = None # Fallback if heuristic fails
                    except Exception as e:
                        # print(f"Warning: Error getting opponent heuristic move: {e}. Falling back.")
                        opponent_move = None
                        
                    if opponent_move is None:
                        # Fallback: if heuristic failed or produced illegal move, pick random
                        if not node.state.legal_moves:
                             break # Terminal if no legal moves
                        opponent_move = random.choice(list(node.state.legal_moves))

                    # Find or create the child node for the opponent's chosen move
                    if opponent_move in node.children:
                        node = node.children[opponent_move]
                    else:
                        # Create child node for opponent move (no network eval needed here)
                        next_state = node.state.copy()
                        next_state.push(opponent_move)
                        # Assign a dummy prior? Or average? For now, 0. Maybe 1/N? Let's use 0.
                        child_node = MCTSNode(next_state, parent=node, prior_p=0.0, move_leading_here=opponent_move)
                        node.children[opponent_move] = child_node
                        node = child_node
                    search_path.append(node)
                    # After opponent moves, it should be player's turn or game over
                    if node.is_terminal(): break # Stop if game ended after opponent move
                    # Now it must be the player's turn, continue selection logic below
                # --- End Opponent Move Selection ---

                # --- Player Move Selection / Expansion Decision ---
                if node.is_fully_expanded():
                     node = node.select_child_uct(self.C_puct)
                     search_path.append(node)
                else:
                    # Reached a leaf node for the player's turn, break selection to expand
                    break

            # --- Expansion & Value Assignment --- 
            value = 0.0 # Default value
            if not node.is_terminal():
                # Expand only if it's the controlled player's turn
                if node.state.turn == self.player_color:
                    board_vector = node.state.get_board_vector()
                    policy_logits, leaf_value = self.network(board_vector)
                    value = leaf_value # Use network's value for player's turn leaf

                    # Apply mask for legal moves
                    legal_action_ids = []
                    legal_moves_map = {}
                    # Use the board's method to get action IDs for legal moves
                    # Assumes FullyTrackedBoard and its methods
                    for move in node.state.legal_moves:
                        action_id = node.state.move_to_action_id(move)
                        if action_id is not None: # Ensure ID calculation is valid
                            # Action IDs are 1-based in analysis, adjust to 0-based for tensor indexing
                            action_id_0based = action_id - 1
                            if 0 <= action_id_0based < self.network.action_space_size:
                                legal_action_ids.append(action_id_0based)
                                legal_moves_map[action_id_0based] = move
                        # else: logging within move_to_action_id handles warnings

                    if not legal_action_ids:
                        # print("Warning: No legal action IDs found during expansion.")
                        # If no legal moves/IDs, treat as terminal? Or assign default low value? 
                        # For now, value remains leaf_value but expansion won't happen.
                        pass # Cannot expand
                    else:
                        masked_logits = policy_logits[legal_action_ids]
                        # Boltzmann distribution (Softmax with temperature)
                        temperature = 1.0 # Adjust temperature as needed
                        if temperature == 0:
                             probs = torch.zeros_like(masked_logits)
                             probs[torch.argmax(masked_logits)] = 1.0
                        else:
                            probs = F.softmax(masked_logits / temperature, dim=0)
                        
                        # Ensure probs sum to 1 (handle potential NaN issues if logits were extreme)
                        if torch.isnan(probs).any():
                            # Fallback to uniform if softmax results in NaN
                            # print("Warning: Softmax resulted in NaN, falling back to uniform distribution.")
                            probs = torch.ones_like(masked_logits) / len(masked_logits)

                        # Sample action ID based on Boltzmann probabilities
                        # Need to handle potential empty probs tensor if legal_action_ids was empty
                        if probs.numel() > 0:
                             sampled_index = torch.multinomial(probs, 1).item()
                             sampled_action_id_0based = legal_action_ids[sampled_index]
                             sampled_move = legal_moves_map[sampled_action_id_0based]

                            # Expand the sampled move
                             if sampled_move in node.children:
                                 # print(f"Warning: Sampled move {sampled_move} already in children during expansion.")
                                 node = node.children[sampled_move] # Move to existing child
                             else:
                                 # Get the prior probability for the chosen action
                                 prior_p = probs[sampled_index].item()
                                 
                                 next_state = node.state.copy()
                                 next_state.push(sampled_move)
                                 
                                 child_node = MCTSNode(next_state, parent=node, prior_p=prior_p, move_leading_here=sampled_move)
                                 node.children[sampled_move] = child_node
                                 node = child_node # Move down to the new node
                                 
                            # Mark original node's corresponding action as 'tried' (conceptually)
                            # The untried_actions logic isn't strictly needed with network policy sampling
                            # if sampled_move in node.parent.untried_actions: # Update parent? No, update node
                            #    node.parent.untried_actions.remove(sampled_move) # This node was the parent before expansion
                            # Instead of managing untried_actions, we rely on sampling from legal moves

                             search_path.append(node)
                             # Value for backpropagation is the value estimated for the *expanded* node
                             # (leaf_value was estimate of the node *before* expansion)
                             # Let's stick with leaf_value for simplicity, as in AlphaZero
                             # value = leaf_value # Already set above
                        else:
                            # print("Warning: No valid probabilities to sample from during expansion.")
                            pass # Cannot expand if probs tensor is empty

                # else: If it's opponent's turn at the leaf, value determined by terminal state check below

            # --- Terminal Node Value --- 
            if node.is_terminal():
                result_str = node.state.result()
                if result_str == "1-0": value = 1.0
                elif result_str == "0-1": value = -1.0
                else: value = 0.0

            # 3. Simulation (Skipped - using network value 'leaf_value')

            # 4. Backpropagation
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
    mcts = MCTS(network, player_color=chess.WHITE, C_puct=1.41)

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