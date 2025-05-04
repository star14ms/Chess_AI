import random
import torch
import torch.nn.functional as F
from rich.progress import Progress
import numpy as np
from typing import Tuple, List

import chess

import sys
import os
# Ensure utils and chess_gym can be found
# Get the absolute path of the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

import chess_gym
from chess_gym.envs import ChessEnv
from utils.policy_human import sample_action
from network import ChessNetwork
from mcts_node import MCTSNode

# --- MCTS Algorithm ---
class MCTS:
    def __init__(self, 
                 network: ChessNetwork, 
                 device: torch.device | str, 
                 env: ChessEnv, # Use the env instance directly
                 player_color: chess.Color = chess.WHITE, # Assume MCTS always plans for this color
                 C_puct: float = 1.41, 
                 dirichlet_alpha: float = 0.3, # Alpha for noise
                 dirichlet_epsilon: float = 0.25): # Weight for noise
        self.network = network
        self.device = torch.device(device)
        self.env = env
        self.player_color = player_color # The color MCTS plans for (e.g., WHITE)
        self.C_puct = C_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.network.eval() # Ensure network is in eval mode for MCTS

    # --- MCTS Phases --- 

    def _select(self, root_node: MCTSNode) -> Tuple[MCTSNode, List[MCTSNode]]:
        """Phase 1: Selects a leaf node using UCT."""
        node = root_node
        search_path = [node]

        while not node.is_terminal():
            if not node.is_expanded:
                break
            elif not node.children:
                 break
            else:
                # Note: MCTSNode.select_child_uct needs access to C_puct
                node = node.select_child_uct(self.C_puct)
                search_path.append(node)

        leaf_node = node
        return leaf_node, search_path

    def _expand_and_evaluate(self, leaf_node: MCTSNode, progress: Progress | None = None) -> float:
        """Phase 2: Expands leaf node, creates children, returns evaluated value."""
        value = 0.0
        if not leaf_node.is_expanded:
            leaf_board = leaf_node.get_board()

            if leaf_board.is_game_over(claim_draw=True):
                # --- Handle Terminal Node --- 
                result_str = leaf_board.result(claim_draw=True)
                if result_str == "1-0": value = 1.0
                elif result_str == "0-1": value = -1.0
                else: value = 0.0 # Draw
                # Mark terminal node as 'expanded' conceptually
                leaf_node.is_expanded = True 
            else:
                # --- Non-Terminal: Expand using Network --- 
                # Get Network Prediction
                obs_vector = leaf_board.get_board_vector()
                obs_tensor = torch.tensor(obs_vector, dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    policy_logits, value = self.network(obs_tensor)
                policy_logits = policy_logits.squeeze(0)
                value = value.item()

                # Calculate Softmax Probabilities
                full_probs = F.softmax(policy_logits, dim=0)
                use_uniform_fallback = False
                uniform_prob = 0.0
                if torch.isnan(full_probs).any():
                    num_legal_moves = len(list(leaf_board.legal_moves))
                    uniform_prob = 1.0 / num_legal_moves if num_legal_moves > 0 else 0.0
                    use_uniform_fallback = True
                    # Reset value if network output unreliable
                    value = 0.0 

                # Add Dirichlet Noise (only if root node)
                probs_to_use = full_probs
                if leaf_node.parent is None and not use_uniform_fallback:
                    legal_moves = list(leaf_board.legal_moves)
                    num_legal = len(legal_moves)
                    # Noise only makes sense for >1 move
                    if num_legal > 1: 
                        legal_indices = []
                        legal_probs = []
                        for move in legal_moves:
                            action_id = leaf_board.move_to_action_id(move)
                            if action_id is not None:
                                idx = action_id - 1
                                if 0 <= idx < full_probs.shape[0]:
                                    legal_indices.append(idx)
                                    legal_probs.append(full_probs[idx].item())
                        # else: Ignore moves with invalid action IDs for noise calculation
                        
                        if legal_indices:
                            noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal_indices))
                            noise_t = torch.tensor(noise, device=self.device, dtype=torch.float32)
                            legal_probs_t = torch.tensor(legal_probs, device=self.device, dtype=torch.float32)
                            combined = (1 - self.dirichlet_epsilon) * legal_probs_t + self.dirichlet_epsilon * noise_t
                            probs_to_use = full_probs.clone()
                            # Ensure indices are in a format suitable for indexing (e.g., list or tensor)
                            if isinstance(legal_indices, list):
                                indices_t = torch.tensor(legal_indices, dtype=torch.long, device=self.device)
                            else:
                                # Assume it's already a tensor
                                indices_t = legal_indices 
                            probs_to_use[indices_t] = combined
                            # Optional: Renormalize probs_to_use if needed

                # --- Expand Children using env.step ---
                # Prepare subtask for move expansion if progress bar is active
                expansion_task_id = None
                legal_moves_list = list(leaf_board.legal_moves)
                total_moves = len(legal_moves_list)
                if progress is not None and total_moves > 0:
                    expansion_task_id = progress.add_task(f"  ├── Expanding Node (0/{total_moves})", total=total_moves, transient=True)

                for i, move in enumerate(legal_moves_list):
                    if move not in leaf_node.children:
                        # Reset the shared env to the parent node's state
                        # Using leaf_node.fen and leaf_node.piece_tracker ensures consistency
                        self.env.board.set_fen(leaf_node.board.fen(), leaf_node.board.piece_tracker)
                        action_id = leaf_board.move_to_action_id(move)
                        if action_id is None: continue

                        # Step 1: Apply White's move (action_id)
                        _, _, _, _, _ = self.env.step(action_id)

                        # Step 2: Sample and apply Black's immediate response
                        opponent_board = self.env.board
                        if not opponent_board.is_game_over(claim_draw=True):
                            opponent_action_id = sample_action(self.env.board, return_id=True)
                            if opponent_action_id is not None:
                                    _, _, _, _, _ = self.env.step(opponent_action_id)

                        # Prior probability is based on White's initial move choice
                        prior_p = 0.0
                        # Recalculate 0-based for safety
                        action_id_0based = action_id - 1
                        if 0 <= action_id_0based < probs_to_use.shape[0]:
                            prior_p = uniform_prob if use_uniform_fallback else probs_to_use[action_id_0based].item()
                        # else: prior_p remains 0.0

                        child_node = MCTSNode(
                            board=self.env.board,
                            parent=leaf_node,
                            prior_p=prior_p,
                            move_leading_here=move
                        )
                        leaf_node.children[move] = child_node

                    # Update progress after processing a move
                    if progress is not None and expansion_task_id is not None:
                        progress.update(expansion_task_id, advance=1, description=f"  ├── Expanding Node ({i+1}/{total_moves})")

                # Stop the expansion subtask after the loop
                if progress is not None and expansion_task_id is not None:
                    progress.update(expansion_task_id, visible=False)

                leaf_node.is_expanded = True
        
        # The backpropagation step needs *a* value from the leaf of the simulation path.
        return value

    def _backpropagate(self, search_path: List[MCTSNode], value: float, leaf_node_turn: chess.Color):
        """Phase 4: Backpropagates value up the search path."""
        # Value is from the perspective of the player whose turn it is at the leaf_node
        for node in reversed(search_path):
            # If the turn at the current node in the path is DIFFERENT from the turn
            # at the leaf node where the value was determined, flip the value sign.
            node_turn = node.get_board().turn
            current_node_perspective_value = value if node_turn == leaf_node_turn else -value
            
            node.N += 1
            node.W += current_node_perspective_value

    # --- Main Search Function --- 
    def search(self, root_node: MCTSNode, iterations: int,
               progress: Progress | None = None):
        
        root_fen = self.env.board.fen()
        root_piece_tracker = self.env.board.piece_tracker

        mcts_task_id = None
        if progress is not None:
            mcts_task_id = progress.add_task(f"  ├─ MCTS Sims (0/{iterations})", total=iterations, transient=True, start=True)

        for i in range(iterations):
            # Phase 1: Selection
            leaf_node, search_path = self._select(root_node)

            # Phase 2: Expansion & Evaluation
            value = self._expand_and_evaluate(leaf_node, progress)
            
            # Phase 4: Backpropagation
            # Get turn at the leaf for perspective
            leaf_node_turn = leaf_node.get_board().turn 
            self._backpropagate(search_path, value, leaf_node_turn)

            if mcts_task_id is not None and progress is not None:
                progress.update(mcts_task_id, advance=1, description=f"  ├─ MCTS Sims ({i+1}/{iterations})")

        if mcts_task_id is not None and progress is not None:
            progress.update(mcts_task_id, visible=False)
            
        self.env.board.set_fen(root_fen, root_piece_tracker)  

    def get_best_move(self, root_node: MCTSNode, temperature=0.0) -> chess.Move | None:
        """
        Selects the best move from the root node after search.
        - temperature = 0: Choose the most visited node.
        - temperature > 0: Sample from visit counts distribution.
        """
        if not root_node.children:
            # Handle case with no children explored
            legal = list(root_node.board.legal_moves)
            return random.choice(legal) if legal else None 

        children_to_consider = {m: c for m, c in root_node.children.items() if c.N > 0}
        if not children_to_consider:
            # Handle case where only opponent moves were explored shallowly or no nodes visited
            # Fallback: Choose randomly among unexplored children if they exist
            unexplored_children = [m for m, c in root_node.children.items() if c.N == 0]
            if unexplored_children:
                return random.choice(unexplored_children)
            else:
                legal = list(root_node.board.legal_moves)
                return random.choice(legal) if legal else None

        if temperature == 0:
            max_visits = -1
            best_move = None
            for move, child in children_to_consider.items():
                if child.N > max_visits:
                    max_visits = child.N
                    best_move = move
            # Should not happen if children_to_consider is not empty
            if best_move is None: 
                 return random.choice(list(children_to_consider.keys()))
            return best_move
        else:
            moves = list(children_to_consider.keys())
            visit_counts = np.array([children_to_consider[m].N for m in moves], dtype=float)

            # Avoid division by zero if temperature is very small
            if temperature < 1e-6: temperature = 1e-6
            
            visit_counts_temp = visit_counts**(1.0 / temperature)
            # Fallback to uniform probability if temp calculation fails
            if np.isinf(visit_counts_temp).any() or np.sum(visit_counts_temp) == 0:
                 probabilities = np.ones_like(visit_counts) / len(visit_counts)
            else:
                 probabilities = visit_counts_temp / np.sum(visit_counts_temp)
            
            probabilities /= probabilities.sum()

            chosen_move = np.random.choice(moves, p=probabilities)
            return chosen_move

    def get_policy_distribution(self, root_node: MCTSNode, temperature: float = 1.0) -> np.ndarray:
        """
        Calculates the MCTS policy distribution (pi) based on visit counts.

        Args:
            root_node: The root node after search.
            temperature: The temperature parameter for sampling. T=1 means proportional
                         to visits, T->0 means deterministic (max visits).

        Returns:
            A numpy array of size self.network.action_space_size representing the
            policy distribution (probabilities for each action ID).
        """
        policy_pi = np.zeros(self.network.action_space_size, dtype=np.float32)

        if not root_node.children:
            legal_moves = list(root_node.board.legal_moves)
            num_legal = len(legal_moves)
            if num_legal > 0:
                uniform_prob = 1.0 / num_legal
                for move in legal_moves:
                    action_id = root_node.board.move_to_action_id(move)
                    if action_id is not None and 1 <= action_id <= self.network.action_space_size:
                        policy_pi[action_id - 1] = uniform_prob
            return policy_pi

        children_moves = list(root_node.children.keys())
        visit_counts = np.array([root_node.children[m].N for m in children_moves], dtype=float)
        total_visits = root_node.N

        if total_visits == 0 or not children_moves:
             # Fallback if no visits recorded
             legal_moves = list(root_node.board.legal_moves)
             num_legal = len(legal_moves)
             if num_legal > 0:
                 uniform_prob = 1.0 / num_legal
                 for move in legal_moves:
                     action_id = root_node.board.move_to_action_id(move)
                     if action_id is not None and 1 <= action_id <= self.network.action_space_size:
                         policy_pi[action_id - 1] = uniform_prob
             return policy_pi

        if temperature == 0:
            max_visits = np.max(visit_counts)
            best_moves = [move for move, count in zip(children_moves, visit_counts) if count == max_visits]
            prob = 1.0 / len(best_moves)
            for move in best_moves:
                 action_id = root_node.board.move_to_action_id(move)
                 if action_id is not None and 1 <= action_id <= self.network.action_space_size:
                     policy_pi[action_id - 1] = prob
        else:
            visit_counts_temp = visit_counts**(1.0 / temperature)
            # Handle potential overflow or zero sum
            sum_counts_temp = np.sum(visit_counts_temp)
            if np.isinf(visit_counts_temp).any() or sum_counts_temp == 0:
                # Fallback to simple normalization if temperature causes issues
                probabilities = visit_counts / total_visits
            else:
                probabilities = visit_counts_temp / sum_counts_temp

            for move, prob in zip(children_moves, probabilities):
                action_id = root_node.board.move_to_action_id(move)
                if action_id is not None and 1 <= action_id <= self.network.action_space_size:
                     policy_pi[action_id - 1] = prob
                # else: Handle move mapping failure?

        # Ensure probabilities sum to 1
        current_sum = np.sum(policy_pi)
        if current_sum > 0:
            policy_pi /= current_sum
        # If sum is 0 but there were moves, fallback to uniform
        elif len(children_moves) > 0: 
            uniform_prob = 1.0 / len(children_moves)
            for move in children_moves:
                 action_id = root_node.board.move_to_action_id(move)
                 if action_id is not None and 1 <= action_id <= self.network.action_space_size:
                    policy_pi[action_id - 1] = uniform_prob
            # Renormalize just in case
            final_sum = np.sum(policy_pi)
            if final_sum > 0:
                 policy_pi /= final_sum

        return policy_pi 