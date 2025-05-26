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

from chess_gym.envs import ChessEnv
from MCTS.models.network import ChessNetwork
from mcts_node import MCTSNode
from utils.analyze import get_action_id_for_piece_abs


# --- MCTS Algorithm ---
class MCTS:
    def __init__(self,
                 network: ChessNetwork,
                 device: torch.device | str,
                 env: ChessEnv | None = None, # Keep main env for selection/initial state
                 C_puct: float = 1.41,
                 dirichlet_alpha: float = 0.3,
                 dirichlet_epsilon: float = 0.25,
                 action_space_mode: str = "1700"):  # Add action_space_mode parameter
        self.network = network
        self.device = torch.device(device)
        self.env = env # If None, board.push() will be used for expansion
        self.C_puct = C_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.action_space_mode = action_space_mode  # Store action space mode
        self.network.eval()

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
                node = node.select_child_uct(self.C_puct)
                search_path.append(node)

        leaf_node = node
        return leaf_node, search_path

    # --- Expansion Helper Methods ---

    def _expand_sequential(self, leaf_node: MCTSNode, probs_to_use: torch.Tensor,
                           uniform_prob: float, use_uniform_fallback: bool):
        """Expands leaf node's children using self.env.step(). Requires self.env.

        Assumes leaf_node is not terminal.
        """
        if self.env is None:
             raise RuntimeError("_expand_sequential called but self.env is None.")

        leaf_board = leaf_node.get_board()
        original_fen = leaf_board.fen()
        original_tracker = getattr(leaf_board, 'piece_tracker', None)
        legal_action_ids = list(leaf_board.legal_actions)

        for action_id in legal_action_ids:
            if action_id not in leaf_node.children:
                self.env.board.set_fen(original_fen, original_tracker)
                move = leaf_board.action_id_to_move(action_id)
                if move is None: continue

                try:
                    self.env.step(action_id)
                except Exception as e:
                    print(f"Error stepping primary action_id {action_id} from FEN {original_fen}: {e}")
                    continue

                prior_p = 0.0
                action_id_0based = action_id - 1
                if 0 <= action_id_0based < probs_to_use.shape[0]:
                    prior_p = uniform_prob if use_uniform_fallback else probs_to_use[action_id_0based].item()

                child_board = self.env.board.copy()
                child_node = MCTSNode(
                    board=child_board,
                    parent=leaf_node,
                    prior_p=prior_p,
                    action_id_leading_here=action_id
                )
                leaf_node.children[action_id] = child_node

        self.env.board.set_fen(original_fen, original_tracker)

    def _expand_parallel(self, leaf_node: MCTSNode, probs_to_use: torch.Tensor,
                         uniform_prob: float, use_uniform_fallback: bool):
        """Expands leaf node's children using board.copy() and board.push().

        Assumes leaf_node is not terminal.
        """
        leaf_board = leaf_node.get_board()
        for action_id in leaf_board.legal_actions:
            action_id_0based = action_id - 1
            prior_p = 0.0
            if 0 <= action_id_0based < probs_to_use.shape[0]:
                prior_p = uniform_prob if use_uniform_fallback else probs_to_use[action_id_0based].item()
            if action_id not in leaf_node.children:
                try:
                    sim_board = leaf_board.copy()
                    move = sim_board.action_id_to_move(action_id)
                    if move is None:
                        continue
                    sim_board.push(move)
                    child_node = MCTSNode(
                        board=sim_board,
                        parent=leaf_node,
                        prior_p=prior_p,
                        action_id_leading_here=action_id
                    )
                    leaf_node.children[action_id] = child_node
                except Exception as e:
                    print(f"Error during MCTS board.push expansion for action_id {action_id} from FEN {leaf_board.fen()}: {e}")

    # --- Main Expansion & Evaluation Method ---

    def _expand_and_evaluate(self, leaf_node: MCTSNode, progress: Progress | None = None) -> float:
        """Phase 2: Gets value for leaf_node and expands its children.

        Dispatches expansion logic to _expand_sequential (if progress is not None)
        or _expand_parallel (if progress is None).
        """
        value = 0.0
        if not leaf_node.is_expanded:
            leaf_board = leaf_node.get_board()

            # 1. Handle Terminal Node
            if leaf_board.is_game_over(claim_draw=True):
                result_str = leaf_board.result(claim_draw=True)
                if result_str == "1-0": value = 1.0
                elif result_str == "0-1": value = -1.0
                else: value = 0.0 # Draw
                leaf_node.is_expanded = True
                # No expansion needed for terminal nodes
            else:
                # 2. Non-Terminal: Get Network Prediction
                obs_vector = leaf_board.get_board_vector()
                obs_tensor = torch.tensor(obs_vector, dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    policy_logits, value_pred = self.network(obs_tensor)
                policy_logits = policy_logits.squeeze(0)
                value = value_pred.item() # Use network's value prediction

                # 3. Calculate Probabilities (with potential fallback and noise)
                full_probs = F.softmax(policy_logits, dim=0)
                use_uniform_fallback = False
                uniform_prob = 0.0
                if torch.isnan(full_probs).any():
                    num_legal_actions = len(list(leaf_board.legal_actions))
                    uniform_prob = 1.0 / num_legal_actions if num_legal_actions > 0 else 0.0
                    use_uniform_fallback = True
                    # Keep value = 0.0 if network output unreliable? Or use predicted value?
                    # Let's keep the predicted value unless it's NaN itself (unlikely)

                probs_to_use = full_probs
                # Add Dirichlet Noise (only if root node)
                if leaf_node.parent is None and not use_uniform_fallback:
                    # --- Dirichlet Noise Logic --- (Same as before)
                    legal_actions = list(leaf_board.legal_actions)
                    num_legal = len(legal_actions)
                    if num_legal > 1:
                        legal_indices = []
                        legal_probs = []
                        for action_id in legal_actions:
                            if action_id is not None:
                                idx = action_id - 1
                                if 0 <= idx < full_probs.shape[0]:
                                    legal_indices.append(idx)
                                    legal_probs.append(full_probs[idx].item())
                        if legal_indices:
                            noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal_indices))
                            noise_t = torch.tensor(noise, device=self.device, dtype=torch.float32)
                            legal_probs_t = torch.tensor(legal_probs, device=self.device, dtype=torch.float32)
                            combined = (1 - self.dirichlet_epsilon) * legal_probs_t + self.dirichlet_epsilon * noise_t
                            probs_to_use = full_probs.clone()
                            indices_t = torch.tensor(legal_indices, dtype=torch.long, device=self.device)
                            probs_to_use[indices_t] = combined

                # 4. Dispatch to appropriate expansion method
                if progress is not None:
                    # Use sequential method (requires self.env)
                    self._expand_sequential(leaf_node, probs_to_use, uniform_prob, use_uniform_fallback)
                else:
                    # Use parallel-safe method (board.push)
                    self._expand_parallel(leaf_node, probs_to_use, uniform_prob, use_uniform_fallback)

                # 5. Mark expanded after attempting expansion
                leaf_node.is_expanded = True

        # Return the value obtained from the network (or terminal state)
        return value

    def _backpropagate(self, search_path: List[MCTSNode], value: float, leaf_node_turn: chess.Color):
        """Phase 4: Backpropagates value up the search path."""
        for node in reversed(search_path):
            node_turn = node.get_board().turn
            current_node_perspective_value = value if node_turn == leaf_node_turn else -value
            node.N += 1
            node.W += current_node_perspective_value

    # --- Main Search Function ---
    def search(self, root_node: MCTSNode, iterations: int, progress: Progress | None = None):
        
        if progress is not None:
            root_fen = self.env.board.fen()
            root_piece_tracker = self.env.board.piece_tracker

        mcts_task_id = None
        if progress is not None:
            mcts_task_id = progress.add_task(f"  ├─ MCTS Sims (0/{iterations})", total=iterations, transient=True, start=False)
            progress.start_task(mcts_task_id)

        for i in range(iterations):
            # Phase 1: Selection
            leaf_node, search_path = self._select(root_node)

            # Phase 2: Expansion & Evaluation
            value = self._expand_and_evaluate(leaf_node, progress=progress)

            # Phase 4: Backpropagation
            leaf_node_turn = leaf_node.get_board().turn
            self._backpropagate(search_path, value, leaf_node_turn)

            if mcts_task_id is not None and progress is not None:
                 progress.update(mcts_task_id, advance=1, description=f"  ├─ MCTS Sims ({i+1}/{iterations})")

        if mcts_task_id is not None and progress is not None:
            progress.stop_task(mcts_task_id)
            progress.update(mcts_task_id, visible=False)

        if progress is not None:
            self.env.board.set_fen(root_fen, root_piece_tracker)

    # --- Get Best Move / Policy --- 
    # Returns Action ID corresponding to the best move
    def get_best_action(self, root_node: MCTSNode, temperature=0.0) -> int | None:
        """
        Selects the best **action ID** corresponding to a legal move from the
        root node after search, based on visit counts.
        - temperature = 0: Choose the action ID of the most visited legal node.
        - temperature > 0: Sample an action ID from visit counts distribution of legal moves.

        Args:
            root_node: The root node of the MCTS search.
            temperature: Controls exploration in selection. 0 for deterministic.

        Returns:
            The integer action ID (1-based) of the selected best legal move, or None
            if no legal moves exist or the action ID cannot be determined.
        """
        if not root_node.children:
            legal_action_ids = list(root_node.board.legal_actions)
            if not legal_action_ids: return None
            selected_action_id = random.choice(legal_action_ids)
            return selected_action_id

        children_to_consider = {aid: c for aid, c in root_node.children.items() if aid in root_node.board.legal_actions and c.N > 0}

        if not children_to_consider:
            legal_action_ids = list(root_node.board.legal_actions)
            if not legal_action_ids: return None
            selected_action_id = random.choice(legal_action_ids)
            return selected_action_id

        if temperature == 0:
            max_visits = -1
            best_action_id = None
            for aid, child in children_to_consider.items():
                if child.N > max_visits:
                    max_visits = child.N
                    best_action_id = aid
            if best_action_id is None:
                best_action_id = random.choice(list(children_to_consider.keys()))
            return best_action_id
        else:
            action_ids = list(children_to_consider.keys())
            visit_counts = np.array([children_to_consider[aid].N for aid in action_ids], dtype=float)
            if temperature < 1e-6: temperature = 1e-6
            visit_counts_temp = visit_counts**(1.0 / temperature)
            sum_counts_temp = np.sum(visit_counts_temp)
            if np.isinf(sum_counts_temp) or sum_counts_temp <= 1e-9 or np.isnan(sum_counts_temp):
                probabilities = np.ones_like(visit_counts) / len(visit_counts)
            else:
                probabilities = visit_counts_temp / sum_counts_temp
            sum_probs = np.sum(probabilities)
            if not np.isclose(sum_probs, 1.0):
                if sum_probs > 1e-9: probabilities /= sum_probs
                else: probabilities = np.ones_like(visit_counts) / len(visit_counts)
            chosen_action_id = np.random.choice(action_ids, p=probabilities)
            return chosen_action_id

    def get_policy_distribution(self, root_node: MCTSNode, temperature: float = 1.0) -> np.ndarray:
        """
        Calculates the MCTS policy distribution (pi) based on visit counts.
        """
        policy_pi = np.zeros(self.network.action_space_size, dtype=np.float32)
        if not root_node.children:
            legal_action_ids = list(root_node.board.legal_actions)
            num_legal = len(legal_action_ids)
            if num_legal > 0:
                uniform_prob = 1.0 / num_legal
                for action_id in legal_action_ids:
                    if action_id is not None and 1 <= action_id <= self.network.action_space_size:
                        policy_pi[action_id - 1] = uniform_prob
            return policy_pi
        children_action_ids = list(root_node.children.keys())
        visit_counts = np.array([root_node.children[aid].N for aid in children_action_ids], dtype=float)
        total_visits = root_node.N
        if total_visits == 0 or not children_action_ids:
            legal_action_ids = list(root_node.board.legal_actions)
            num_legal = len(legal_action_ids)
            if num_legal > 0:
                uniform_prob = 1.0 / num_legal
                for action_id in legal_action_ids:
                    if action_id is not None and 1 <= action_id <= self.network.action_space_size:
                        policy_pi[action_id - 1] = uniform_prob
            return policy_pi
        if temperature == 0:
            max_visits = np.max(visit_counts)
            if max_visits == 0:
                best_action_ids = children_action_ids
            else:
                best_action_ids = [aid for aid, count in zip(children_action_ids, visit_counts) if count == max_visits]
            prob = 1.0 / len(best_action_ids) if best_action_ids else 0.0
            for action_id in best_action_ids:
                if action_id is not None and 1 <= action_id <= self.network.action_space_size:
                    policy_pi[action_id - 1] = prob
        else:
            visit_counts_temp = visit_counts**(1.0 / temperature)
            sum_counts_temp = np.sum(visit_counts_temp)
            if np.isinf(visit_counts_temp).any() or sum_counts_temp == 0:
                probabilities = visit_counts / total_visits if total_visits > 0 else np.ones_like(visit_counts) / len(visit_counts)
            else:
                probabilities = visit_counts_temp / sum_counts_temp
            for action_id, prob in zip(children_action_ids, probabilities):
                if action_id is not None and 1 <= action_id <= self.network.action_space_size:
                    policy_pi[action_id - 1] = prob
        current_sum = np.sum(policy_pi)
        if current_sum > 1e-6:
            policy_pi /= current_sum
        elif len(children_action_ids) > 0:
            uniform_prob = 1.0 / len(children_action_ids)
            for action_id in children_action_ids:
                if action_id is not None and 1 <= action_id <= self.network.action_space_size:
                    policy_pi[action_id - 1] = uniform_prob
            final_sum = np.sum(policy_pi)
            if final_sum > 1e-6:
                policy_pi /= final_sum
        return policy_pi 