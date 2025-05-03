import random
import time
import torch
import torch.nn.functional as F
from rich.progress import Progress
import numpy as np
import copy

import chess

import sys
import os
sys.path.append(os.path.abspath('.'))

import chess_gym
from chess_gym.envs import ChessEnv
from utils.policy_human import sample_action

# Import from sibling modules
from network import ChessNetwork
from mcts_node import MCTSNode

# --- MCTS Algorithm ---
class MCTS:
    def __init__(self, network: ChessNetwork, device: torch.device | str, env: ChessEnv, player_color: chess.Color = chess.WHITE, C_puct=1.41): # Add env params
        self.network = network
        self.device = torch.device(device) # Ensure it's a torch.device object
        self.player_color = player_color
        self.C_puct = C_puct
        self.env = env

    def search(self, root_node: MCTSNode, iterations: int,
               progress: Progress | None = None, parent_task_id = None):
        """Runs the MCTS search for a fixed number of iterations.

        Args:
            root_node: The node to start the search from.
            iterations: The number of MCTS simulations to run.
            progress: Optional rich Progress object for displaying sub-progress.
            parent_task_id: Optional task ID in the Progress object under which to nest.
        """
        start_time = time.time()

        mcts_task_id = None
        if progress is not None and parent_task_id is not None:
            # Add a sub-task for MCTS iterations if progress context is provided
            # Note: Rich doesn't have explicit nesting, but this shows a second bar.
            # We can make it transient so it disappears after this search call.
            mcts_task_id = progress.add_task("  ├─ MCTS Sims", total=iterations, transient=True, start=False)
            # Start the task explicitly if needed
            progress.start_task(mcts_task_id)
            print(f"Starting MCTS search with {iterations} iterations...")

        for i in range(iterations):
            # =================================
            #        Phase 1: Selection
            # =================================
            node = root_node
            search_path = [node]

            while not node.is_terminal():
                # --- Opponent Move Selection (Heuristic/Random) ---
                # This part becomes tricky with env. Needs careful env state handling.
                # If we use a heuristic that needs the board, access via node.board.
                # Stepping the env here modifies it in-place within the node, which is problematic
                # for MCTS state purity. Let's simplify for now and assume player vs player self-play
                # or handle opponent moves outside MCTS search if using envs strictly.
                # For now, let's assume the search focuses only on the 'player_color' turns
                # and opponent moves happen in the outer game loop.
                # --- OR: We need env copying here too --- 
                
                # Let's proceed assuming the player_color check works and we only expand player nodes
                if node.board.turn != self.player_color:
                    # If MCTS needs to handle opponent moves, we MUST copy the env before stepping
                    # Option 1: Skip opponent turns in search (handled by outer loop)
                    # Option 2: Implement copy and step for opponent (complex)
                    # Let's assume Option 1 for now for simplicity - MCTS only guides player_color
                    # This means the search path might not be fully accurate if opponent moves aren't simulated
                    # but matches the previous logic pattern. We select based on UCT for player turns.
                    # If we reach an opponent node, we can't select further down that path using UCT.
                    # Let's refine the logic: If it's opponent's turn, stop selection here for this path.
                    break # Stop selection phase if it lands on opponent's turn (treat as leaf for this sim)

                # --- Player Move Selection / Expansion Decision ---
                else: # It is self.player_color's turn
                    if node.N == 0:
                        # Leaf node for the player, break to expand
                        break
                    else:
                        if not node.children:
                            # Visited but no children yet (maybe all moves led to terminal/opponent?) break to expand
                            break
                        # Node visited and has children, select best child using UCT
                        node = node.select_child_uct(self.C_puct)
                        search_path.append(node)
                        # No need to check terminal here, outer loop does

            # =================================
            #        Phase 2: Expansion
            # =================================
            leaf_node = node # The node where selection ended
            value = 0.0 # Initialize value for backpropagation

            # Expand if the selection didn't end on a terminal node
            # And if it's the player's turn (we only expand nodes where player needs to move)
            if not leaf_node.is_terminal() and leaf_node.board.turn == self.player_color:
                # Get observation from the node's environment state
                obs_vector = leaf_node.board.get_board_vector() # Use the property
                obs_tensor = torch.tensor(obs_vector, dtype=torch.float32).to(self.device)
                
                # Get network prediction
                with torch.no_grad():
                    policy_logits, value = self.network(obs_tensor)
                
                policy_logits = policy_logits.squeeze(0) # <-- Add this line to remove batch dimension
                value = value.item() # Get scalar value

                # --- Calculate probabilities over FULL action space ---
                full_probs = F.softmax(policy_logits, dim=0)
                use_uniform_fallback = False
                if torch.isnan(full_probs).any():
                    num_legal_moves = len(list(leaf_node.board.legal_moves))
                    uniform_prob = 1.0 / num_legal_moves if num_legal_moves > 0 else 0.0
                    use_uniform_fallback = True
                    # print("Warning: Softmax resulted in NaN, falling back to uniform priors for legal moves and assuming network value is 0.")
                    value = 0.0 # If probs are NaN, network value is likely unreliable

                # --- Check if network's top choice is legal ---
                if not use_uniform_fallback:
                    top_prob, top_action_id_0based = torch.max(full_probs, dim=0)
                    top_action_id = top_action_id_0based.item() + 1 # Convert back to 1-based ID
                    try:
                        # Attempt to convert the top action ID back to a move
                        top_move = leaf_node.board.action_id_to_move(top_action_id)
                        # Check if the move is actually legal in the current state
                        if top_move in leaf_node.board.legal_moves:
                            value = value # Top choice is legal, use network's value
                        else:
                            value = -1.0 # Top choice is ILLEGAL, penalize heavily
                    except ValueError:
                         # If action_id_to_move fails (e.g., ID out of range for current board state)
                         value = -1.0 # Treat invalid action ID as illegal suggestion, penalize
                else:
                    # If using uniform fallback due to NaN, use the (potentially reset) value
                    value = value

                # --- Expand children for legal moves --- 
                # No need for legality check using network top choice here, 
                # value is directly from network. Backprop handles learning. 
                
                legal_moves_list = list(leaf_node.board.legal_moves) # Get legal moves from board in env
                if not legal_moves_list and not leaf_node.is_terminal():
                    pass # Should be caught by is_terminal()
                
                for move in legal_moves_list:
                    # Get prior probability for this move
                    action_id = leaf_node.board.move_to_action_id(move) # Use board from env
                    prior_p = 0.0
                    if action_id is not None:
                         action_id_0based = action_id - 1
                         if 0 <= action_id_0based < self.network.action_space_size:
                             if use_uniform_fallback:
                                 prior_p = uniform_prob
                             else:
                                 prior_p = full_probs[action_id_0based].item() # Now indexing 1D tensor
                         else:
                             # Handle case where action_id is out of bounds for the network output
                             # This might happen if env and network action spaces differ
                             if not use_uniform_fallback:
                                  print(f"Warning: Action ID {action_id_0based} from move {move.uci()} is out of bounds for network output size {full_probs.shape[0]}. Assigning prior 0.0.")
                             # prior_p remains 0.0 or uniform_prob if fallback is active
                    else:
                        # Handle case where move couldn't be converted to action ID
                        if not use_uniform_fallback:
                             print(f"Warning: Could not get action ID for move {move.uci()}. Assigning prior 0.0.")
                        # prior_p remains 0.0 or uniform_prob if fallback is active

                    # Create child node if it doesn't exist
                    if move not in leaf_node.children:
                        # *** Crucial Step: Create independent env state for child ***
                        try:
                            self.env.board.set_fen(leaf_node.fen, leaf_node.board.piece_tracker)

                            # --- Convert move to action for env.step --- 
                            action_to_take = self.env.board.move_to_action_id(move)
                            if action_to_take is None:
                                print(f"Warning (MCTS Expansion): Could not convert move {move.uci()} to action ID. Trying legacy format.")
                                # Fallback to legacy format if ID fails (using the env's action space)
                                action_to_take = self.env.action_space._move_to_action(move, return_id=False)
                            # --- End Conversion --- 
                            
                            # Step the copied environment with the converted action
                            # We don't strictly need the return values here
                            _, _, _, _, _ = self.env.step(action_to_take) # Apply action to the *copy*
                            # Step opponent agent
                            self.env.step(sample_action(self.env.action_space, return_id=True))
                            fen = self.env.action_space.board.fen()

                            child_node = MCTSNode(fen, piece_tracker=self.env.board.piece_tracker, parent=leaf_node, prior_p=prior_p, move_leading_here=move)
                            leaf_node.children[move] = child_node
                        except Exception as e:
                             # Handle cases where deepcopy or step fails
                             print(f"Error during child node creation for move {move.uci()}: {e}")
                             # Skipping for now
                             continue
                                
            elif leaf_node.is_terminal():
                 # Get terminal value from the board state in the environment
                 result_str = leaf_node.board.result()
                 if result_str == "1-0": value = 1.0
                 elif result_str == "0-1": value = -1.0
                 else: value = 0.0
            else:
                # Landed on opponent turn node or non-expandable terminal node
                # Use 0 as neutral value if not terminal, otherwise terminal value
                if leaf_node.is_terminal():
                     result_str = leaf_node.board.result()
                     if result_str == "1-0": value = 1.0
                     elif result_str == "0-1": value = -1.0
                     else: value = 0.0
                else:
                    # E.g., Selection stopped because it hit opponent's turn
                    value = 0.0 # Neutral value

            # =================================
            #        Phase 3: Simulation (Skipped)
            # =================================
            pass

            # =================================
            #      Phase 4: Backpropagation
            # =================================
            # Logic largely remains the same, but uses value perspective based on board.turn
            for node_in_path in reversed(search_path):
                node_in_path.N += 1
                # Determine value perspective for the node being updated
                current_node_perspective_value = value
                # Check if the turn *at the leaf node where value was determined* is different
                # from the turn *at the node currently being updated*. If so, flip value.
                if node_in_path.board.turn != leaf_node.board.turn:
                    current_node_perspective_value = -value
                
                node_in_path.W += current_node_perspective_value

            # Update MCTS sub-task progress if active
            if mcts_task_id is not None and progress is not None:
                progress.update(mcts_task_id, advance=1)

        # Stop/remove the MCTS sub-task after the loop
        if mcts_task_id is not None and progress is not None:
            progress.stop_task(mcts_task_id)
            progress.update(mcts_task_id, visible=False) # Hide it explicitly
            # Alternatively, remove it, but stopping might be cleaner if transient=True works well
            # progress.remove_task(mcts_task_id)

        end_time = time.time()
        # print(f"Search finished {iterations} iterations in {end_time - start_time:.2f} seconds.")

    def get_best_move(self, root_node: MCTSNode, temperature=0.0) -> chess.Move:
        """
        Selects the best move from the root node after search.
        - temperature = 0: Choose the most visited node.
        - temperature > 0: Sample from visit counts distribution.
        """
        if not root_node.children:
            # Handle case with no children explored (maybe root is terminal or iterations=0)
            legal = list(root_node.board.legal_moves)
            return random.choice(legal) if legal else None 
            # raise RuntimeError("MCTS search hasn't been run or root has no children.")

        children_to_consider = {m: c for m, c in root_node.children.items() if c.N > 0}
        if not children_to_consider:
            # Handle case where only opponent moves were explored shallowly or no nodes visited
            # Fallback: Choose randomly among unexplored children if they exist
            unexplored_children = [m for m, c in root_node.children.items() if c.N == 0]
            if unexplored_children:
                return random.choice(unexplored_children)
            else: # If NO children at all (should be caught above, but safety) 
                legal = list(root_node.board.legal_moves)
                return random.choice(legal) if legal else None

        if temperature == 0:
            max_visits = -1
            best_move = None
            for move, child in children_to_consider.items():
                if child.N > max_visits:
                    max_visits = child.N
                    best_move = move
            if best_move is None: # Should not happen if children_to_consider is not empty
                 return random.choice(list(children_to_consider.keys()))
            return best_move
        else:
            moves = list(children_to_consider.keys())
            visit_counts = np.array([children_to_consider[m].N for m in moves], dtype=float)

            # Avoid division by zero if temperature is very small
            if temperature < 1e-6: temperature = 1e-6
            
            visit_counts_temp = visit_counts**(1.0 / temperature)
            # Handle potential overflow or zero counts leading to NaN/Inf probabilities
            if np.isinf(visit_counts_temp).any() or np.sum(visit_counts_temp) == 0:
                 # Fallback to uniform probability if temp calculation fails
                 probabilities = np.ones_like(visit_counts) / len(visit_counts)
            else:
                 probabilities = visit_counts_temp / np.sum(visit_counts_temp)
            
            # Ensure probabilities sum to 1 (handle floating point issues)
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
            # Use board from env
            legal_moves = list(root_node.board.legal_moves)
            num_legal = len(legal_moves)
            if num_legal > 0:
                uniform_prob = 1.0 / num_legal
                for move in legal_moves:
                    # Use board from env for move_to_action_id
                    action_id = root_node.board.move_to_action_id(move)
                    if action_id is not None and 1 <= action_id <= self.network.action_space_size:
                        policy_pi[action_id - 1] = uniform_prob
            return policy_pi

        children_moves = list(root_node.children.keys())
        visit_counts = np.array([root_node.children[m].N for m in children_moves], dtype=float)
        total_visits = root_node.N

        if total_visits == 0 or not children_moves:
             # Fallback if no visits recorded (should not happen after search)
             # Return uniform as above
             legal_moves = list(root_node.board.legal_moves)
             num_legal = len(legal_moves)
             if num_legal > 0:
                 uniform_prob = 1.0 / num_legal
                 for move in legal_moves:
                     action_id = root_node.board.move_to_action_id(move)
                     if action_id is not None and 1 <= action_id <= self.network.action_space_size:
                         policy_pi[action_id - 1] = uniform_prob
             return policy_pi

        # Calculate probabilities based on temperature
        if temperature == 0:
            # Deterministic: Find the move(s) with max visits
            max_visits = np.max(visit_counts)
            best_moves = [move for move, count in zip(children_moves, visit_counts) if count == max_visits]
            prob = 1.0 / len(best_moves)
            for move in best_moves:
                 action_id = root_node.board.move_to_action_id(move)
                 if action_id is not None and 1 <= action_id <= self.network.action_space_size:
                     policy_pi[action_id - 1] = prob
        else:
            # Apply temperature
            visit_counts_temp = visit_counts**(1.0 / temperature)
            # Handle potential overflow or zero sum
            sum_counts_temp = np.sum(visit_counts_temp)
            if np.isinf(visit_counts_temp).any() or sum_counts_temp == 0:
                # Fallback to simple normalization if temperature causes issues
                probabilities = visit_counts / total_visits
            else:
                probabilities = visit_counts_temp / sum_counts_temp

            # Assign probabilities to the policy vector
            for move, prob in zip(children_moves, probabilities):
                action_id = root_node.board.move_to_action_id(move)
                if action_id is not None and 1 <= action_id <= self.network.action_space_size:
                     policy_pi[action_id - 1] = prob
                # else: Handle move mapping failure? Should be logged by move_to_action_id

        # Ensure probabilities sum to 1 (handle floating point issues)
        current_sum = np.sum(policy_pi)
        if current_sum > 0:
            policy_pi /= current_sum
        elif len(children_moves) > 0: # If sum is 0 but there were moves, fallback to uniform
            uniform_prob = 1.0 / len(children_moves)
            for move in children_moves:
                 action_id = root_node.board.move_to_action_id(move)
                 if action_id is not None and 1 <= action_id <= self.network.action_space_size:
                    policy_pi[action_id - 1] = uniform_prob
            policy_pi /= np.sum(policy_pi) # Renormalize just in case

        return policy_pi 