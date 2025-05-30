import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple
from mcts_node import MCTSNode

class MCTS:
    def __init__(self, network, device, env, C_puct=1.41):
        self.network = network
        self.device = torch.device(device)
        self.env = env
        self.C_puct = C_puct
        self.network.eval()

    def _select(self, root_node: MCTSNode) -> Tuple[MCTSNode, List[MCTSNode]]:
        node = root_node
        search_path = [node]
        while node.is_expanded and node.children:
            best_score = -float('inf')
            best_child = None
            for action_id, child in node.children.items():
                score = child.Q() + child.U(self.C_puct)
                if score > best_score:
                    best_score = score
                    best_child = child
            if best_child is None:
                break
            node = best_child
            search_path.append(node)
        return node, search_path

    def _expand_and_evaluate(self, node: MCTSNode):
        if node.is_expanded:
            return 0.0
        obs_tensor = torch.tensor(node.observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            policy_logits, value_pred = self.network(obs_tensor)
        policy_logits = policy_logits.squeeze(0)
        value = value_pred.item()
        probs = F.softmax(policy_logits, dim=0).cpu().numpy()
        legal_actions = range(self.env.action_space.n)
        for action_id in legal_actions:
            if action_id not in node.children:
                # Try to clone and step the environment
                if hasattr(self.env, 'clone_state') and hasattr(self.env, 'restore_state'):
                    state = self.env.clone_state()
                    obs, reward, terminated, truncated, info = self.env.step(action_id)
                    self.env.restore_state(state)
                else:
                    print("Warning: Environment does not support clone_state/restore_state. Expansion may be incorrect.")
                    obs, reward, terminated, truncated, info = self.env.step(action_id)
                child_node = MCTSNode(obs, parent=node, prior_p=probs[action_id], action_id_leading_here=action_id)
                child_node._is_terminal = terminated or truncated
                node.children[action_id] = child_node
        node.is_expanded = True
        return value

    def _backpropagate(self, search_path: List[MCTSNode], value: float):
        for node in reversed(search_path):
            node.N += 1
            node.W += value
            value = -value  # For two-player games; for single-player, you may want to keep value

    def search(self, root_node: MCTSNode, iterations: int):
        for _ in range(iterations):
            leaf, path = self._select(root_node)
            value = self._expand_and_evaluate(leaf)
            self._backpropagate(path, value)

    def get_best_action(self, root_node: MCTSNode, temperature=0.0):
        if not root_node.children:
            return np.random.choice(range(self.env.action_space.n))
        visit_counts = np.array([child.N for child in root_node.children.values()])
        action_ids = list(root_node.children.keys())
        if np.sum(visit_counts) == 0 or np.isnan(np.sum(visit_counts)):
            # Uniform fallback
            return np.random.choice(action_ids)
        if temperature == 0:
            return action_ids[np.argmax(visit_counts)]
        else:
            probs = visit_counts ** (1.0 / max(temperature, 1e-6))
            sum_probs = np.sum(probs)
            if sum_probs == 0 or np.isnan(sum_probs):
                probs = np.ones_like(probs) / len(probs)
            else:
                probs = probs / sum_probs
            return np.random.choice(action_ids, p=probs)

    def get_policy_distribution(self, root_node: MCTSNode, temperature=1.0):
        policy = np.zeros(self.env.action_space.n, dtype=np.float32)
        if not root_node.children:
            policy += 1.0 / self.env.action_space.n
            return policy
        visit_counts = np.array([child.N for child in root_node.children.values()])
        action_ids = list(root_node.children.keys())
        if np.sum(visit_counts) == 0 or np.isnan(np.sum(visit_counts)):
            # Uniform fallback
            for aid in action_ids:
                policy[aid] = 1.0 / len(action_ids)
            return policy
        if temperature == 0:
            best = np.argmax(visit_counts)
            policy[action_ids[best]] = 1.0
        else:
            probs = visit_counts ** (1.0 / max(temperature, 1e-6))
            sum_probs = np.sum(probs)
            if sum_probs == 0 or np.isnan(sum_probs):
                for aid in action_ids:
                    policy[aid] = 1.0 / len(action_ids)
            else:
                probs = probs / sum_probs
                for aid, p in zip(action_ids, probs):
                    policy[aid] = p
        return policy 