# Learning to Prevent Stalemate & Insufficient Material Blunders

This guide explains how the training pipeline teaches the model to avoid blundering into stalemate or insufficient material when winning, and how it handles the opponent's salvaging draws. Combined from `docs/STALEMATE_LEARNING_GUIDE.md` and `docs/insufficient_material_analysis.md`, updated for recent changes.

---

## 1. How Draw Terminals Work in MCTS

MCTS must detect stalemate and insufficient material so it can both avoid blundering and correctly simulate the opponent's salvaging moves.

### Terminal check during expansion

When we **expand** a node and create children (one per legal move), we push each move and create a child board. The child = position after our move. We **immediately** check `child_board.is_game_over()`. If stalemate, checkmate, or insufficient material, we:

- Set `child.W` and `child.N` (pre-initialize)
- Mark `child.is_expanded = True`

This way the parent's UCT sees the terminal move's value **without needing to select it first**.

### Pre-init behavior by termination type

| Termination         | Pre-init               | Term value (W)                                             |
|---------------------|------------------------|------------------------------------------------------------|
| Checkmate           | Yes (high prior)       | 1.0 or -1.0 from `get_terminal_value_for_mcts`             |
| INSUFFICIENT_MATERIAL | Yes (high prior)     | 1.0 — ensures opponent's salvaging capture is force-selected |
| Other draws         | Only if `pre_init_draws: true` | `draw_reward` or `draw_reward_table`             |

**Fix:** Enable `pre_init_draws: true` so draw terminals are marked immediately. They get W from config, N=1, so UCT treats them as known → fewer visits on draw moves → stronger "avoid" signal for our blunders.

---

## 2. Recent Implementation: Force-select and move selection

### Force-select in MCTS selection

In `select_child_uct` (`mcts_node.py`), **before** PUCT:

- If any child is terminal with `Q >= 0.99`, force-select uniformly among those (checkmate or opponent's salvaging INSUFFICIENT_MATERIAL).
- Ensures the opponent's salvaging capture is always explored when it's the opponent's turn.
- Our blunder (INSUFFICIENT_MATERIAL from our capture) is two plies away, so it cannot be detected at expansion and relies on normal search.

### Temperature = 0 when winning terminal exists

In `train.py`, after MCTS search:

- If any root child has `Q >= 0.99` (checkmate or opponent's salvaging INSUFFICIENT_MATERIAL), use `temperature=0` for the policy distribution.
- Ensures we always pick the mating move (or opponent's salvaging move) regardless of initial FEN, without sampling noise.

### Uniform over multiple mating moves

In `get_policy_distribution` (`mcts_algorithm.py`), when `temperature=0`:

- If winning terminals exist (Q ≥ 0.99), use uniform probability over them so multiple mating moves get equal weight.
- Otherwise, use max visits with tie-break by smallest action_id.

### Shared selection logic

`MCTS.get_best_action_ids()` (and `get_best_action_deterministic()`) centralize the logic:
- Winning terminals (Q ≥ 0.99) first
- Else max visits
- Used by `get_policy_distribution` (temp=0), `get_best_action` (temp=0), and the diagnostic

### Diagnostic alignment

`test/diagnostic_mcts.py` uses `mcts_player.get_best_action_deterministic(root_node)`, which:

- Ensures mating move (checkmate) and opponent's salvaging INSUFFICIENT_MATERIAL when available.
- Uses deterministic selection: winning terminals (Q≥0.99) first, else max visits, tie-break by smallest action_id.

---

## 3. Draw reward table and position quality

When a game ends in a draw, the model uses `draw_reward_table` (if `draw_reward` is null) to assign reward by:

1. **Termination type** (STALEMATE, INSUFFICIENT_MATERIAL, etc.)
2. **Position quality** (winning / equal / losing) — derived from the *initial* position in mate training (side to move = winning)

For the **winning side** that blunders:

- **STALEMATE** (winning → -0.9): The side with the advantage stalemates instead of mating.
- **INSUFFICIENT_MATERIAL** (winning → -1.0 or -0.9): The winning side captures in a way that leaves insufficient material to mate.

For the **losing side** that salvages:

- **INSUFFICIENT_MATERIAL** (losing → 1.0): The opponent captures our last piece, salvaging a draw. MCTS pre-init gives `term_val = 1.0` so we force-select this move when simulating the opponent.

If `draw_reward` is set (e.g. 0.0), it overrides the table for all draws in `get_terminal_value_for_mcts`.

---

## 4. What gets stored (`last_n_moves_to_store`)

Only the **move that directly causes** the blunder matters — not earlier moves.

- **STALEMATE: 1** — The position where the blunderer is to move and plays the stalemating move.
- **INSUFFICIENT_MATERIAL: 2** (or 1–6) — The position(s) where the blunderer makes the capture that loses mating material. More moves (e.g. 2–6) can provide extra signal; 1 is minimal.

Earlier positions in the same game do not carry a useful signal; the model needs to learn "don't play this move in this position."

---

## 5. Sample imbalance (Insufficient material)

**Problem:** Win games store the **full game**. Draw games store only **last_n moves**. With ~70% wins:

- Win game (10 plies) → 10 samples in buffer
- Draw game (10 plies) → 5 samples (if last_n=5)
- Roughly 2–3× more training signal from wins than from blunder positions

**Fix:** Set `draw_sample_ratio` to oversample draws during training:

```yaml
draw_sample_ratio: 0.35  # ~35% of each batch from draw positions (incl. blunder positions)
```

---

## 6. Value vs policy training (no bug)

- Blunder position = last game_history entry before game ends (player to move is about to blunder).
- `last_n_moves_to_store` keeps blunder position in `game_data[-n:]`.
- Value: from `draw_reward_table` for "winning" (e.g. -0.9 or -1.0).
- Policy target: from MCTS at that position; draw move should get low mass.

The negative target is applied at the blunder position (two plies before game end), not at the opponent's last move.

---

## 7. Recommended config

```yaml
# config/train_mcts.yaml

mcts:
  pre_init_draws: true   # Stronger "avoid draw" signal in MCTS

training:
  draw_reward: 0.0      # Or null to use draw_reward_table
  draw_sample_ratio: 0.35   # Oversample draws so blunder positions appear more in batches
  last_n_moves_to_store:
    THREEFOLD_REPETITION: 9
    FIVEFOLD_REPETITION: 9
    STALEMATE: 1
    FIFTY_MOVES: 0
    SEVENTYFIVE_MOVES: 0
    MAX_MOVES: 0
    INSUFFICIENT_MATERIAL: 2  # Optional: 1 for minimal, 6 for extra signal

  # Optional: stronger penalty for blunders (requires draw_reward: null)
  # draw_reward_table:
  #   STALEMATE:
  #     winning: -1.0
  #     equal: -0.5
  #     losing: 0.0
  #   INSUFFICIENT_MATERIAL:
  #     winning: -1.0
  #     equal: -0.05
  #     losing: 1.0
```

---

## 8. Position quality for mate datasets

For mate-in-1 and mate-in-2 JSON, the side to move is the winning side. Because `minimal_endgames_*.json` may have no `quality` field, the code uses:

```python
initial_position_quality = 'winning' if white_to_move else 'losing'
```

which is correct for mate positions. You can add `"quality": "winning"` to each entry for explicitness.

---

## 9. Things to monitor

- **Draw rate:** Should decrease if learning works.
- **Mate success (m1, m2, mm1, mm2):** Ensure it doesn't drop (avoid over-penalizing).
- **Policy loss:** May rise slightly with more draw samples (they're harder).

---

## 10. BrokenPipeError on shutdown

The `BrokenPipeError` at the end of training comes from multiprocessing workers: the main process exits and closes the manager pipe before workers finish. It does not affect training results. Fixing it would require more careful shutdown (e.g. `stop_event.set()` and `process.join()` with a timeout) in the continual self-play worker logic.
