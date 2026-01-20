AlphaZero Differences (Codebase vs Paper)
=========================================

This document summarizes non-hyperparameter differences between this codebase
and the standard AlphaZero chess setup. It focuses on algorithmic behavior and
data representation rather than tuning values.

1) Value Target Construction (Resolved)
   - AlphaZero (paper): value head is trained on the final game outcome z.
   - This codebase: value target uses the MCTS root value Q(s) stored per state.
     Positions without an MCTS value are skipped (no outcome fallback).

2) Replay Buffer Policy (Resolved)
   - AlphaZero: uniform replay buffer with FIFO eviction.
   - This codebase: uniform replay buffer with FIFO eviction.

3) Draw Handling and Termination
   - AlphaZero: draws are 0 and game termination follows standard rules
     (no artificial max-move cutoff beyond the 50-move rule).
   - This codebase: explicit MAX_MOVES termination and repetition tracking
     (threefold/fivefold) can end games early. Draw rewards are configured
     and used for logging/statistics even if value targets use MCTS Q.

4) Illegal Move Handling (Resolved)
   - AlphaZero: assumes perfect legality; no illegal-move penalties needed.
   - This codebase: enforces legal moves by construction and no longer applies
     foul states or illegal-move penalties; only optional monitoring metrics remain.

5) Self-Play Scheduling
   - AlphaZero: sequential loop of self-play -> training -> evaluation.
   - This codebase: supports continual self-play with multiprocessing and
     queue-based collection, mixing device contributions.

6) Temperature Schedule
   - AlphaZero: temperature = 1 for first N moves, then near-deterministic.
   - This codebase: temperature decays continuously across moves.

7) Observation Planes (Same Count, Different Content Details)
   - Planes count matches AlphaZero-style input (119 channels = 8 history
     steps * 14 planes + 7 meta planes).
   - Differences in plane content:
     * Current player plane uses +1 (white) / -1 (black) instead of binary.
     * Includes normalized fullmove number as a meta plane.
     * Repetition planes are computed over the history window, not full game.

8) Action Encoding
   - Both use 4672 actions (8x8x73-style mapping). Verify exact move-plane
     mapping if strict identity to AlphaZero is required.

Notes
-----
- These items are based on current code in MCTS/train.py, chess_gym/chess_custom.py,
  and related training utilities.
- Hyperparameters (e.g., c_puct, iterations, batch sizes) are intentionally
  excluded from this list.
