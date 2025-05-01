# Summary: Chess Action Space Definition Discussion

This document summarizes the discussion regarding the definition of the chess action space for reinforcement learning.

1.  **Initial Request & First Approach:**
    *   Goal: Map UCI move strings to action IDs.
    *   First Implementation: IDs 1-N assigned clockwise (N, E, S, W), shortest moves first *within* each direction. Resulted in inconsistent ID ranges based on piece position.

2.  **Correction to Relative IDs:**
    *   Refinement: A specific move type (e.g., "1 square East") should have a consistent ID regardless of start square.
    *   Action: Adopted fixed ID ranges based on relative direction/distance (e.g., Rook N=1-7, E=8-14, S=15-21, W=22-28).

3.  **Implementation (Relative Scheme):**
    *   Action: Implemented `uci_to_*_action_id` functions for R, N, B, Q, K using the relative scheme.
    *   Action: Developed a preliminary Pawn function mapping basic moves/Q+N promotions (IDs 1-62).
    *   Action: Created `board_moves_to_action_ids` helper for visualizing moves of a specific piece type.

4.  **Identifying the Conflict:**
    *   Observation: A conflict was noted between a theoretical max action count (~754, including all piece instances) and the implemented functions' limited output range (max ~62).
    *   Causes: Lack of distinction between multiple instances of the same piece type (e.g., both white rooks shared IDs 1-28) and incomplete handling of all move types (castling, full promotions).

5.  **Introducing the Absolute Action Space:**
    *   Concept: Shifted to an Absolute Action Space where each unique piece instance (e.g., White Rook 0, Black Pawn 7) gets a dedicated ID range.
    *   Formula: `Absolute ID = Base ID + Relative ID - 1`.
    *   Requirement: Needed a `Base ID` calculation per instance and refined `Relative ID` calculations per move type.

6.  **First Absolute Space Proposal (548 Actions):**
    *   Design: Assumed standard piece counts (2R, 2N, 2B, 1Q, 1K, 8P per side) and 10 relative pawn actions (incl. Q/N promotions).
    *   Result: 548 total absolute actions.
    *   Action: Modified `get_all_legal_move_action_ids` to use this scheme internally.

7.  **User's "70 Pawn Actions" Calculation:**
    *   User Input: Proposed a calculation where each pawn represents ~70 actions (basic moves + full embedded Q moves + full embedded N moves) to anticipate future promotions.

8.  **Discussion: Standard RL vs. Embedded Futures:**
    *   Analysis: Discussed why embedding future Q/N move sets into the pawn's current action space (~82 actions/pawn) is problematic for standard RL due to sparsity (most moves illegal), redundancy (promoted Queen also needs IDs), and ambiguity (meaning of illegal move predictions).
    *   Standard Approach Explained: Handle promotion as a specific pawn action leading to a state change; the *new* piece uses its own dedicated action IDs in the *next* state.

9.  **Refined Standard Approach (Max Instances, ~2596 Actions):**
    *   Proposed Solution: Use 18 relative pawn actions (incl. all R/N/B/Q promotions, distinct EP) and an absolute action space sized for the *maximum theoretical* piece instances (10R, 10N, 10B, 9Q, 1K per side).
    *   Requirement: Robust instance tracking in the environment/main loop.
    *   Result: ~2596 total absolute actions. Aligns with standard practice, cleaner logic, but larger space.

10. **User's "Pawn-Forever" Concept (~1636 Actions):**
    *   User Preference: Reiterate the idea of keeping pawns conceptually as pawns but conditionally enabling embedded Q/N move sets upon promotion (~78 relative actions/pawn).
    *   Benefit: Smaller total action space (~1636) by avoiding slots for max theoretical pieces.
    *   Challenge: Requires significant custom logic for conditional action masking and state handling, deviating from standard piece transformation models.

11. **Outcome:**
    *   Two main paths identified:
        *   **Standard (~2596 actions):** Larger space, standard logic, explicit piece transformation.
        *   **"Pawn-Forever" (~1636 actions):** Smaller space, complex custom conditional logic.
    *   The user indicated a preference for the "Pawn-Forever" approach, accepting the associated implementation complexity to achieve a smaller action space.
