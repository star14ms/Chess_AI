# Chess Game State Observation Vector

This document defines the structure of the observation vector used to represent the state of the chess game for the AI agent. The goal is to capture all information necessary to determine legal moves and evaluate the position, consistent with the FEN standard.

## I. Board Representation (Piece Placement)

Instead of representing each piece individually with separate features like color, type, and position, we use a spatial representation common in machine learning models for board games. This involves stacking multiple 8x8 planes, each representing a specific aspect of the board.

*   **Total Planes:** 12 (minimum, can be extended with more features)
*   **Dimensions per Plane:** 8x8 = 64
*   **Representation:** Binary (1 for presence, 0 for absence)
*   **Total Board State Dimensions:** 12 * 64 = 768 (using the 12 planes below)

**Plane Breakdown (12 Planes):**

These planes indicate the presence (1) or absence (0) of a specific piece type for a specific color at each square.

1.  `White Pawns`: 8x8 binary map
2.  `White Knights`: 8x8 binary map
3.  `White Bishops`: 8x8 binary map
4.  `White Rooks`: 8x8 binary map
5.  `White Queens`: 8x8 binary map
6.  `White King`: 8x8 binary map
7.  `Black Pawns`: 8x8 binary map
8.  `Black Knights`: 8x8 binary map
9.  `Black Bishops`: 8x8 binary map
10. `Black Rooks`: 8x8 binary map
11. `Black Queens`: 8x8 binary map
12. `Black King`: 8x8 binary map

*(Alternative: A single 8x8 plane where each cell contains an integer from -6 (Black King) to +6 (White King), and 0 for empty squares. This has fewer dimensions (64) but might be less directly usable by certain neural network architectures like CNNs).*

*(Further Alternative: Could add planes for attack maps, move history, repetition counts, etc.)*

## II. Auxiliary State Information

This vector captures the game state aspects not directly represented on the board squares.

*   **Player Turn:**
    *   Dimensions: 1
    *   Value: `1` if White to move, `-1` (or `0`) if Black to move.
    *   *Purpose: Essential for knowing which pieces can move.*
*   **Castling Rights:**
    *   Dimensions: 4
    *   Values: Binary (`1` for available, `0` for unavailable) in a fixed order:
        1.  White Kingside (K)
        2.  White Queenside (Q)
        3.  Black Kingside (k)
        4.  Black Queenside (q)
    *   *Purpose: Determines legality of castling moves.*
*   **En Passant Target Square:**
    *   Dimensions: 64 (Option A - Spatial) or 1 (Option B - Scalar)
    *   **Option A (Spatial):** An 8x8 binary map. A single `1` marks the target square if an en passant capture is legal on the next half-move, `0`s otherwise.
    *   **Option B (Scalar):** An integer representing the target square index (0-63), or a special value (e.g., `-1`, `64`) if no en passant is possible.
    *   *Purpose: Determines legality of en passant captures.*
    *   *(Recommendation: Option A might integrate better with convolutional layers if the board state uses stacked planes).*
*   **Halfmove Clock:**
    *   Dimensions: 1
    *   Value: Integer count of halfmoves since the last capture or pawn advance (range: 0-100).
    *   *Purpose: Used for detecting draws by the 50-move rule.*
    *   *(Normalization Recommended: Often useful to normalize to a float [0.0, 1.0] by dividing by 100 for neural network stability).*
*   **Fullmove Number:**
    *   Dimensions: 1
    *   Value: Integer count of the full moves, starting at 1 and incrementing after Black's move.
    *   *Purpose: Primarily informational, can sometimes be used in evaluating game phase.*
    *   *(Normalization Optional: Can be used directly or potentially normalized if very long games are a concern).*

## III. Total Observation Vector Dimensions

The total size depends on the chosen representation for the En Passant square:

*   **Using Stacked Planes (Board) + Spatial En Passant (Aux):**
    *   768 (Board) + 1 (Turn) + 4 (Castling) + 64 (EP Square) + 1 (Halfmove) + 1 (Fullmove) = **839 dimensions**
*   **Using Stacked Planes (Board) + Scalar En Passant (Aux):**
    *   768 (Board) + 1 (Turn) + 4 (Castling) + 1 (EP Square) + 1 (Halfmove) + 1 (Fullmove) = **776 dimensions**
