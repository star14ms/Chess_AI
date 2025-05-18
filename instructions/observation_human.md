# Observation Vector (8 x 8 Grid)

> Vector Shape: ```26 X 8 X 8```

- ```Color```: ```1 Dim```
    - Range: -1 or 0 or 1 (0 for Empty Tile)

- ```BehaviorType```: ```6 Dims``` 
    - Range: 0 or 1
    - (Each flag refers to Pawn, Knight, Bishop, Rook, Queen, and King)

- ```EnPassantTarget```: ```1 Dim```
    - Range: 0 or 1 (1 if this square is the target for an en passant capture on the next move, 0 otherwise)

- ```CastlingTarget```: ```1 Dim```
    - Range: 0 or 1 (1 if this square is the target that can involve in castling in next move, 0 otherwise)

- ```Current Player```: ```1 Dim```
    - Range: -1 or 1 (e.g., 1 for White's turn, -1 for Black's turn. Consistent across the 8x8 plane)

- ```Piece Identity```: ```16 Dims```
    - Range: 0 or 1
    - One-hot encoding for each piece's identity
    - If the square is occupied (i.e., ```Color``` is not 0), these 16 dimensions form a one-hot encoding that uniquely identifies one of up to 16 distinct pieces of that color
    - If the square is empty (i.e., ```Color``` is 0), this field is all zeros
