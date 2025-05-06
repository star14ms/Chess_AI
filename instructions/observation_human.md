# Observation Vector (8 x 8 Grid)

> Vector Shape: ```8 X 8 X 14```

- ```Color```: ```1 Dim```
    - Range: -1 or 0 or 1 (0 for Empty Tile)

- ```BehaviorType```: ```6 Dims``` 
    - Range: 0 or 1
    - (Each flag refers to Pawn, Knight, Bishop, Rook, Queen, and King)

- ```EnPassantTarget```: ```1 Dim```
    - Range: 0 or 1 (1 if this square is the target for an en passant capture on the next move, 0 otherwise)

- ```CastliingTarget```: ```1 Dim```
    - Range: 0 or 1 (1 if this square is the target that can involve in castling in next move, 0 otherwise)

- ```Current Player```: ```1 Dim```
    - Range: -1 or 1 (e.g., 1 for White's turn, -1 for Black's turn. Consistent across the 8x8 plane.

- ```Piece ID```: ```4 Dims```
    - Each dimension is binary (Range: 0 or 1).
    - If the square is occupied (i.e., ```Color``` is not 0), these 4 dimensions form a binary code (e.g., 0000 to 1111) that uniquely identifies one of up to 16 distinct pieces of that color (including the King).
    - A specific mapping should be defined (e.g., King: 0000, Queen: 0001, Rook1: 0010, etc.).
    - If the square is empty (i.e., ```Color``` is 0), this field is all zeros.

# Reference Vector in Network
- Each pieces will be bound to specific actions in the action space

- ```Feature Vector```: ```V Dims```
    - Range: All Real Numbers
    - will store the feature of the vector of observation passing through the convolution layers.

- ```Position```: ```2 Dims``` 
    - Range: 1 to 8 for each dimension (Static vector used for integrating with Board Vector)