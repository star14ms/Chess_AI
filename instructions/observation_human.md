# Observation Vector (8 x 8 Grid)

> Vector Shape: ```8 X 8 X 11```

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

- ```Piece ID```: ```1 Dim```
    - Range: 1 to 32

# Reference Vector in Network
- Each pieces will be bound to specific actions in the action space

- ```Feature Vector```: ```V Dims```
    - Range: All Real Numbers
    - will store the feature of the vector of observation passing through the convolution layers.

- ```Position```: ```2 Dims``` 
    - Range: 1 to 8 for each dimension (Static vector used for integrating with Board Vector)