# Observation: 

> How to represent the state with less memory? 


## Piece Information (32 Pieces in Total on the Board)


### Basic Information (Vector Shape: ```32 x 9```)

- ```Color```: ```1 Dim```
    - Range: -1 or 0 or 1 (0 for Empty Tile)
- ```PieceType```: ```6 Dims``` 
    - Range: 0 or 1
- ```Position```: ```2 Dims``` 
    - Range(1 to 8)

### King, Rook, Pawn Specific Information (Vector Shape ```(1+2+8) X 1)```
- ```Moved```: ```1 Dim```
    - Range: 0 or 1
    - Used for Castling and Two-Space Forward Move of Pawn, En Passant Availability

### Pawn Specific Information (Vector Shape ```8 X 5```
- ```BehaviorType```: ```5 Dims ```
    - Range: 0 or 1
    - Five Roles that can be changed: Pawn, Knight, Rook, Bishop, and Queen


## Board Information (8 X 8 Grid)
> Vector Shape: ```8 X 8 X 13``` 
- ```Width X Height X (Color + 6 PieceTypes + Moved + 5 BehaviorTypes)```
