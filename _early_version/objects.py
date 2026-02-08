from typing import Optional, Union, List, Tuple


class Board:
    def __init__(self, f: bool) -> None:
        self.front = f #플레이어가 플레이 할 색깔. False면 흰색 폰을 움직였을 때 +방향으로 나아가고 흑색 폰을 움직이면 -방향으로 나아간다.
        self.board = [[Empty() for _ in range(8)] for _ in range(8)]
        for i in range(0, 8):
            for j in range(0, 8): self.board[i][j] = Empty()
        self.history: List[Piece] = [] # 기보 기록
        self.en_passant_target: Optional[Tuple[int, int]] = None  # Track the square where en passant capture is possible
        
    def delete(self, x: int, y: int) -> None:
        self.board[y][x] = Empty()

    def pos(self, x: int, y: int) -> 'Piece':
        return self.board[y][x]

    def insert(self, x: int, y: int, piece: 'Piece') -> None:
        self.board[y][x] = piece

    def move(self, x1: int, y1: int, x2: int, y2: int) -> None:
        # Handle en passant capture
        if self.en_passant_target and (x2, y2) == self.en_passant_target:
            # Remove the captured pawn
            captured_y = y1  # The captured pawn is on the same rank as the capturing pawn
            self.delete(x2, captured_y)
            
        # Set en passant target if a pawn moves two squares
        piece = self.pos(x1, y1)
        if isinstance(piece, Pawn):
            if abs(y2 - y1) == 2:  # If pawn moved two squares
                self.en_passant_target = (x2, (y1 + y2) // 2)  # The square the pawn passed over
            else:
                self.en_passant_target = None
        else:
            self.en_passant_target = None
            
        self.insert(x2, y2, self.pos(x1, y1))
        self.history.append(self.pos(x1,y1))#(x1,y1)좌표의 말 클래스를 기보에 기록한다.
        self.delete(x1, y1)

    def killable(self, x1: int, y1: int, x2: int, y2: int) -> bool:
        # Regular capture
        if (not isinstance(self.pos(x2,y2), Empty)) and (self.pos(x1,y1).color != self.pos(x2, y2).color):
            return True
            
        # En passant capture
        if (isinstance(self.pos(x1,y1), Pawn) and  # Capturing piece must be a pawn
            self.en_passant_target and  # There must be an en passant target
            (x2, y2) == self.en_passant_target and  # Target square must be en passant target
            abs(x2 - x1) == 1 and  # Must move diagonally
            abs(y2 - y1) == 1):  # Must move diagonally
            return True
            
        return False

    def is_safe(self, x2: int, y2: int, color: int) -> bool: # x2, y2 자리가 공격받는 자리인가 판단
        for y1 in range(8): # 보드 전체를 검사해
            for x1 in range(8): # x2, y2 좌표에 있는 말을 잡을 수 있는 상대 기물이 있는지 검사
                if (self.pos(x1, y1).color == color*-1) and (self.pos(x1, y1)._is_move_valid(self, x2, y2)): ### self.board -> self
                    return False

        return True

    def can_defend(self, x: int, y: int, color: int) -> bool: # x, y 자리를 방어할 수 있는지 검사

        board2 = self.copy_board()
        being_attacked = True

        for y1 in range(8): # 보드 전체를 검사해 내 기물들을 찾고
            for x1 in range(8): 
                if (self.pos(x1, y1).color == color):
                     
                    for y2 in range(8): 
                        for x2 in range(8):
                            
                            # backup pieces
                            xy1_piece = board2.pos(x1, y1)
                            xy2_piece = board2.pos(x2, y2)

                            if xy1_piece.movable(board2, x2, y2): # 그 기물이 움직일 수 있는 모든 경우를 
                                board2.insert(x2, y2, board2.pos(x1, y1))
                                board2.delete(x1, y1)

                                # 그 기물이 움직였을때 x, y에 있는 기물이 안전해질 수 있는 경우가 있나 검사
                                if (x == x1) and (y == y1): # 1. 공격받는 자신이 직접 자리를 피하기
                                    is_safe = board2.is_safe(x2, y2, color)
                                else:                       # 2. 다른 기물을 움직여 방패세우기
                                    is_safe = board2.is_safe(x, y, color) ### self. -> board2.

                                if is_safe:
                                    being_attacked = False

                                # restore pieces
                                board2.insert(x1, y1, xy1_piece)
                                board2.insert(x2, y2, xy2_piece)
                                
        if not being_attacked:
            return True
        return False
       
    def copy_board(self) -> 'Board':
        """Create a deep copy of the board with new piece instances"""
        new_board = Board(-1)
        for y in range(8):
            for x in range(8):
                piece = self.pos(x, y)
                if not isinstance(piece, Empty):
                    # Create new instance of the same piece type
                    new_piece = type(piece)(new_board, x, y, piece.color)
                    if isinstance(piece, Pawn):
                        new_piece.first_turn = piece.first_turn
                    elif isinstance(piece, (Rook, King)):
                        new_piece.moved = piece.moved
                    new_board.insert(x, y, new_piece)
                else:
                    new_board.insert(x, y, Empty())
        return new_board


class Piece:
    def __init__(self, board: Board, x: int, y: int, color: int) -> None:
        """Initialize a chess piece"""
        self.p_x = x
        self.p_y = y
        self.color = color  # -1 -> white, 1 -> black
        board.insert(x, y, self)

        # Initialize piece-specific attributes
        if isinstance(self, Pawn):
            self.first_turn = True
        elif isinstance(self, (Rook, King)):
            self.moved = False

    def move(self, board: Board, x2: int, y2: int) -> bool:
        """Move piece to position (x2,y2)"""
        if self.p_x == x2 and self.p_y == y2:
            return False

        movable = self.movable(board, x2, y2)
        if not movable:
            return False

        # Move the piece
        if board.killable(self.p_x, self.p_y, x2, y2):
            board.move(self.p_x, self.p_y, x2, y2)
            self.p_x = x2
            self.p_y = y2
        elif not isinstance(board.pos(x2, y2), King):
            board.move(self.p_x, self.p_y, x2, y2)
            self.p_x = x2
            self.p_y = y2

        # Handle castling
        if isinstance(self, King):
            if movable == "White_Queen_Side_Castling":
                board.move(0, y2, 3, y2)
                board.board[y2][3].p_x = 3
                board.board[y2][3].p_y = y2
            elif movable == "Black_Queen_Side_Castling":
                board.move(0, y2, 3, y2)
                board.board[y2][3].p_x = 3
                board.board[y2][3].p_y = y2
            elif movable == "White_King_Side_Castling":
                board.move(7, y2, 5, y2)
                board.board[y2][5].p_x = 5
                board.board[y2][5].p_y = y2
            elif movable == "Black_King_Side_Castling":
                board.move(7, y2, 5, y2)
                board.board[y2][5].p_x = 5
                board.board[y2][5].p_y = y2

        # Update moved status for castling eligibility
        if isinstance(self, (Rook, King)):
            self.moved = True
        
        return True

    def _is_king_safe_after_move(self, board: Board, x2: int, y2: int, is_en_passant: bool = False) -> bool:
        """Check if the king would be safe after a move"""
        # Store original state
        original_piece = board.board[y2][x2]
        captured_pawn = None
        if is_en_passant:
            captured_pawn = board.board[y2][self.p_y]  # The pawn being captured en passant
        
        # Make the move
        board.board[y2][x2] = self
        board.board[self.p_y][self.p_x] = Empty()
        if is_en_passant:
            board.board[y2][self.p_y] = Empty()  # Remove the captured pawn
        
        original_x, original_y = self.p_x, self.p_y
        self.p_x, self.p_y = x2, y2
        
        # Check if king is safe
        king_safe = True
        for y in range(8):
            for x in range(8):
                piece = board.board[y][x]
                if piece and piece.color == self.color and isinstance(piece, King):
                    if not board.is_safe(x, y, self.color):
                        king_safe = False
                        break
            if not king_safe:
                break

        # Restore original state
        board.board[original_y][original_x] = self
        board.board[y2][x2] = original_piece
        if is_en_passant:
            board.board[y2][self.p_y] = captured_pawn
        self.p_x, self.p_y = original_x, original_y
        
        return king_safe

    def movable(self, board: Board, x2: int, y2: int) -> Union[bool, str]:
        """Check if the piece can move to (x2,y2)"""
        # Phase 1: Check if the move is valid for this piece type
        move_valid = self._is_move_valid(board, x2, y2)
        if not move_valid:
            return False

        # For castling, return the castling type string
        if isinstance(move_valid, str):
            return move_valid

        # Phase 2: Check king's safety
        return self._is_king_safe_after_move(board, x2, y2)

    def _is_move_valid(self, board: Board, x2: int, y2: int) -> Union[bool, str]:
        """Check if the move is valid for this piece type"""
        raise NotImplementedError("Subclasses must implement _is_move_valid")

    def _is_valid_square(self, x: int, y: int) -> bool:
        """Check if coordinates are within board bounds"""
        return 0 <= x <= 7 and 0 <= y <= 7

    def _is_empty_or_enemy(self, board: Board, x: int, y: int) -> bool:
        """Check if square is empty or contains enemy piece"""
        piece = board.pos(x, y)
        return isinstance(piece, Empty) or piece.color != self.color

    def _is_path_clear(self, board: Board, x1: int, y1: int, x2: int, y2: int) -> bool:
        """Check if path between two squares is clear"""
        dx = x2 - x1
        dy = y2 - y1
        steps = max(abs(dx), abs(dy))
        
        if steps == 0:
            return True
            
        x_step = dx // steps
        y_step = dy // steps
        
        for i in range(1, steps):
            x = x1 + i * x_step
            y = y1 + i * y_step
            if not isinstance(board.pos(x, y), Empty):
                return False
        return True


class Empty:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.color = 0
        return cls._instance


class Pawn(Piece):
    def _is_move_valid(self, board: Board, x2: int, y2: int) -> bool:
        """Check if pawn move is valid"""
        if not self._is_valid_square(x2, y2):
            return False

        dx = x2 - self.p_x
        dy = y2 - self.p_y
        direction = -1 if board.front == self.color else 1

        # Moving forward
        if dx == 0:
            # One square forward
            if dy == direction and isinstance(board.pos(x2, y2), Empty):
                return True
            # Two squares forward on first move
            if (self.p_y == (6 if board.front == self.color else 1) and 
                dy == 2 * direction and 
                isinstance(board.pos(x2, y2), Empty) and 
                isinstance(board.pos(x2, self.p_y + direction), Empty)):
                return True
            return False

        # Capturing diagonally
        if abs(dx) == 1 and dy == direction:
            # Regular capture
            if not isinstance(board.pos(x2, y2), Empty) and board.pos(x2, y2).color != self.color:
                return True
            # En passant capture
            if board.en_passant_target and (x2, y2) == board.en_passant_target:
                return True

        return False

    def movable(self, board: Board, x2: int, y2: int) -> Union[bool, str]:
        """Check if pawn can move to (x2,y2)"""
        if not self._is_move_valid(board, x2, y2):
            return False
        
        # Handle en passant captures separately
        if board.en_passant_target and (x2, y2) == board.en_passant_target:
            return self._is_king_safe_after_move(board, x2, y2, is_en_passant=True)
        
        # For regular moves, use the parent class's king safety check
        return super().movable(board, x2, y2)


class Bishop(Piece):
    def _is_move_valid(self, board: Board, x2: int, y2: int) -> bool:
        """Check if bishop move is valid"""
        if not self._is_valid_square(x2, y2):
            return False

        dx = x2 - self.p_x
        dy = y2 - self.p_y

        # Must move diagonally
        if abs(dx) != abs(dy):
            return False

        # Target square must be empty or enemy
        if not self._is_empty_or_enemy(board, x2, y2):
            return False

        return self._is_path_clear(board, self.p_x, self.p_y, x2, y2)


class Rook(Piece):
    def _is_move_valid(self, board: Board, x2: int, y2: int) -> bool:
        """Check if rook move is valid"""
        if not self._is_valid_square(x2, y2):
            return False

        dx = x2 - self.p_x
        dy = y2 - self.p_y

        # Must move horizontally or vertically
        if dx != 0 and dy != 0:
            return False

        # Target square must be empty or enemy
        if not self._is_empty_or_enemy(board, x2, y2):
            return False

        return self._is_path_clear(board, self.p_x, self.p_y, x2, y2)


class Knight(Piece):
    def _is_move_valid(self, board: Board, x2: int, y2: int) -> bool:
        """Check if knight move is valid"""
        if not self._is_valid_square(x2, y2):
            return False

        dx = abs(x2 - self.p_x)
        dy = abs(y2 - self.p_y)

        # Must move in L-shape
        if not ((dx == 2 and dy == 1) or (dx == 1 and dy == 2)):
            return False

        # Target square must be empty or enemy
        return self._is_empty_or_enemy(board, x2, y2)


class Queen(Piece):
    def _is_move_valid(self, board: Board, x2: int, y2: int) -> bool:
        """Check if queen move is valid"""
        if not self._is_valid_square(x2, y2):
            return False

        dx = x2 - self.p_x
        dy = y2 - self.p_y

        # Must move diagonally, horizontally, or vertically
        if dx != 0 and dy != 0 and abs(dx) != abs(dy):
            return False

        # Target square must be empty or enemy
        if not self._is_empty_or_enemy(board, x2, y2):
            return False

        return self._is_path_clear(board, self.p_x, self.p_y, x2, y2)


class King(Piece):
    def _is_move_valid(self, board: Board, x2: int, y2: int) -> Union[bool, str]:
        """Check if king move is valid"""
        if not self._is_valid_square(x2, y2):
            return False

        dx = x2 - self.p_x
        dy = y2 - self.p_y

        # Regular king move
        if abs(dx) <= 1 and abs(dy) <= 1:
            return self._is_empty_or_enemy(board, x2, y2)

        # Castling
        if not self.moved and dy == 0 and abs(dx) == 2:
            # King-side castling
            if dx == 2:
                if (isinstance(board.pos(self.p_x+3, self.p_y), Rook) and 
                    not board.pos(self.p_x+3, self.p_y).moved and
                    isinstance(board.pos(self.p_x+1, self.p_y), Empty) and
                    isinstance(board.pos(self.p_x+2, self.p_y), Empty) and
                    not self.checked(board) and
                    board.is_safe(self.p_x+1, self.p_y, self.color) and
                    board.is_safe(self.p_x+2, self.p_y, self.color)):
                    return "White_King_Side_Castling" if self.color == -1 else "Black_King_Side_Castling"

            # Queen-side castling
            if dx == -2:
                if (isinstance(board.pos(self.p_x-4, self.p_y), Rook) and
                    not board.pos(self.p_x-4, self.p_y).moved and
                    isinstance(board.pos(self.p_x-1, self.p_y), Empty) and
                    isinstance(board.pos(self.p_x-2, self.p_y), Empty) and
                    isinstance(board.pos(self.p_x-3, self.p_y), Empty) and
                    not self.checked(board) and
                    board.is_safe(self.p_x-1, self.p_y, self.color) and
                    board.is_safe(self.p_x-2, self.p_y, self.color)):
                    return "White_Queen_Side_Castling" if self.color == -1 else "Black_Queen_Side_Castling"

        return False
    
    def checked(self, board: Board) -> bool:
        """Check if king is in check"""
        return not board.is_safe(self.p_x, self.p_y, self.color)

    def state(self, board: Board) -> str:
        """Determine game state (check, checkmate, stalemate)"""
        check = self.checked(board)
        next_turn_check = not board.can_defend(self.p_x, self.p_y, self.color)

        if check and not next_turn_check:
            return "Check"
        elif check and next_turn_check:
            return "Checkmate"
        elif not check and next_turn_check:
            return "Stalemate"
        return "Normal"
