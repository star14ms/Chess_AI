class Board:
    def __init__(self, f):
        self.front = f #플레이어가 플레이 할 색깔. False면 흰색 폰을 움직였을 때 +방향으로 나아가고 흑색 폰을 움직이면 -방향으로 나아간다.
        self.board = [[0 for col in range(8)] for row in range(8)]
        for i in range(0, 8):
            for j in range(0, 8): self.board[i][j] = Empty()
        self.history = []#기보 기록
        self.en_passant_target = None  # Track the square where en passant capture is possible
        
    def delete(self, x, y):#x,y 좌표의 말 삭제
        self.board[y][x] = Empty()

    def pos(self, x, y):#x, y좌표의 말 클래스 출력
        return self.board[y][x]

    def insert(self, x, y, horse):#x, y좌표의 말 입력
        self.board[y][x] = horse

    def move(self, x1, y1, x2, y2):#(x1, y1) -> (x2, y2)로 말의 이동. 색깔 상관없이 기존 (x2, y2)의 말을 지워버리니 주의할 것
        # Handle en passant capture
        if self.en_passant_target and (x2, y2) == self.en_passant_target:
            # Remove the captured pawn
            captured_y = y1  # The captured pawn is on the same rank as the capturing pawn
            self.delete(x2, captured_y)
            
        # Set en passant target if a pawn moves two squares
        piece = self.pos(x1, y1)
        if type(piece) == Pawn:
            if abs(y2 - y1) == 2:  # If pawn moved two squares
                self.en_passant_target = (x2, (y1 + y2) // 2)  # The square the pawn passed over
            else:
                self.en_passant_target = None
        else:
            self.en_passant_target = None
            
        self.insert(x2, y2, self.pos(x1, y1))
        self.pos(x1,y1).horse_history.append((x1, y1))#자신의 좌표 튜플을 기록한다.
        self.history.append(self.pos(x1,y1))#(x1,y1)좌표의 말 클래스를 기보에 기록한다.
        self.delete(x1, y1)

    def killable(self, x1, y1, x2, y2):#(x2,y2)에 말이 있고 색깔이 다르면 True, 아니면 False 출력
        # Regular capture
        if (type(self.pos(x2,y2)) != Empty) and (self.pos(x1,y1).color != self.pos(x2, y2).color):
            return True
            
        # En passant capture
        if (type(self.pos(x1,y1)) == Pawn and  # Capturing piece must be a pawn
            self.en_passant_target and  # There must be an en passant target
            (x2, y2) == self.en_passant_target and  # Target square must be en passant target
            abs(x2 - x1) == 1 and  # Must move diagonally
            abs(y2 - y1) == 1):  # Must move diagonally
            return True
            
        return False

    def is_safe(self, x2: int, y2: int, color: int): # x2, y2 자리가 공격받는 자리인가 판단
        for y1 in range(8): # 보드 전체를 검사해
            for x1 in range(8): # x2, y2 좌표에 있는 말을 잡을 수 있는 상대 기물이 있는지 검사
                if (self.pos(x1, y1).color == color*-1) and (self.pos(x1, y1)._is_move_valid(self, x2, y2)): ### self.board -> self
                    return False

        return True

    def can_defend(self, x, y, color): # x, y 자리를 방어할 수 있는지 검사

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
       
    def copy_board(self):
        """Create a deep copy of the board with new piece instances"""
        new_board = Board(-1)
        for y in range(8):
            for x in range(8):
                piece = self.pos(x, y)
                if type(piece) != Empty:
                    # Create new instance of the same piece type
                    new_piece = type(piece)(new_board, x, y, piece.color)
                    if type(piece) == Pawn:
                        new_piece.first_turn = piece.first_turn
                    elif type(piece) in [Rook, King]:
                        new_piece.moved = piece.moved
                    new_piece.horse_history = piece.horse_history.copy()
                    new_board.insert(x, y, new_piece)
                else:
                    new_board.insert(x, y, Empty())
        return new_board


class Piece:

    def __init__(self, board, x, y, color):

        # 말의 기본 정보
        self.p_x = x
        self.p_y = y
        self.color = color # -1 -> 백, 1 -> 흑
        self.horse_history = [(self.p_x, self.p_y)]#말 기록
        board.insert(x, y, self)

        # 특정 말의 추가 정보
        if (type(self) == Pawn):
            self.first_turn = True
        elif (type(self) == Rook) or (type(self) == King):
            self.moved = False

    def move(self, board, x2, y2, force=False):
        if self.p_x == x2 and self.p_y == y2:
            return False

        if not force:
            movable = self.movable(board, x2, y2)
            if not movable:
                return False

        # 말 움직이기
        if board.killable(self.p_x, self.p_y, x2, y2) :#인공지능 활용을 위해 남겨둠
            board.move(self.p_x, self.p_y, x2, y2)
            self.p_x = x2
            self.p_y = y2
        elif type(board.pos(x2, y2)) != King:
            board.move(self.p_x, self.p_y, x2, y2)
            self.p_x = x2
            self.p_y = y2

        # 캐슬링 시, 룩도 이동
        if not force and type(self) == King:
            if movable == "White_Queen_Side_Castling":
                board.move(0, 7, 3, 7)
            elif movable == "Black_Queen_Side_Castling":
                board.move(0, 0, 3, 0)
            elif movable == "White_King_Side_Castling":
                board.move(7, 7, 5, 7)
            elif movable == "Black_King_Side_Castling":
                board.move(7, 0, 5, 0)

        # 캐슬링 조건에 쓰이는 정보 업데이트 (킹과 룩이 움직인 적이 없어야 함)
        if (type(self) == Rook) or (type(self) == King):
            self.moved = True
        
        return True

    def _is_king_safe_after_move(self, board: Board, x2: int, y2: int, is_en_passant: bool = False) -> bool:
        """Check if the king would be safe after a move.
        
        Args:
            board: The current board state
            x2, y2: The target square for the move
            is_en_passant: Whether this is an en passant capture
            
        Returns:
            bool: True if the king would be safe after the move, False otherwise
        """
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

    def movable(self, board: Board, x2: int, y2: int):
        # Phase 1: Check if the move is valid for this piece type
        if not self._is_move_valid(board, x2, y2):
            return False

        # Phase 2: Check king's safety
        return self._is_king_safe_after_move(board, x2, y2)

    def _is_move_valid(self, board, x2, y2):
        # This method should be overridden by each piece type
        # to implement their specific movement rules
        raise NotImplementedError("Subclasses must implement _is_move_valid")


class Empty:
    color = 0
    horse_history = [(-1,-1)]


class Pawn(Piece):

    def _is_move_valid(self, board, x2, y2):
        # Basic coordinate validation
        if not (0 <= x2 <= 7 and 0 <= y2 <= 7):
            return False

        # Check if moving forward
        if board.front == self.color:
            # Moving straight forward
            if self.p_x == x2 and type(board.pos(x2, y2)) == Empty:
                # First move can be 1 or 2 squares
                if self.p_y == 6 and (1 <= self.p_y - y2 <= 2):
                    return True
                # Subsequent moves can only be 1 square
                elif self.p_y - y2 == 1:
                    return True
            
            # Capturing diagonally
            elif abs(self.p_x - x2) == 1 and self.p_y - y2 == 1:
                # Regular capture
                if type(board.pos(x2, y2)) != Empty and board.pos(x2, y2).color != self.color:
                    return True
                # En passant capture
                elif board.en_passant_target and (x2, y2) == board.en_passant_target:
                    return True

        else:
            # Moving straight forward (black's perspective)
            if self.p_x == x2 and type(board.pos(x2, y2)) == Empty:
                # First move can be 1 or 2 squares
                if self.p_y == 1 and (-2 <= self.p_y - y2 <= -1):
                    return True
                # Subsequent moves can only be 1 square
                elif self.p_y - y2 == -1:
                    return True
            
            # Capturing diagonally
            elif abs(self.p_x - x2) == 1 and self.p_y - y2 == -1:
                # Regular capture
                if type(board.pos(x2, y2)) != Empty and board.pos(x2, y2).color != self.color:
                    return True
                # En passant capture
                elif board.en_passant_target and (x2, y2) == board.en_passant_target:
                    return True
        
        return False

    def movable(self, board: Board, x2: int, y2: int):
        # First check if the move is valid for this piece type
        if not self._is_move_valid(board, x2, y2):
            return False
        
        # Handle en passant captures separately
        if board.en_passant_target and (x2, y2) == board.en_passant_target:
            return self._is_king_safe_after_move(board, x2, y2, is_en_passant=True)
        
        # For regular moves, use the parent class's king safety check
        return super().movable(board, x2, y2)


class Bishop(Piece):

    def _is_move_valid(self, board, x2, y2):
        # Check if the move is within bounds and diagonal
        if (not 0 <= x2 <= 7 or not 0 <= y2 <= 7 or not abs(self.p_x - x2) == abs(self.p_y - y2)): 
            return False
            
        # Check if target square is not occupied by a friendly piece
        if board.pos(x2, y2).color == self.color:
            return False

        amount = abs(x2 - self.p_x) #거리
        if x2-self.p_x < 0 : lr = -1 #lr : -1, +1 오른쪽 위면 lr = 1, du = -1
        else : lr = 1
        if y2-self.p_y < 0 : du = -1 #ud : -1, +1
        else : du = 1
        
        # Check all squares along the path except the final square
        for i in range(1, amount):
            x3 = self.p_x + i * lr
            y3 = self.p_y + i * du
            if (type(board.pos(x3, y3)) != Empty):
                return False
                
        return True


class Rook(Piece):
    
    def _is_move_valid(self, board, x2, y2):
        if (not 0 <= x2 <= 7 or not 0 <= y2 <= 7 or board.pos(x2,y2).color == self.color): 
            return False
        if x2-self.p_x == y2-self.p_y == 0 : 
            return False

        if (x2 == self.p_x):
            am = abs(self.p_y - y2)#거리
            for i in range(1, am):
                if self.p_y < y2:
                    if type(board.pos(x2, self.p_y+i)) != Empty : 
                        return False
                else :
                    if type(board.pos(x2, self.p_y-i)) != Empty : 
                        return False
        elif (y2-self.p_y == 0):
            am = abs(self.p_x - x2)#거리
            for i in range(1, am):
                if self.p_x < x2:
                    if type(board.pos(self.p_x+i, y2)) != Empty : 
                        return False
                else :
                    if type(board.pos(self.p_x-i, y2)) != Empty : 
                        return False
        else: 
            return False
        return True


class Knight(Piece):

    def _is_move_valid(self, board, x2, y2):
        # 나이트 기본 행마
        if (x2-self.p_x == 2) or (x2-self.p_x == -2): # 동쪽 or 서쪽으로 2칸일 때
            if (y2-self.p_y != 1) and (y2-self.p_y != -1): 
                return False # 남쪽 or 북쪽으로 1칸이 아니면, 이동 실패
        elif (y2-self.p_y == 2) or (y2-self.p_y == -2): # 남쪽 or 북쪽으로 2칸일 때
            if (x2-self.p_x != 1) and (x2-self.p_x != -1): 
                return False # 동쪽 or 서쪽으로 1칸이 아니면, 이동 실패
        else:
            return False

        if (not 0 <= x2 <= 7 or not 0 <= y2 <= 7) or (board.pos(x2, y2).color == self.color): 
            return False # 좌표 범위 밖이거나, 우리편과 겹치나 검사

        return True


class Queen(Piece):

    def _is_move_valid(self, board, x2, y2):
        lr = 1 if (x2-self.p_x > 0) else (-1 if (x2-self.p_x < 0) else 0) # x 이동 방향 right(1), left(-1), 0(None)
        ud = 1 if (y2-self.p_y > 0) else (-1 if (y2-self.p_y < 0) else 0) # y 이동 방향 down(1), up(-1), 0(None)

        # 대각선, 가로, 세로 방향 중 하나인가 검사
        if ((abs(x2-self.p_x) == abs(y2-self.p_y)) and (lr != 0)) or ((lr != 0) and (ud == 0)) or ((lr == 0) and (ud != 0)):
            
            amount = abs(x2-self.p_x) if (x2-self.p_x != 0) else abs(y2-self.p_y)
            x_focus, y_focus = self.p_x, self.p_y

            for i in range(1, amount): # 시점을 한 칸씩 이동시키며 빈 곳인지 검사
                x_focus = x_focus + lr
                y_focus = y_focus + ud
                if (type(board.pos(x_focus, y_focus)) != Empty):
                    return False
        else: 
            return False

        if (not 0 <= x2 <= 7 or not 0 <= y2 <= 7) or (board.pos(x2, y2).color == self.color): 
            return False # 좌표 범위 밖이거나, 우리편과 겹치나 조사
            
        return True


class King(Piece):

    def _is_move_valid(self, board, x2, y2):
        # 킹 기본 행마
        if (((x2-self.p_x == -1) or (x2-self.p_x == +1)) and (-1 <= y2-self.p_y <= 1)) or (
            ((y2-self.p_y == -1) or (y2-self.p_y == +1)) and (x2-self.p_x == 0)):
            if (not 0 <= x2 <= 7 or not 0 <= y2 <= 7) or (board.pos(x2, y2).color == self.color): 
                return False # 좌표 범위 밖이거나, 우리편과 겹치나 조사
            return True

        # 캐슬링
        elif (not self.moved):
            # 킹 사이드 캐슬링
            if (x2-self.p_x == +2) and (y2 == self.p_y) and (
                type(board.pos(self.p_x+3, self.p_y)) == Rook and not board.pos(self.p_x+3, self.p_y).moved) and (
                type(board.pos(self.p_x+1, self.p_y)) == Empty) and (type(board.pos(self.p_x+2, self.p_y)) == Empty) and (
                not self.checked(board)) and (
                board.is_safe(self.p_x+1, self.p_y, self.color)) and (
                board.is_safe(self.p_x+2, self.p_y, self.color)):
                
                if self.color == -1:
                    return "White_King_Side_Castling"
                else:
                    return "Black_King_Side_Castling"

            # 퀸 사이드 캐슬링
            elif (x2-self.p_x == -2) and (y2 == self.p_y) and (
                type(board.pos(self.p_x-4, self.p_y)) == Rook and not board.pos(self.p_x-4, self.p_y).moved) and (
                type(board.pos(self.p_x-1, self.p_y)) == Empty) and (type(board.pos(self.p_x-2, self.p_y)) == Empty) and (type(board.pos(self.p_x-3, self.p_y)) == Empty) and (
                not self.checked(board)) and (
                board.is_safe(self.p_x-1, self.p_y, self.color)) and (
                board.is_safe(self.p_x-2, self.p_y, self.color)):

                if self.color == -1:
                    return "White_Queen_Side_Castling"
                else:
                    return "Black_Queen_Side_Castling"
            else:
                return False
        else:
            return False
    
    def checked(self, board):
        return not board.is_safe(self.p_x, self.p_y, self.color)

    def state(self, board): # 체크, 체크메이트, 스테일메이트 판정
        check = self.checked(board) # 체크 상태인가
        next_turn_check = not board.can_defend(self.p_x, self.p_y, self.color) # 어떤 기물을 움직이든 체크가 되는 상태인가
        
        if check and not next_turn_check: # 체크
            return "Check"
        elif check and next_turn_check: # 체크메이트
            return "Checkmate"
        elif not check and next_turn_check: # 스테일메이트
            return "Stalemate"
        else:
            return False
