class Board:
    def __init__(self, f):
        self.front = f #플레이어가 플레이 할 색깔. False면 흰색 폰을 움직였을 때 +방향으로 나아가고 흑색 폰을 움직이면 -방향으로 나아간다.
        self.board = [[0 for col in range(8)] for row in range(8)]
        for i in range(0, 8):
            for j in range(0, 8): self.board[i][j] = Empty()
        self.history = []#기보 기록
        
    def delete(self, x, y):#x,y 좌표의 말 삭제
        self.board[y][x] = Empty()

    def pos(self, x, y):#x, y좌표의 말 클래스 출력
        return self.board[y][x]

    def insert(self, x, y, horse):#x, y좌표의 말 입력
        self.board[y][x] = horse
        
    def move(self, x1, y1, x2, y2):#(x1, y1) -> (x2, y2)로 말의 이동. 색깔 상관없이 기존 (x2, y2)의 말을 지워버리니 주의할 것
        self.insert(x2, y2, self.pos(x1, y1))
        self.pos(x1,y1).horse_history.append((x1, y1))#자신의 좌표 튜플을 기록한다.
        self.history.append(self.pos(x1,y1))#(x1,y1)좌표의 말 클래스를 기보에 기록한다.
        self.delete(x1, y1)

    def killable(self, x1, y1, x2, y2):#(x2,y2)에 말이 있고 색깔이 다르면 True, 아니면 False 출력
        if (type(self.pos(x2,y2)) != Empty) and (self.pos(x1,y1).color != self.pos(x2, y2).color) : return True
        
    # def attack(self, x1, y1):
    #     if(Pawn.moveable(board, x1, y1)) : return False
    #     elif(Bishop.moveable(board,x1,y1)) : return False
    #     elif(Rook.moveable(board,x1,y1)) : return False
    #     elif(Knight.moveable(board,x1,y1)) : return False
    #     elif(Queen.moveable(board,x1,y1)) : return False
    #     elif(King.moveable(board,x1,y1)) : return False
        
    # def move_check(self,x1,y1,horse):
    #     Board.delete(x1,y1)
    #     if(King.check == True):
    #         Board.insert(x1,y1,horse)
    #         return False
        
    def being_attacked(self, x2, y2, color): # x2, y2 자리가 공격받는 자리인가 판단
        
        board2 = Board(-1) # 보드 복사하기 (시뮬레이션용)
        board2.board = self.board
        xy2_horse = board2.pos(x2, y2)
        Pawn(board2, x2, y2, color) # x2, y2 좌표의 아군 말이 먹히고 상대 말이 와 있다고 가정한다.
        
        for y1 in range(8): # 보드 전체를 검사해
            for x1 in range(8): # x2, y2 좌표에 있는 말을 잡을 수 있는 상대 기물이 있는지 검사
                if (board2.pos(x1, y1).color == color*-1) and (board2.pos(x1, y1).moveable(board2, x2, y2)): ### self.board -> self
                    board2.insert(x2, y2, xy2_horse)
                    return True
        
        board2.insert(x2, y2, xy2_horse)
        return False

    def can_defend(self, x, y, color): # x, y 자리를 방어할 수 있는지 검사
        
        board2 = Board(-1) # 보드 복사하기 (시뮬레이션용)
        board2.board = self.board
        dangerous = True

        for y1 in range(8): # 보드 전체를 검사해 내 기물들을 찾고
            for x1 in range(8): 
                if (self.pos(x1, y1).color == color):
                     
                    for y2 in range(8): 
                        for x2 in range(8):

                            xy1_horse = board2.pos(x1, y1) # 움직일 기물 백업
                            xy2_horse = board2.pos(x2, y2) # 먹히는 기물 백업

                            try:
                                if board2.pos(x1, y1).moveable(board2, x2, y2): # 그 기물이 움직일 수 있는 모든 경우를 둬본다

                                    # print(board2.pos(x1, y1), "가", x2, y2, "로 이동가능", end=" ")
                                    board2.insert(x2, y2, board2.pos(x1, y1))
                                    board2.delete(x1, y1)
    
                                    # 그 기물이 움직였을때 x, y에 있는 기물이 안전해질 수 있는 경우가 있나 검사
    
                                    if (x == x1) and (y == y1): # 1. 공격받는 자신이 직접 자리를 피하기
                                        dangerous2 = board2.being_attacked(x2, y2, color)
                                    else:                       # 2. 다른 기물을 움직여 방패세우기
                                        dangerous2 = board2.being_attacked(x, y, color) ### self. -> board2.
    
                                    board2.insert(x1, y1, xy1_horse) # 보드 원상태로
                                    board2.insert(x2, y2, xy2_horse)
    
                                    if not dangerous2:
                                        # print(x, y, "방어 가능!")
                                        dangerous = False
                                    # else:
                                        # print()
                            except:
                                print(x1, y1, x2, y2)
        if not dangerous:
            return True
        return False
                    

class Horse:#말 정의하는 부모클래스 -> 폰, 킹, 나이트 등은 자식클래스가 됨
    
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
    
    def move(self, board, x2, y2):
        
        movable = self.moveable(board, x2, y2)
        if not movable:
            return False
        
        # 움직였을 때 자신의 왕이 체크당하는지 확인
        board2 = Board(-1) # 보드 복사하기 (시뮬레이션용)
        board2.board = board.board 
        xy1_horse = board2.pos(self.p_x, self.p_y) # 움직일 기물 백업
        xy2_horse = board2.pos(x2, y2)             # 먹히는 기물 백업
        board2.insert(x2, y2, board2.pos(self.p_x, self.p_y)) # 말을 시험삼아 움직여본다
        board2.delete(self.p_x, self.p_y)

        for y in range(8):
            for x in range(8):
                if (type(board2.pos(x, y)) == King) and (board2.pos(x, y).color == self.color):
                    
                    # 킹이 체크가 되면
                    if board2.being_attacked(x, y, self.color):
                        board2.insert(self.p_x, self.p_y, xy1_horse) # 보드 원상태로
                        board2.insert(x2, y2, xy2_horse)
                        return False
        
        board2.insert(self.p_x, self.p_y, xy1_horse) # 보드 원상태로
        board2.insert(x2, y2, xy2_horse)
        
        # 말 움직이기
        if movable == 'enp':
            if board.killable(self.p_x, self.p_y, x2, y2-1):
                board.move(self.p_x, self.p_y, x2, y2)
                board.delete(x2, y2-1)
                self.p_x = x2
                self.p_y = y2
            elif board.killable(self.p_x, self.p_y, x2, y2+1):
                board.move(self.p_x, self.p_y, x2, y2)
                board.delete(x2, y2+1)
                self.p_x = x2
                self.p_y = y2
        elif board.killable(self.p_x, self.p_y, x2, y2) :#인공지능 활용을 위해 남겨둠
            board.move(self.p_x, self.p_y, x2, y2)
            self.p_x = x2
            self.p_y = y2
        else:
            board.move(self.p_x, self.p_y, x2, y2)
            self.p_x = x2
            self.p_y = y2
        
        # 캐슬링 시, 룩도 이동
        if type(self) == King:
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


class Empty:
    color = 0
    horse_history = [(-1,-1)]


class Pawn(Horse):#폰
    
    def moveable(self, board, x2, y2):
        if (not 0 <= x2 <= 7 or not 0 <= y2 <= 7): return False#좌표값 체크
         #(x2,y2-1)좌표의 말의 색이 다르고, 폰이면 앙파상 가능

        if board.front == self.color:
            if self.p_x == x2 and type(board.pos(x2,y2)) == Empty:#앞으로 움직임
                if self.p_y == 6 and (1 <= self.p_y - y2 <=2) : return True  #첫 턴, 움직임
                elif self.p_y - y2 == 1 : return True# 1>턴, 움직임

            elif abs(self.p_x - x2) == 1 and self.p_y - y2 == 1 and type(board.pos(x2,y2)) != Empty and board.pos(x2, y2).color != self.color : return True#공격
            elif y2 < 7 and board.pos(x2,y2+1).horse_history[0][1] == 1 and abs(self.p_x - x2) == 1 and self.p_y - y2 == 1 and type(board.pos(x2,y2)) == Empty:
                print('enp!')
                return 'enp'

        else:
            if self.p_x == x2 and type(board.pos(x2,y2) == Empty):#앞으로 움직임
                if self.p_y == 1 and (-2 <= self.p_y - y2 <= -1) : return True #첫 턴, 움직임
                elif self.p_y - y2 == -1 : return True# 1>턴, 움직임
            elif abs(self.p_x - x2) == 1 and self.p_y - y2 == -1 and type(board.pos(x2,y2)) != Empty and board.pos(x2, y2).color != self.color : return True#공격
            elif y2 > 0 and board.pos(x2,y2-1).horse_history[0][1] == 6 and abs(self.p_x - x2) == 1 and self.p_y - y2 == -1 and type(board.pos(x2,y2)) == Empty :
                print('enp!')
                return 'enp'
        return False

        
    # def move(self, board, x2, y2):
    #     tf = self.moveable(board, x2, y2)
    #     if tf == False : return False

    #     if board.front == self.color and tf == 'enp':
    #         if board.killable(self.p_x, self.p_y, x2, y2-1):
    #             board.move(self.p_x, self.p_y, x2, y2)
    #             board.delete(x2, y2-1)
    #             self.p_x = x2
    #             self.p_y = y2
    #         elif board.killable(self.p_x, self.p_y, x2, y2+1):
    #             board.move(self.p_x, self.p_y, x2, y2)
    #             board.delete(x2, y2+1)
    #             self.p_x = x2
    #             self.p_y = y2
    #     elif board.killable(self.p_x, self.p_y, x2, y2) :#인공지능 활용을 위해 남겨둠
    #         board.move(self.p_x, self.p_y, x2, y2)
    #         self.p_x = x2
    #         self.p_y = y2
    #     else:
    #         board.move(self.p_x, self.p_y, self.p_x, y2)
    #         self.p_x = x2
    #         self.p_y = y2
    #     return True


class Bishop(Horse):#비숍

    def moveable(self, board, x2, y2):
        #체크 해야할 조건 :
        #(x2, y2)범위 조건
        #대각선 조건
        #(x2,y2)에 같은 색의 말이 아닌 조건
        #가는 길을 다른 말이 막지 않는 조건
        if (not 0 <= x2 <= 7 or not 0 <= y2 <= 7 or not abs(self.p_x - x2) == abs(self.p_y - y2) or board.pos(x2,y2).color == self.color): return False#(x2,y2)범위, 대각선 조건, (x2,y2)에 같은 색의 말이 아닌 조건 체크

        amount = abs(x2 - self.p_x) #거리
        if x2-self.p_x < 0 : lr = -1 #lr : -1, +1 오른쪽 위면 lr = 1, du = -1
        else : lr = 1
        if y2-self.p_y < 0 : du = -1 #ud : -1, +1
        else : du = 1
        for i in range(1, amount+1):#가는 길을 다른 말이 막지 않을 조건
            x3 = self.p_x + i * lr
            y3 = self.p_y + i * du
            if (type(board.pos(x3, y3)) != Empty):
                # print(x3, y3)
                return False
            else: break
        return True # 모든 조건을 검사했으니, True 출력


class Rook(Horse):#룩
    
    def moveable(self, board, x2, y2):
        if (not 0 <= x2 <= 7 or not 0 <= y2 <= 7 or board.pos(x2,y2).color == self.color): return False
        if x2-self.p_x == y2-self.p_y == 0 : return False

        if (x2 == self.p_x):
            am = abs(self.p_y - y2)#거리
            for i in range(1, am):
                if self.p_y < y2:
                    if type(board.pos(x2, self.p_y+i)) != Empty : return False
                else :
                    # print(board.pos(x2, self.p_y-i))
                    if type(board.pos(x2, self.p_y-i)) != Empty : return False
        elif (y2-self.p_y == 0):
            am = abs(self.p_x - x2)#거리
            for i in range(1, am):
                if self.p_x < x2:
                    if type(board.pos(self.p_x+i, y2)) != Empty : return False
                else :
                    if type(board.pos(self.p_x-i, y2)) != Empty : return False
        else: return False
        return True


class Knight(Horse):#나이트
    
    def moveable(self, board, x2, y2):
        
        # 나이트 기본 행마
        if (x2-self.p_x == 2) or (x2-self.p_x == -2): # 동쪽 or 서쪽으로 2칸일 때
            if (y2-self.p_y != 1) and (y2-self.p_y != -1): return False # 남쪽 or 북쪽으로 1칸이 아니면, 이동 실패
        elif (y2-self.p_y == 2) or (y2-self.p_y == -2): # 남쪽 or 북쪽으로 2칸일 때
            if (x2-self.p_x != 1) and (x2-self.p_x != -1): return False # 동쪽 or 서쪽으로 1칸이 아니면, 이동 실패
        else:
            return False

        if (not 0 <= x2 <= 7 or not 0 <= y2 <= 7) or (board.pos(x2, y2).color == self.color): # 좌표 범위 밖이거나, 우리편과 겹치나 검사
            return False

        return True


class Queen(Horse):#퀸
        
    def moveable(self, board, x2, y2):

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


class King(Horse):#킹
    
    def moveable(self, board, x2, y2):

        # 킹 기본 행마
        if (((x2-self.p_x == -1) or (x2-self.p_x == +1)) and (-1 <= y2-self.p_y <= 1)) or (
            ((y2-self.p_y == -1) or (y2-self.p_y == +1)) and (x2-self.p_x == 0)):
            if (not 0 <= x2 <= 7 or not 0 <= y2 <= 7) or (board.pos(x2, y2).color == self.color): 
                return False # 좌표 범위 밖이거나, 우리편과 겹치나 조사
            else:
                return True

        # 캐슬링
        elif (not self.moved):
            
            # 킹 사이드 캐슬링
            if (x2-self.p_x == +2) and (y2 == self.p_y) and (
                type(board.pos(self.p_x+3, self.p_y)) == Rook and not board.pos(self.p_x+3, self.p_y).moved) and (
                type(board.pos(self.p_x+1, self.p_y)) == Empty) and (type(board.pos(self.p_x+2, self.p_y)) == Empty) and (
                not self.checked(board)) and (
                not board.being_attacked(self.p_x+1, self.p_y, self.color)) and (
                not board.being_attacked(self.p_x+2, self.p_y, self.color)):
                
                if self.color == -1:
                    return "White_King_Side_Castling"
                else:
                    return "Black_King_Side_Castling"

            # 퀸 사이드 캐슬링
            elif (x2-self.p_x == -2) and (y2 == self.p_y) and (
                type(board.pos(self.p_x-4, self.p_y)) == Rook and not board.pos(self.p_x-4, self.p_y).moved) and (
                type(board.pos(self.p_x-1, self.p_y)) == Empty) and (type(board.pos(self.p_x-2, self.p_y)) == Empty) and (type(board.pos(self.p_x-3, self.p_y)) == Empty) and (
                not self.checked(board)) and (
                not board.being_attacked(self.p_x-1, self.p_y, self.color)) and (
                not board.being_attacked(self.p_x-2, self.p_y, self.color)):

                if self.color == -1:
                    return "White_Queen_Side_Castling"
                else:
                    return "Black_Queen_Side_Castling"
            else:
                return False
        else:
            return False
    
    def checked(self, board):
        if board.being_attacked(self.p_x, self.p_y, self.color):
            return True # 체크이다
        else:
            return False # 체크가 아니다
    
    def checkmate(self, board): # 체크, 체크메이트, 스테일메이트 판정
        
        # # 왕 주변 8곳이 공격받거나 막혀있을 때
        # if ((not 0 <= self.p_x-1) or board.being_attacked(self.p_x-1, self.p_y, self.color) or not self.moveable(board, self.p_x-1, self.p_y)) and (
        #     (not self.p_x+1 <= 7) or board.being_attacked(self.p_x+1, self.p_y, self.color) or not self.moveable(board, self.p_x+1, self.p_y)) and (
        #     (not 0 <= self.p_y-1) or board.being_attacked(self.p_x, self.p_y-1, self.color) or not self.moveable(board, self.p_x, self.p_y-1)) and (
        #     (not self.p_y+1 <= 7) or board.being_attacked(self.p_x, self.p_y+1, self.color) or not self.moveable(board, self.p_x, self.p_y+1)) and (
        #     (not 0 <= self.p_x-1) or (not 0 <= self.p_y-1) or board.being_attacked(self.p_x-1, self.p_y-1, self.color) or not self.moveable(board, self.p_x-1, self.p_y-1)) and (
        #     (not 0 <= self.p_x-1) or (not self.p_y+1 <= 7) or board.being_attacked(self.p_x-1, self.p_y+1, self.color) or not self.moveable(board, self.p_x-1, self.p_y+1)) and (
        #     (not self.p_x+1 <= 7) or (not 0 <= self.p_y-1) or board.being_attacked(self.p_x+1, self.p_y-1, self.color) or not self.moveable(board, self.p_x+1, self.p_y-1)) and (
        #     (not self.p_x+1 <= 7) or (not self.p_y+1 <= 7) or board.being_attacked(self.p_x+1, self.p_y+1, self.color) or not self.moveable(board, self.p_x+1, self.p_y+1)):
        
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

    # def checkmate(self,board):
    #     if(board.attack(board,self.p_x,self.p_y) == False): # 왕을 제외한 나머지 기물은 모두 이동불가이거나 모두 잡혀있을 떄의 조건 추가해야함
    #         if(self.p_x < 7 and board.attack(board,self.p_x+1,self.p_y) == False):
    #             if(self.p_x > 0 and board.attack(board,self.p_x-1,self.p_y) == False):
    #                 if(self.p_y < 7 and board.attack(board,self.p_x,self.p_y+1) == False):
    #                     if(self.p_x > 0 and board.attack(board,self.p_x,self.p_y-1) == False):
    #                         if(self.p_x < 7 and self.p_y < 7 and board.attack(board,self.p_x+1,self.p_y+1) == False):
    #                             if(self.p_x < 7 and self.p_y >0 and board.attack(board,self.p_x+1,self.p_y-1) == False):
    #                                 if(self.p_x > 0 and self.p_y < 7 and board.attack(board,self.p_x-1,self.p_y+1) == False):
    #                                     if(self.p_x < 7 and self.p_y < 7 and board.attack(board,self.p_x+1,self.p_y+1) == False):
    #                                         return True # 체크 메이트 상태이다.