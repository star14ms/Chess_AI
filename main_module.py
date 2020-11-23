from numpy import zeros

class Board:
    def __init__(self, f):
        self.front = f #플레이어가 플레이 할 색깔. False면 흰색 폰을 움직였을 때 +방향으로 나아가고 흑색 폰을 움직이면 -방향으로 나아간다.
        self.board = zeros((8,8), dtype=Empty)#int로 하면 나중에 클래스 insert할 때 오류남
        self.history = []#기보 기록
        
    def delete(self, x, y):#x,y 좌표의 말 삭제
        self.board[y][x] = 0

    def pos(self, x, y):#x, y좌표의 말 클래스 출력
        return self.board[y][x]

    def insert(self, x, y, horse):#x, y좌표의 말 입력
        self.board[y][x] = horse
        
    def move(self, x1, y1, x2, y2):#(x1, y1) -> (x2, y2)로 말의 이동. 색깔 상관없이 기존 (x2, y2)의 말을 지워버리니 주의할 것
        self.insert(x2, y2, self.pos(x1, y1))
        self.delete(x1, y1)
        self.history.append(self.pos(x1,y1))#(x1,y1)좌표의 말을 기보에 기록한다.

    def killable(self, x1, y1, x2, y2):#(x2,y2)에 말이 있고 색깔이 다르면 True, 아니면 False 출력
        if (self.pos(x2,y2) != 0) and (self.pos(x1,y1).color != self.pos(x2, y2).color) : return True
        
class Horse:#말 정의하는 부모클래스 -> 폰, 킹, 나이트 등은 자식클래스가 됨
    p_x = 0
    p_y = 0
    color = None # -1 -> 백, 1 -> 흑

    def move(self, board, x2, y2):
        if not self.moveable(board, x2, y2) : return False

        if board.killable(self.p_x, self.p_y, x2, y2) :#인공지능 활용을 위해 남겨둠
            board.move(self.p_x, self.p_y, x2, y2)
            self.p_x = x2
            self.p_y = y2
        else:
            board.move(self.p_x, self.p_y, x2, y2)
            self.p_x = x2
            self.p_y = y2
        return True


class Empty:
    def __repr__(self): #해당 클래스 호출 시 출력하는 것
        return 0


class Pawn(Horse):#폰
    def __init__(self, board, x, y, c):
        self.p_x = x
        self.p_y = y
        self.color = c
        self.first_turn = True
        board.insert(x, y, self)
        self.promotionable = False
        

    def moveable(self, board, x2, y2):
        enp = False
        if (not 0 <= x2 <= 7 or not 0 <= y2 <= 7): return False#좌표값 체크
         #(x2,y2-1)좌표의 말의 색이 다르고, 폰이면 앙파상 가능

        if (board.front == self.color):#플레이어 폰
            if board.pos(x2,y2-1) != 0 and board.pos(x2,y2-1).color != self.color and type(board.pos(x2,y2-1)) == Pawn and board.pos(x2,y2) == 0: enp = True
            if self.first_turn == True:
                if ((self.p_x == x2) and (1 <= y2 - self.p_y <=2)): self.first_turn = False
                else: return False
            else:
                if ((self.p_x == x2) and self.p_y == y2-1): pass
                elif enp == True : pass
                else : return False
        else:#상대방 폰
            if board.pos(x2,y2+1) != 0 and board.pos(x2,y2+1).color != self.color and type(board.pos(x2,y2+1)) == Pawn and board.pos(x2,y2) == 0: enp = True
            if self.first_turn == True:
                if ((self.p_x == x2) and (-2 <= y2 - self.p_y <= -1)): self.first_turn = False
                else : return False
            else:
                if ((self.p_x == x2) and self.p_y == y2+1): pass
                elif enp == True : pass
                else : return False
        
    def move(self, board, x2, y2):
        if not self.moveable(board, x2, y2) : return False

        if board.killable(self.p_x, self.p_y, x2, y2) :#인공지능 활용을 위해 남겨둠
            board.move(self.p_x, self.p_y, x2, y2)
            self.p_x = x2
            self.p_y = y2
        elif board.killable(self.p_x, self.p_y, x2, y2-1):
            board.move(self.p_x, self.p_y, x2, y2)
            board.delete(x2, y2-1)
            self.p_x = x2
            self.p_y = y2
        else:
            board.move(self.p_x, self.p_y, self.p_x, y2)
            self.p_x = x2
            self.p_y = y2
        return True


class Bishop(Horse):#비숍
    def __init__(self, board, x, y, c):
        self.p_x = x
        self.p_y = y
        self.color = c
        board.insert(x, y, self)

    def moveable(self, board, x2, y2):
        #체크 해야할 조건 :
        #(x2, y2)범위 조건
        #대각선 조건
        #(x2,y2)에 같은 색의 말이 아닌 조건
        #가는 길을 다른 말이 막지 않는 조건
        if (not 0 <= x2 <= 7 or not 0 <= y2 <= 7 or not abs(p_x - x2) == abs(p_y - y2) or board.pos(x2,y2).color == self.color): return False#(x2,y2)범위, 대각선 조건, (x2,y2)에 같은 색의 말이 아닌 조건 체크

        amount = abs(x2 - p_x) #거리
        if x2-p_x < 0 : lr = 1
        else : lr = -1
        if y2-p_x < 0 : ud = 1
        else : ud = -1

        for i in range(1, amount+1):#가는 길을 다른 말이 막지 않을 조건
            x3 = i * lr
            y3 = -(i * ud)
            if (self.pos(x3, y3) != 0): return False
            else: break
        
        return True # 모든 조건을 검사했으니, True 출력


class Rook(Horse):#룩
    def __init__(self, board, x, y, c):
        self.p_x = x
        self.p_y = y
        self.color = c
        board.insert(x, y, self)

    def moveable(self, board, amount, direction): #amount : 움직이는 양, direction : 방향(오른쪽 0, 왼쪽 1, 위 2, 아래 3)
        x2 = p_x
        y2 = p_y
        if(x2 == p_x and y2 == p_y) : return False
        elif (x2 - p_x != 0  and y2 - p_x != 0): return False
        else :
            if direction == 0 : #오른쪽으로 이동할 경우
                for i in range(1, amount):  # 한칸씩 다른 물체가 있는 지 확인
                  x2 = x2+i
                  if pos(x2,y2) != 0 or x2>=8 or x2<0: # 룩이 이동 범위가 좌표를 벗어나거나 이동하는 좌표 사이에 말이 있으면 갈 수 없다.
                    x2 = x2-i
                    return False
                  else: break
          
            elif direction == 1 : #왼쪽으로 이동할 경우
                for i in range(1, amount):
                  x2 = x2-i
                  if pos(x2,y2) != 0 or x2>=8 or x2<0:
                    x2 = x2+i
                    return False
                  else: break
            elif direction == 2 : #위쪽으로 이동할 경우
                for i in range(1, amount):
                  y2 = y2+i
                  if pos(x2,y2) != 0 or y2>=8 or y2<0:
                    y2 = y2-i
                    return False
                  else: break
      
            elif direction == 3 : #아래쪽으로 이동할 경우
                for i in range(1, amount):
                  y2 = y2-i
                  if pos(x2,y2) != 0 or y2>=8 or y2<0:
                    y2 = y2+i
                    return False
                  else: break

            return True


class Knight(Horse):#나이트
    def __init__(self, board, x, y, c):
        self.x = x
        self.y = y
        self.color = c
        board.insert(x, y, self)
    
    def moveable(self, board, x2, y2):
        
        # 나이트 기본 행마
        if (board.pos(x2, y2) != 0) and (board.pos(x2, y2).color == self.color):
            return False
        else :
            if (x2-self.x == 2) or (x2-self.x == -2): # 동쪽 or 서쪽으로 2칸일 때
                if (y2-self.y != 1) and (y2-self.y != -1): return False # 남쪽 or 북쪽으로 1칸이 아니면, 이동 실패
                else : return True
            elif (y2-self.y == 2) or (y2-self.y == -2): # 남쪽 or 북쪽으로 2칸일 때
                if (x2-self.x != 1) and (x2-self.x != -1): return False # 동쪽 or 서쪽으로 1칸이 아니면, 이동 실패
                else : return True
            else:
                return False
        
        

class Queen(Horse):#퀸
    def __init__(self, board, x, y, c):
        self.p_x = x
        self.p_y = y
        self.color = c
        board.insert(x, y, self)

    def moveable(self, board, amount, lr, ud): #amount : 움직이는 양, lr : 좌우(좌 : -1, 우 : +1), ud : 위아래(아래 : -1, 위 : +1)
        x2 = self.p_x + amount * lr
        y2 = self.p_y + amount * ud
        if(x2 - p_x != 0 and y2 - p_x != 0) : return False
        elif(x2 == p_x and y2 == p_y) : return False
        elif(x2 - p_x != y2 - p_y) : return False
        else :
            for i in range(1, amount+1):#킹이 이동하는 좌표 사이에 말이 있으면 그 좌표로 갈 수 없다.
                x3 = self.p_x + i*lr
                y3 = self.p_y + i*ud
                if (0 <= x3 <= 7) and (0 <= y3 <= 7):
                    if (pos(x3, y3) != 0): return False
                else: break
            return True
        


class King(Horse):#킹
    def __init__(self, board, x, y, c):
        self.x = x
        self.y = y
        self.color = c
        self.moved = False # 캐슬링 조건 : 킹이 움직인 적이 없어야함
        board.insert(x, y, self)

    def moveable(self, board, x2, y2): # lr: 가만히 0, 오른쪽 1, 왼쪽 2 / ud: 가만히 0, 위쪽 1, 아래쪽 2
        
        # 킹 기본 행마
        if (((x2-self.x == -1) or (x2-self.x == +1)) and (-1 <= y2-self.y <= 1)) or (
            ((y2-self.y == -1) or (y2-self.y == +1)) and (-1 <= x2-self.x <= 1)):
            if (board.pos(x2, y2) != 0) and (board.pos(x2, y2).color == self.color): # 같은 색 기물이 있는 곳이면, 이동 실패
                return False  
            else : return True
        # 캐슬링
        elif (not self.moved):
            
            # 킹 사이드 캐슬링
            if (x2-self.x == -2) and (y2 == self.y) and (
                board.pos(self.x-1, self.y) == 0) and (board.pos(self.x-2, self.y) == 0) and (board.pos(self.x-3, self.y) == 0):
                if self.color == -1:
                    board.move(0, 7, 3, 7) # 룩도 이동
                else:
                    board.move(0, 0, 3, 0)
            
            # 퀸 사이드 캐슬링
            elif (x2-self.x == +2) and (y2 == self.y) and (
                board.pos(self.x+1, self.y) == 0) and (board.pos(self.x+2, self.y) == 0):
                if self.color == -1:
                    board.move(7, 7, 5, 7)
                else:
                    board.move(7, 0, 5, 0)
            else:
                return False
        else:
            return False
        
