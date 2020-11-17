from numpy import zeros

class Board:
    def __init__(self, f):
        self.front = f #플레이어가 플레이 할 색깔. False면 흰색 폰을 움직였을 때 +방향으로 나아가고 흑색 폰을 움직이면 -방향으로 나아간다.
        self.board = zeros((8,8), dtype=Empty)#int로 하면 나중에 클래스 insert할 때 오류남
        
    def delete(self, x, y):#x,y 좌표의 말 삭제
        self.board[x][y] = 0

    def pos(self, x, y):#x, y좌표의 말 클래스 출력
        a = self.board[x][y]
        
        if a ==0: return 0
        else :    return a

    def insert(self, x, y, horse):#x, y좌표의 말 입력
        self.board[x][y] = horse
        

    def move(self, x1, y1, x2, y2):#(x1, y1) -> (x2, y2)로 말의 이동. 색깔 상관없이 말의 위치를 지워버리니 주의할 것
        self.insert(x2, y2, self.pos(x1, y1))
        self.delete(x1, y1)

    def killable(self, x1, y1, x2, y2):#가고자 하는 위치에 말이 있고 색깔이 다르면 True, 아니면 False 출력
        return (self.pos(x2, y2) != 0) and (self.pos(x1,y1).color != self.pos(x2, y2).color)
        
class Horse:#말 정의하는 부모클래스 -> 폰, 킹, 나이트 등은 자식클래스가 됨
    p_x = 0
    p_y = 0
    color = 0 #False -> 백, True -> 흑

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
        
    def move(self, board, amount):#amount : 움직이는 양(최대 : 2) - 첫 턴에만 사용됨
        if (board.front == self.color):

            if self.first_turn == True:#첫턴시 두 칸 이동
                y2 = self.p_y + amount
                self.first_turn == False
                
            else:
                y2 = self.p_y + 1
        else:
            if self.first_turn == True:#첫턴시 두 칸 이동
                y2 = self.p_y - amount
                self.first_turn == False
                
            else:
                y2 = self.p_y - 1
            
        if board.killable(self.p_x, self.p_y, self.p_x, y2) :#인공지능 활용을 위해 남겨둠
            board.move(self.p_x, self.p_y, self.p_x, y2)
        else:
            board.move(self.p_x, self.p_y, self.p_x, y2)

class Bishop(Horse):#비숍
    def __init__(self, board, x, y, c):
        self.p_x = x
        self.p_y = y
        self.color = c
        self.first_turn = True
        board.insert(x, y, self)

    def move(self, board, amount, lr, ud): #amount : 움직이는 양, lr : 좌우, ud : 위아래(오른쪽, 위쪽을 향하면 1, 아니면 -1)
        x2 = p_x + lr * amount
        y2 = p_y + ud * amount

        for i in range(1, amount+1):#비숍이 이동하는 좌표 사이에 말이 있으면 그 좌표로 갈 수 없다.
            x3 = self.p_x + i*lr
            y3 = self.p_y + i*ud
            if (0 >= x3 >= 7) and (0 >= y3 >= 7):
                if (pos(x3, y3) != 0): return False
            else: break
            

        if board.killable(self.p_x, self.p_y, x2, y2) :#인공지능 활용을 위해 남겨둠
            board.move(self.p_x, self.p_y, x2, y2)
        else:
            board.move(self.p_x, self.p_y, x2, y2)



class Rook(Horse):#룩
    def __init__(self):
        pass
    
class Knight(Horse):#나이트
    def __init__(self):
        pass
    
class King(Horse):#킹
    def __init__(self):
        pass
    
class Queen(Horse):#퀸
    def __init__(self):
        pass
