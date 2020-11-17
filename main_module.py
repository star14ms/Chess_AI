import numpy

class Board:
    def __init__(self):
        self.board = numpy.zeros((8,8), dtype=int)
        
    def delete(self, x, y):#x,y 좌표의 말 삭제
        self.board[x][y] = 0
        
    def insert(self, x, y, horse):#x, y좌표의 말 입력
        self.board[x][y] = horse
        
    def say(self, x, y):#x, y좌표의 말 클래스 출력
        return self.board[x][y]

class Horse:#말 정의하는 부모클래스 -> 폰, 킹, 나이트 등은 자식클래스가 됨
    def __init__(self, x, y, c):
        self.p_x = x
        self.p_y = y
        self.color = c#0이 백, 1이 흑

class Pawn(Horse):#폰
    def __init__(self, x, y):
        p_x = x
        p_y = y
        first_turn = True
        
    def move(self):#폰의 이동
        if first_turn == True:
            p_y += 2
            first_turn == False
            
        else:
            p_y += 1
            
        if (Board.board[p_x][p_y] != 0):
            


     


class Rook(Horse):#룩
    def __init__(self):
        pass
    
class Knight(Horse):#나이트
    def __init__(self):
        pass
    
class Bishop(Horse):#비숍
    def __init__(self):
        pass
    
class King(Horse):#킹
    def __init__(self):
        pass
    
class Queen(Horse):#퀸
    def __init__(self):
        pass
